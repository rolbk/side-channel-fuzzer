"""
File: Test Case Generation

Copyright (C) Microsoft Corporation
SPDX-License-Identifier: MIT
"""
from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, List, Tuple, Optional, Final, Callable, Dict, TextIO, Iterable
from subprocess import CalledProcessError, run
from copy import deepcopy
from abc import ABC, abstractmethod

from .tc_components.actor import Actor
from .tc_components.instruction import Instruction, RegisterOp, FlagsOp, MemoryOp, \
    ImmediateOp, AgenOp, LabelOp, CondOp, AnyOperand
from .tc_components.test_case_code import TestCaseProgram, Function, BasicBlock, CodeSection, \
    TC_EXIT_LABEL
from .instruction_spec import OT
from .logs import GeneratorLogger, error, inform
from .config import CONF, ActorsConf

if TYPE_CHECKING:
    from .tc_components.test_case_code import InstructionNode
    from .target_desc import TargetDesc
    from .asm_parser import AsmParser
    from .instruction_spec import InstructionSpec, OperandSpec
    from .isa_spec import InstructionSet
    from .elf_parser import ELFParser


# ==================================================================================================
# Interfaces and common functionality of ISA-specific service classes
# ==================================================================================================
class Pass(ABC):
    """
    Interface to an instrumentation pass that modifies a generated test case
    """

    @abstractmethod
    def run_on_test_case(self, test_case: TestCaseProgram) -> None:
        """
        Run the pass on all instructions in a given test case
        """


class Printer(ABC):
    """
    Interface to an ISA-specific assembly printer; that is, a class that prints
    a valid assembly representation of a test case
    """

    prologue_template: List[str]
    """ List of lines that must be printed at the beginning of the assembly file """

    epilogue_template: List[str]
    """ List of lines that must be printed at the end of the assembly file """

    def __init__(self, target_desc: TargetDesc) -> None:
        self.target_desc = target_desc

    def print(self, test_case: TestCaseProgram) -> None:
        """
        Print the assembly representation of a test case to a file associated with the test case
        (i.e., test_case.asm_file)

        :param test_case: The test case to print
        """
        with open(test_case.asm_path(), "w") as f:
            for line in self.prologue_template:
                f.write(line)

            for section in test_case:
                self._print_section(section, f)

            for line in self.epilogue_template:
                f.write(line)

            # trailing raw asm (indirect-call jump tables), emitted after the code
            for line in getattr(test_case, "appended_asm", None) or []:
                f.write(line if line.endswith("\n") else line + "\n")

    def _print_section(self, sec: CodeSection, file_: TextIO) -> None:
        file_.write(f".section .data.{sec.name}\n")
        for func in sec:
            coll = getattr(func, "coll_section", None)
            if coll:
                file_.write(f".section {coll}\n")
            self._print_function(func, file_)
            if coll:
                file_.write(f".section .data.{sec.name}\n")

    def _print_function(self, func: Function, file_: TextIO) -> None:
        file_.write(f"{func.name}:\n")
        for bb in func:
            self._print_basic_block(bb, file_)

        self._print_basic_block(func.get_exit_bb(), file_)

    def _print_basic_block(self, bb: BasicBlock, file_: TextIO) -> None:
        file_.write(f"{bb.name.lower()}:\n")
        for inst in bb:
            file_.write(self._instruction_to_str(inst) + "\n")
        for inst in bb.terminators:
            file_.write(self._instruction_to_str(inst) + "\n")

    @abstractmethod
    def _instruction_to_str(self, inst: Instruction) -> str:
        """ Convert an instruction object to its assembly representation """

    @abstractmethod
    def _operand_to_str(self, op: AnyOperand) -> str:
        """ Convert an operand object to its assembly representation """

    @abstractmethod
    def _macro_to_str(self, inst: Instruction) -> str:
        """ Convert a macro instruction object to its assembly representation """


# ==================================================================================================
# ISA-independent Code Generator
# ==================================================================================================
class CodeGenerator(ABC):
    """
    ISA-independent implementation of the class responsible for generating test case code.

    Some of the methods are abstract and must be implemented by the ISA-specific subclasses.
    """

    _instruction_set: InstructionSet  # Specification of the tested instruction set
    _target_desc: TargetDesc  # Description of the tested architecture
    _asm_parser: AsmParser  # Parser for assembly files
    _elf_parser: ELFParser  # Parser for ELF files
    _passes: List[Pass]  # List of passes to run on the generated test case; set by subclasses
    _printer: Printer  # Printer for the generated test case; set by subclasses

    _state: int = 0  # Current seed value
    _cached_template: Optional[TestCaseProgram] = None  # Parsed template assembly file

    _function_generator: Final[_FunctionGenerator]
    _instruction_generator: Final[_InstructionGenerator]

    __log: Final[GeneratorLogger]

    def __init__(self, seed: int, instruction_set: InstructionSet, target_desc: TargetDesc,
                 asm_parser: AsmParser, elf_parser: ELFParser) -> None:
        self._instruction_set = instruction_set
        self._target_desc = target_desc
        self._asm_parser = asm_parser
        self._elf_parser = elf_parser
        self._set_seed(seed)

        self.__log = GeneratorLogger()
        self.__log.dbg_dump_instruction_pool(instruction_set.instructions)
        self._passes = []

        self._function_generator = _FunctionGenerator(self._target_desc, instruction_set)
        self._instruction_generator = _InstructionGenerator(self._target_desc)

    # ----------------------------------------------------------------------------------------------
    # Public Interface
    def create_test_case(self, asm_file: str, disable_assembler: bool = False) -> TestCaseProgram:
        """
        Generate a random test case, write its assembly code to a file,
        and assemble it into an object (unless disabled).
        :param asm_file: the path to the output file
        :param disable_assembler: if True, the function will not assemble the test case
        :return: the generated test case object
        """
        if not asm_file:
            asm_file = 'generated.asm'
        test_case = TestCaseProgram(asm_file, seed=self._state)

        # create actors and their corresponding sections
        actors_config: ActorsConf = CONF.get_actors_conf()
        if len(actors_config) != 1:
            error("Generation of test cases with multiple actors is not yet supported")
        self.generate_actors_with_sections(test_case, actors_config)

        # create the functions and fill them with random instructions (function 0 is the entry)
        main_section = test_case[0]
        default_actor = main_section.owner
        assert default_actor.is_main
        n_funcs = random.randint(1, CONF.max_functions)
        funcs = [
            self._function_generator.generate_empty(
                f".function_{i}", main_section, is_entry=(i == 0)) for i in range(n_funcs)
        ]
        for func in funcs:
            self._function_generator.fill_function(func)

        # seed rax with the training-loop counter (r13) at entry, so runs vary per iteration
        if CONF.seed_rax_loop_var:
            self._function_generator.seed_loop_var(funcs[0])

        # raw asm appended after the epilogue (indirect-call jump tables)
        test_case.appended_asm = []  # type: ignore[attr-defined]

        # wire a forward-only call graph; no-op when n_funcs == 1
        self._function_generator.wire_call_graph(funcs, test_case.appended_asm)  # type: ignore
        self._function_generator.wire_indirect_jmps(funcs, test_case.appended_asm)  # type: ignore
        self._function_generator.perturb_return_stack(funcs)
        self._function_generator.assign_collision_sections(funcs)

        # add them to the test case, in the first section
        for func in funcs:
            test_case[0].append(func)

        # process the test case
        for p in self._passes:
            p.run_on_test_case(test_case)

        # add symbols to test case
        self._add_required_symbols(test_case)

        self._printer.print(test_case)

        if disable_assembler:
            return test_case

        test_case.assign_obj(asm_file[:-4] + ".o")
        assemble(test_case)
        self._elf_parser.populate_elf_data(test_case.get_obj(), test_case)

        self._update_state()
        return test_case

    def create_test_case_from_template(self, template_file: str) -> TestCaseProgram:
        """
        Generate a test case based on a template by expanding RANDOM_* macros.
        Run instrumentation _passes and print the result into a file

        :param template_file: The path to the template file
        :return: The generated test case object
        :raises FileNotFoundError: if the template file does not exist
        :raises CalledProcessError: if the assembler fails to assemble the test case
        """
        # create a TestCaseProgram object from the template file
        if self._cached_template:
            test_case = deepcopy(self._cached_template)
            test_case.generator_seed = self._state
        else:
            test_case = self._asm_parser.parse_file(
                template_file, self, self._elf_parser, is_template=True)
            test_case.generator_seed = self._state
            self._cached_template = deepcopy(test_case)

        # Label all instructions from the template as such
        for func in test_case.iter_functions():
            for bb in func:
                for instr in bb:
                    instr.is_from_template = True

        # Expand the template
        self._set_seed(self._state)  # reset the seed in case it was updated by other modules
        self._expand_template(test_case, CONF.get_actors_conf())
        for p in self._passes:
            p.run_on_test_case(test_case)

        # Print into assembly and assemble into an object file
        asm_file = 'generated.asm'
        test_case.reassign_asm_file(asm_file)
        self._printer.print(test_case)

        test_case.assign_obj(asm_file[:-4] + ".o")
        assemble(test_case)
        self._elf_parser.populate_elf_data(test_case.get_obj(), test_case)

        self._update_state()
        return test_case

    def generate_actors_with_sections(self, test_case: TestCaseProgram,
                                      actors_dict: ActorsConf) -> None:
        """
        Stand-alone interface to create actors for the given test case and
        populate them with the corresponding sections.

        NOTE: This method leaves the sections *empty*; i.e., it does not populate the test case
        with functions, basic blocks, and instructions.

        :param test_case: The test case to which the actors will be added
        :param actors_dict: The configuration of the actors
        :return: None
        """
        for name, actor_dict in actors_dict.items():
            actor = Actor.from_dict(actor_dict, self._target_desc)

            # add the actor to the test case
            if name == "main":  # the main actor is created by default; overwrite it
                test_case.add_actor_with_section(actor, allow_overwrite=True)
            else:  # all other actors should not exist yet
                test_case.add_actor_with_section(actor)

    def generate_instruction(self,
                             spec: InstructionSpec,
                             is_instrumentation: Optional[bool] = None) -> Instruction:
        """
        Stand-alone interface to generate a random instruction based on the specification.

        :param spec: The specification of the instruction
        :return: The generated instruction
        """
        # To correctly inherit the default value of is_instrumentation from the instruction
        # generator, we have two separate calls
        if is_instrumentation is None:
            return self._instruction_generator.generate(spec)
        return self._instruction_generator.generate(spec, is_instrumentation)

    # ----------------------------------------------------------------------------------------------
    # Private: Seed Management
    def _set_seed(self, seed: int) -> None:
        """
        Set the seed value used to generate test programs
        :param seed: The seed value
        """
        if seed == 0:
            seed = random.randint(1, 1000000)
            inform("prog_gen", f"Setting program_generator_seed to random value: {seed}")
        self._state = seed
        random.seed(self._state)

    def _update_state(self) -> None:
        self._state += 1
        random.seed(self._state)

    # ----------------------------------------------------------------------------------------------
    # Private: Misc.
    def _add_required_symbols(self, test_case: TestCaseProgram) -> None:
        # add measurement_start and measurement_end symbols
        sec_main = test_case[0]
        assert sec_main.owner.is_main
        func_main = sec_main[0]

        bb_first = func_main[0]
        instr = Instruction("macro", category="MACRO") \
            .add_op(LabelOp(".measurement_start")) \
            .add_op(LabelOp(".noarg"))
        bb_first.insert_before(bb_first.get_first(), instr)

        bb_last = func_main.get_exit_bb()
        instr = Instruction("macro", category="MACRO") \
            .add_op(LabelOp(".measurement_end")) \
            .add_op(LabelOp(".noarg"))
        bb_last.insert_after(bb_last.get_last(), instr)

    def _expand_template(self, test_case: TestCaseProgram, actors_config: ActorsConf) -> None:
        nodes_to_expand: List[Tuple[InstructionNode, str]] = []

        # find all instances of .macro.random_instructions
        for bb in test_case.iter_basic_blocks():
            for node in bb.iter_nodes():
                inst = node.instruction
                if inst.name == "macro" and inst.operands[0].value == ".random_instructions":
                    nodes_to_expand.append((node, bb.get_owner().name))

        # replace all instances of .macro.random_instructions with random instructions
        for node, actor_name in nodes_to_expand:
            inst = node.instruction
            bb = node.parent
            operands = inst.operands[1].value.split(".")
            assert len(operands) >= 3 and len(operands) <= 5
            n_instr = int(operands[1])
            n_mem = int(operands[2])

            # determine the instruction set for this actor
            block = actors_config[actor_name]["instruction_blocklist"]
            non_memory_access_instructions = \
                [i for i in self._instruction_set.non_memory_access_specs if i.name not in block]
            store_instructions = \
                [i for i in self._instruction_set.store_instructions if i.name not in block]
            load_instruction = \
                [i for i in self._instruction_set.load_instruction if i.name not in block]

            # replace the macro with random instructions
            bb.delete(node)
            for _ in range(n_instr):
                inst = self._instruction_generator.generate_from_random_spec(
                    non_memory_access_instructions=non_memory_access_instructions,
                    store_instructions=store_instructions,
                    load_instructions=load_instruction,
                    memory_access_probability=n_mem / n_instr)
                if node.previous:
                    bb.insert_after(node.previous, inst)
                else:
                    bb.insert_before(bb.get_first(), inst)


def assemble(test_case: TestCaseProgram) -> None:
    """
    Assemble an assembly file into an object file and creates a stripped binary
    :param test_case: The test case to be assembled
    """

    def pretty_error_msg(error_msg: str) -> str:
        with open(asm_file, "r") as f:
            lines = f.read().split("\n")

        msg = "Error appeared while assembling the test case:\n"
        for line in error_msg.split("\n"):
            line = line.removeprefix(asm_file + ":")
            line_num_str = re.search(r"(\d+):", line)
            if not line_num_str:
                msg += line
            else:
                parsed = lines[int(line_num_str.group(1)) - 1]
                msg += f"\n  Line {line}\n    (the line was parsed as {parsed})"
        return msg

    asm_file = test_case.asm_path()
    obj_container = test_case.get_obj()
    obj_file = obj_container.obj_path

    try:
        out = run(f"as {asm_file} -o {obj_file}", shell=True, check=True, capture_output=True)
    except CalledProcessError as e:
        error_msg = e.stderr.decode()
        if "Assembler messages:" in error_msg:
            print(pretty_error_msg(error_msg))
        else:
            print(error_msg)
        exit(1)
    finally:
        pass
        # run(f"rm {patched_asm_file}", shell=True, check=True)

    output = out.stderr.decode()
    if "Assembler messages:" in output:
        print("WARNING: [generator]" + pretty_error_msg(output))

    obj_container.mark_as_assembled()


# ==================================================================================================
# Private Service Classes
# ==================================================================================================
class _FunctionGenerator:
    """ Class responsible for generating random functions """

    _instruction_generator: _InstructionGenerator
    _isa_spec: InstructionSet

    # GPRs usable as an indirect-branch target (64-bit name -> 32-bit name)
    _TARGET_REGS = {"rax": "eax", "rbx": "ebx", "rcx": "ecx",
                    "rdx": "edx", "rsi": "esi", "rdi": "edi"}

    def __init__(self, target_desc: TargetDesc, isa_spec: InstructionSet) -> None:
        self._instruction_generator = _InstructionGenerator(target_desc)
        self._isa_spec = isa_spec
        self._itable_counter = 0  # unique labels for indirect-call jump tables

    def generate_empty(self, label: str, parent: CodeSection, is_entry: bool = True) -> Function:
        """ Generates an empty function with a random DAG of basic blocks.
        is_entry: entry ends with a jump to the exit; a callee ends with a return. """
        func = Function(label, parent)

        # Define the maximum allowed number of successors for any BB
        if self._isa_spec.has_conditional_branch:
            max_successors = CONF.max_successors_per_bb if CONF.max_successors_per_bb < 2 else 2
            min_successors = CONF.min_successors_per_bb if CONF.min_successors_per_bb < 2 else 2
            assert min_successors <= max_successors, "min_successors_per_bb > max_successors_per_bb"
        else:
            max_successors = 1
            min_successors = 1

        # Create basic blocks
        if CONF.min_bb_per_function == CONF.max_bb_per_function:
            node_count = CONF.min_bb_per_function
        else:
            node_count = random.randint(CONF.min_bb_per_function, CONF.max_bb_per_function)
        func_name = label.removeprefix(".function_")
        nodes = [BasicBlock(f".bb_{func_name}.{i}", func) for i in range(node_count)]

        # Connect BBs into a graph
        for i in range(node_count):
            current_bb = nodes[i]

            # the last node has only one successor - exit
            if i == node_count - 1:
                current_bb.successors = [func.get_exit_bb()]
                break

            # the rest of the node have a random number of successors
            successor_count = random.randint(min_successors, max_successors)
            if successor_count + i > node_count:
                # the number is adjusted to the position when close to the end
                successor_count = node_count - i

            # one of the targets (the first successor) is always the next node - to avoid dead code
            current_bb.successors.append(nodes[i + 1])

            # all other successors are random, selected from next nodes
            options = nodes[i + 2:]
            options.append(func.get_exit_bb())
            for _ in range(1, successor_count):
                target = random.choice(options)
                options.remove(target)
                current_bb.successors.append(target)

        # entry ends with a jump to the exit; a callee returns to its caller
        if is_entry:
            inst = self._instruction_generator.generate(self._isa_spec.get_unconditional_jump_spec())
            assert isinstance(inst.operands[0], LabelOp)
            inst.operands[0].value = TC_EXIT_LABEL
        else:
            inst = self._instruction_generator.generate(self._isa_spec.get_return_spec())
        func.get_exit_bb().terminators = [inst]

        # Finalize the function
        func.extend(nodes)
        return func

    def fill_function(self, func: Function) -> None:
        """
        Fill an (assumed empty) function with random instructions
        :param func: the function to fill
        :return: None
        :raises AssertionError: if the function is not empty
        :raises NotImplementedError: if one of the basic blocks has more than two successors
        """
        self._add_terminators_in_function(func)
        self._add_instructions_in_function(func)

    def _add_terminators_in_function(self, func: Function) -> None:

        def add_fallthrough(bb: BasicBlock, destination: BasicBlock) -> None:
            # create an unconditional branch and add it
            terminator_spec = self._isa_spec.get_unconditional_jump_spec()
            terminator = self._instruction_generator.generate(terminator_spec)
            label = terminator.get_label_operand()
            assert label is not None
            label.value = destination.name
            bb.terminators.append(terminator)

        for bb in func:
            assert not bb.terminators, "Basic block already has terminators"
            if len(bb.successors) == 0:
                # Return instruction
                continue

            if len(bb.successors) == 1:
                # Unconditional branch
                dest = bb.successors[0]
                if dest.is_exit:
                    # DON'T insert a branch to the exit
                    # the last basic block always falls through implicitly
                    continue
                add_fallthrough(bb, dest)
                continue

            if len(bb.successors) == 2:
                # Conditional branch
                spec = random.choice(self._isa_spec.cond_branches)
                terminator = self._instruction_generator.generate(spec)
                label = terminator.get_label_operand()
                assert label
                label.value = bb.successors[0].name
                bb.terminators.append(terminator)

                add_fallthrough(bb, bb.successors[1])
                continue

            # > 2 successors
            raise NotImplementedError("Indirect jumps/calls are not yet supported")

    def _add_instructions_in_function(self, func: Function) -> None:
        """
        Fill the function with random instructions.
        Ensures that all basic blocks are filled with roughly the same number of instructions
        :param func: the function to fill
        :return: None
        """
        bb_list: List[BasicBlock] = list(func)
        assert all(len(bb) == 0 for bb in bb_list), "Basic blocks are not empty"
        for _ in range(0, CONF.program_size):
            bb = random.choice(bb_list)
            inst = self._instruction_generator.generate_from_random_spec(
                self._isa_spec.non_memory_access_specs, self._isa_spec.store_instructions,
                self._isa_spec.load_instruction, CONF.avg_mem_accesses / CONF.program_size)
            bb.insert_after(bb.get_last(), inst)

        # an empty BB breaks X86PatchUndefinedFlagsPass (no node to anchor the patch on); add a nop
        for bb in bb_list:
            if len(bb) == 0:
                bb.insert_after(None, Instruction("nop"))

    def wire_call_graph(self, funcs: List[Function], appended: List[str]) -> None:
        """
        Insert calls (direct or register-indirect) between functions, forming a forward-only
        (acyclic) call graph: function i may call only function j > i, so there is no recursion and
        the depth is bounded. The call goes in as the first terminator of a block -- the parser wants
        control flow at the end of a block, and the last terminator must stay the unconditional
        branch. The call kind is a weighted choice (CONF.call_weight_*).
        """
        n_funcs = len(funcs)
        for i in range(n_funcs - 1):  # the last function has no forward target
            for bb in funcs[i]:
                if random.random() >= CONF.call_probability:
                    continue
                kind = self._pick_call_kind()
                if kind == "direct":
                    callee = funcs[self._pick_forward_index(i, n_funcs)]
                    inst = self._instruction_generator.generate(self._isa_spec.get_call_spec())
                    label = inst.get_label_operand()
                    assert label is not None
                    label.value = callee.name
                    bb.terminators.insert(0, inst)
                else:
                    # indirect call: materialize a forward target into a register, then call it
                    reg = self._materialize_forward_target(i, funcs, bb, appended,
                                                           poly=(kind == "poly"))
                    call = Instruction("call", category="BASE-CALL", is_control_flow=True) \
                        .add_op(RegisterOp(reg, 64, True, False))
                    bb.terminators.insert(0, call)

    @staticmethod
    def _pick_call_kind() -> str:
        """ Weighted choice of call kind: 'direct' | 'mono' | 'poly' (see CONF.call_weight_*) """
        weights = [CONF.call_weight_direct, CONF.call_weight_indirect_mono,
                   CONF.call_weight_indirect_poly]
        if sum(weights) <= 0:
            return "direct"
        return random.choices(["direct", "mono", "poly"], weights=weights)[0]

    @staticmethod
    def _pick_forward_index(i: int, n: int) -> int:
        """ Pick a forward callee index in (i, n), biased toward earlier functions by
        CONF.call_target_bias. bias <= 0 keeps the original uniform draw (RNG-identical). """
        if CONF.call_target_bias <= 0.0:
            return random.randint(i + 1, n - 1)
        cand = list(range(i + 1, n))
        weights = [(n - j) ** CONF.call_target_bias for j in cand]
        return random.choices(cand, weights=weights)[0]

    @staticmethod
    def _pick_forward(forward: List[Function]) -> Function:
        """ Like _pick_forward_index but over a forward slice of Function objects. """
        if CONF.call_target_bias <= 0.0:
            return random.choice(forward)
        m = len(forward)
        weights = [(m - k) ** CONF.call_target_bias for k in range(m)]
        return random.choices(forward, weights=weights)[0]

    def wire_indirect_jmps(self, funcs: List[Function], appended: List[str]) -> None:
        """
        Optionally replace a callee's exit `ret` with a tail `jmp reg` to a forward function. The
        target returns for us (a tail call); forward-only targets keep it acyclic and terminating.
        The entry is left alone so its `jmp .test_case_exit` bounds the measurement window.
        """
        for i in range(1, len(funcs) - 1):
            if random.random() >= CONF.indirect_jmp_probability:
                continue
            exit_bb = funcs[i].get_exit_bb()
            reg = self._materialize_forward_target(i, funcs, exit_bb, appended,
                                                   poly=random.random() < 0.5)
            jmp = Instruction("jmp", category="BASE-UNCOND_BR", is_control_flow=True) \
                .add_op(RegisterOp(reg, 64, True, False))
            exit_bb.terminators = [jmp]

    def _materialize_forward_target(self, i: int, funcs: List[Function],
                                    bb: BasicBlock, appended: List[str], poly: bool) -> str:
        """
        Leave a forward function entry (index > i) in a GPR and return its 64-bit name. Two modes:
          A (mono): `mov reg, offset .function_j`  -- fixed target.
          B (poly): `and reg, N-1; mov reg, [table + reg*8]`  -- data-dependent target via a table.
        The gadget is is_instrumentation so the sandbox pass leaves mode B's table read alone.
        """
        reg = random.choice(list(self._TARGET_REGS))
        reg32 = self._TARGET_REGS[reg]
        forward = funcs[i + 1:]

        if not poly or len(forward) == 1:
            # `offset` is required: `mov reg, .function_j` would assemble as a load from the label
            callee = self._pick_forward(forward)
            mov = Instruction("mov", is_instrumentation=True) \
                .add_op(RegisterOp(reg32, 32, False, True)) \
                .add_op(ImmediateOp("offset " + callee.name, 32))
            bb.insert_after(bb.get_last(), mov)
            return reg

        # Mode B: power-of-two pointer table
        n = 1
        while n < len(forward):
            n <<= 1
        label = f".itable_{self._itable_counter}"
        self._itable_counter += 1
        if CONF.call_target_bias <= 0.0:
            entries = ", ".join(forward[k % len(forward)].name for k in range(n))
        else:
            entries = ", ".join(self._pick_forward(forward).name for _ in range(n))
        appended.append(f"{label}: .quad {entries}")

        mask = Instruction("and", is_instrumentation=True) \
            .add_op(RegisterOp(reg32, 32, True, True)).add_op(ImmediateOp(str(n - 1), 32))
        load = Instruction("mov", is_instrumentation=True) \
            .add_op(RegisterOp(reg, 64, False, True)) \
            .add_op(MemoryOp(f"{label} + {reg}*8", 64, True, False))
        bb.insert_after(bb.get_last(), mask)
        bb.insert_after(bb.find_instruction_node(mask), load)
        return reg

    def perturb_return_stack(self, funcs: List[Function]) -> None:
        """
        Sparsely inject branch-free return-stack perturbations. No runtime guards are used -- a guard
        would add a branch and train the predictors we want to watch; termination is guaranteed by
        the static call graph instead. All injected instructions are is_instrumentation. Gated by
        CONF.perturb_probability, capped at CONF.max_perturbations. Mechanisms:
          * pop-skip      -- `add rsp, 8` before a callee `ret`: returns one frame further out.
          * abandon-frame -- callee exits via `jmp .test_case_exit` instead of `ret`.
          * rsb-desync    -- `call .+0 ; add rsp, 8`: pushes the RSB but is an rsp no-op.
        """
        if CONF.perturb_probability <= 0.0:
            return
        applied = 0
        for i, func in enumerate(funcs):
            if applied >= CONF.max_perturbations:
                break
            if random.random() >= CONF.perturb_probability:
                continue
            is_callee = (i != 0)  # func 0 is the entry, holds no return address
            mech = random.choice(
                ["pop_skip", "abandon_frame", "rsb_desync"] if is_callee else ["rsb_desync"])
            if mech == "pop_skip":
                exit_bb = func.get_exit_bb()
                exit_bb.insert_after(exit_bb.get_last(), self._add_rsp8())
            elif mech == "abandon_frame":
                jmp = self._instruction_generator.generate(
                    self._isa_spec.get_unconditional_jump_spec())
                label = jmp.get_label_operand()
                assert label is not None
                label.value = TC_EXIT_LABEL
                func.get_exit_bb().terminators = [jmp]
            else:  # rsb_desync
                bb = random.choice(list(func))
                call0 = Instruction(".byte 0xe8, 0x00, 0x00, 0x00, 0x00", is_instrumentation=True)
                bb.insert_after(bb.get_last(), call0)     # call .+0
                bb.insert_after(bb.find_instruction_node(call0), self._add_rsp8())
            applied += 1

    @staticmethod
    def _add_rsp8() -> Instruction:
        """ `add rsp, 8` (instrumentation, so the sandbox pass leaves rsp alone) """
        return Instruction("add", is_instrumentation=True) \
            .add_op(RegisterOp("rsp", 64, True, True)) \
            .add_op(ImmediateOp("8", 8))

    @staticmethod
    def seed_loop_var(entry: Function) -> None:
        """ Prepend `mov rax, r13` to the entry: rax starts each run at the harness training-loop
        counter (r13), so repeated runs of the program differ. r13 is set by the harness and
        blocklisted from the operand pool, so nothing else reads it. """
        seed = Instruction("mov", is_instrumentation=True) \
            .add_op(RegisterOp("rax", 64, False, True)) \
            .add_op(RegisterOp("r13", 64, True, False))
        next(iter(entry)).insert_before(None, seed)

    @staticmethod
    def assign_collision_sections(funcs: List[Function]) -> None:
        """
        With probability CONF.collision_probability, place a callee (not the entry) into one of the
        fixed-VA BTB-collision sections (.spec_coll1..4) instead of .spec_main, so its branches can
        alias others in the BTB. Per-function, so the function's basic blocks stay contiguous.
        """
        if CONF.collision_probability <= 0.0:
            return
        coll = [".spec_coll1", ".spec_coll2", ".spec_coll3", ".spec_coll4"]
        for func in funcs[1:]:
            if random.random() < CONF.collision_probability:
                func.coll_section = random.choice(coll)  # type: ignore[attr-defined]


class _InstructionGenerator:
    """
    Class responsible for generating random instructions
    """

    _operand_generator: _OperandGenerator

    def __init__(self, target_desc: TargetDesc) -> None:
        self._operand_generator = _OperandGenerator(target_desc)

    def generate(self, spec: InstructionSpec, is_instrumentation: bool = False) -> Instruction:
        """
        Generate a random instruction object based on the specification
        :param spec: The specification of the instruction
        :param is_instrumentation: Whether to label the instruction as instrumentation
        :return: The generated instruction
        """

        # fill up with random operands, following the spec
        inst = Instruction.from_spec(spec, is_instrumentation=is_instrumentation)

        # generate explicit operands
        for operand_spec in spec.operands:
            operand = self._operand_generator.generate(operand_spec, inst)
            inst.operands.append(operand)

        # generate implicit operands
        for operand_spec in spec.implicit_operands:
            operand = self._operand_generator.generate(operand_spec, inst)
            inst.implicit_operands.append(operand)

        return inst

    def generate_from_random_spec(self,
                                  non_memory_access_instructions: List[InstructionSpec],
                                  store_instructions: List[InstructionSpec],
                                  load_instructions: List[InstructionSpec],
                                  memory_access_probability: float = 0.0) -> Instruction:
        """
        Generate an instruction from a randomly-selected specification
        :param non_memory_access_instructions: The list of available non-memory access instructions
        :param store_instructions: The list of available store instructions
        :param load_instructions: The list of available load instructions
        :return: The generated instruction
        """

        def pick_spec() -> InstructionSpec:
            # ensure the requested avg. number of mem. accesses
            search_for_memory_access = random.random() < memory_access_probability
            if not search_for_memory_access:
                return random.choice(non_memory_access_instructions)

            if store_instructions:
                search_for_store = random.random() < 0.5  # 50% probability of stores
            else:
                search_for_store = False

            if search_for_store:
                return random.choice(store_instructions)

            return random.choice(load_instructions)

        spec = pick_spec()
        return self.generate(spec)


class _OperandGenerator:
    """
    Class responsible for generating random operands for instructions
    """

    def __init__(self, target_desc: TargetDesc) -> None:
        self.target_desc = target_desc

    def generate(self, spec: OperandSpec, parent: Instruction) -> AnyOperand:
        """
        Generate a random operand object based on the specification
        """
        generators: Dict[OT, Callable[[OperandSpec, Instruction], AnyOperand]] = {
            OT.REG: self._generate_reg_operand,
            OT.MEM: self._generate_mem_operand,
            OT.IMM: self._generate_imm_operand,
            OT.LABEL: self._generate_label_operand,
            OT.AGEN: self._generate_agen_operand,
            OT.FLAGS: self._generate_flags_operand,
            OT.COND: self._generate_cond_operand,
        }
        return generators[spec.type](spec, parent)

    def _generate_reg_operand(self, spec: OperandSpec, _: Instruction) -> RegisterOp:
        choices = spec.values
        reg = random.choice(choices)
        return RegisterOp(reg, spec.width, spec.src, spec.dest)

    def _generate_mem_operand(self, spec: OperandSpec, _: Instruction) -> MemoryOp:
        if spec.values:
            address_reg = random.choice(spec.values)
        else:
            address_reg = random.choice(self.target_desc.mem_index_registers)
        return MemoryOp(address_reg, spec.width, spec.src, spec.dest)

    def _generate_imm_operand(self, spec: OperandSpec, inst: Instruction) -> ImmediateOp:
        # generate bitmask
        if spec.values and spec.values[0] == "bitmask":
            return self._generate_bitmask_operand(spec, inst)

        # generate from a predefined list
        if spec.values and "[" not in spec.values[0]:
            options: Iterable[str] | Iterable[int]
            try:
                options = [int(v) for v in spec.values]
            except ValueError:
                # handle non-digit immediates (e.g., dsb SY in ARM64)
                options = list(spec.values)
            value = str(random.choice(options))
            return ImmediateOp(value, spec.width)

        # generate from a predefined range
        if spec.values:
            assert "[" in spec.values[0], f"Invalid IMM spec for instruction: {inst}"
            range_ = spec.values[0][1:-1].split("-")
            if range_[0] == "":
                range_ = range_[1:]
                range_[0] = "-" + range_[0]
            assert len(range_) == 2
            value = str(random.randint(int(range_[0]), int(range_[1])))
            return ImmediateOp(value, spec.width)

        # generate from width
        if spec.is_signed:
            range_min = pow(2, spec.width - 1) * -1
            range_max = pow(2, spec.width - 1) - 1
        else:
            range_min = 0
            range_max = pow(2, spec.width) - 1
        value = str(random.randint(range_min, range_max))
        return ImmediateOp(value, spec.width)

    def _generate_bitmask_operand(self, spec: OperandSpec, _: Instruction) -> ImmediateOp:
        assert CONF.instruction_set == "arm64"

        if spec.width == 64:
            imms_zero_pos = random.randint(1, 6)
        else:
            imms_zero_pos = random.randint(1, 5)
        imms_ones = random.randint(1, 2**imms_zero_pos - 1)

        immr = random.randint(0, spec.width - 1)

        pattern = "0" * (2**imms_zero_pos - imms_ones) + "1" * imms_ones
        multiplier = spec.width // (2**imms_zero_pos)
        value_str = pattern * multiplier
        value = int(value_str, 2)
        value = (value >> immr) | (value << (spec.width - immr)) & (2**spec.width - 1)

        if spec.width == 64:
            value_str = f"0x{value:016x}"
        else:
            value_str = f"0x{value:08x}"
        return ImmediateOp(value_str, spec.width)

    def _generate_label_operand(self, _: OperandSpec, __: Instruction) -> LabelOp:
        return LabelOp("")  # the actual label will be set in add_terminators_in_function

    def _generate_agen_operand(self, spec: OperandSpec, __: Instruction) -> AgenOp:
        n_operands = random.randint(1, 3)
        reg1 = random.choice(self.target_desc.mem_index_registers)
        if n_operands == 1:
            return AgenOp(reg1, spec.width)

        reg2 = random.choice(self.target_desc.mem_index_registers)
        if n_operands == 2:
            return AgenOp(reg1 + " + " + reg2, spec.width)

        imm = str(random.randint(0, pow(2, 16) - 1))
        return AgenOp(reg1 + " + " + reg2 + " + " + imm, spec.width)

    def _generate_flags_operand(self, spec: OperandSpec, parent: Instruction) -> FlagsOp:
        # pylint: disable=too-many-branches
        # NOTE: there are many options for COND flags, so many branches are needed

        cond_op = parent.get_cond_operand()
        if not cond_op:
            return FlagsOp(spec.values)
        raise NotImplementedError("COND operand is not yet supported")
        # pylint: disable=unreachable
        # NOTE: the code below is temporary disabled

        flag_values = self.target_desc.branch_conditions[cond_op.value]
        if not spec.values:
            return FlagsOp(flag_values)

        # combine implicit flags with the condition
        merged_flags = []
        for flag_pair in zip(flag_values, spec.values):
            if "undef" in flag_pair:
                merged_flags.append("undef")
            elif "r/w" in flag_pair:
                merged_flags.append("r/w")
            elif "w" in flag_pair:
                if "r" in flag_pair:
                    merged_flags.append("r/w")
                else:
                    merged_flags.append("w")
            elif "cw" in flag_pair:
                if "r" in flag_pair:
                    merged_flags.append("r/cw")
                else:
                    merged_flags.append("cw")
            elif "r" in flag_pair:
                merged_flags.append("r")
            else:
                merged_flags.append("")
        return FlagsOp(merged_flags)

    def _generate_cond_operand(self, _: OperandSpec, __: Instruction) -> CondOp:
        cond = random.choice(list(self.target_desc.branch_conditions))
        return CondOp(cond)
