#!/usr/bin/env python

"""a tiny language that compiles to intcode"""

from argparse import ArgumentParser
from contextlib import contextmanager
from itertools import chain, repeat
from lark import Lark, Token, Transformer, v_args
from pathlib import Path
import sys

from intcode import OPCODE


class OP:
    OR = 'or'
    AND = 'and'
    NOT = 'not'
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    EQ = '=='
    NE = '!='
    ADD = '+'
    SUB = '-'
    MUL = '*'
    ASSIGN = '='
    ASSIGN_ADD = '+='
    ASSIGN_SUB = '-='
    ASSIGN_MUL = '*='


assign_ops = {
    OP.ASSIGN_ADD: OP.ADD,
    OP.ASSIGN_SUB: OP.SUB,
    OP.ASSIGN_MUL: OP.MUL
}


class ASTNode:
    _id = ''
    _props = []

    def __init__(slf, **kwargs):
        for prop, val in kwargs.items():
            if prop not in slf._props:
                raise Exception(
                    f"'{prop}' is not a property of '{type(slf).__name__}'"
                )
            setattr(slf, prop, val)

        for prop in slf._props:
            if getattr(slf, prop) is None:
                setattr(slf, prop, None)


class ASTProgram(ASTNode):
    _id = 'program'
    _props = ['stmts']


class ASTBlockStatement(ASTNode):
    _id = 'block_stmt'
    _props = ['body']


class ASTAssignStatement(ASTNode):
    _id = 'assign_stmt'
    _props = ['target', 'op', 'expr']


class ASTInputStatement(ASTNode):
    _id = 'input_stmt'
    _props = ['dest']


class ASTOutputStatement(ASTNode):
    _id = 'output_stmt'
    _props = ['src']


class ASTIfStatement(ASTNode):
    _id = 'if_stmt'
    _props = ['cond', 'true', 'false']


class ASTWhileStatement(ASTNode):
    _id = 'while_stmt'
    _props = ['cond', 'body']


class ASTContinueStatement(ASTNode):
    _id = 'continue_stmt'


class ASTBreakStatement(ASTNode):
    _id = 'break_stmt'


class ASTArrayStatement(ASTNode):
    _id = 'array_stmt'
    _props = ['target', 'size']


class ASTBinaryExpression(ASTNode):
    _id = 'binary_expr'
    _props = ['left', 'op', 'right']


class ASTUnaryExpression(ASTNode):
    _id = 'unary_expr'
    _props = ['op', 'right']


class ASTSubscriptExpression(ASTNode):
    _id = 'subscript_expr'
    _props = ['addr', 'index']


class ASTLiteral(ASTNode):
    _id = 'literal'
    _props = ['value']


class ASTIdentifer(ASTNode):
    _id = 'identifier'
    _props = ['value']


@v_args(inline=True)
class ASTTransformer(Transformer):
    def program(slf, *stmts):
        return ASTProgram(stmts=stmts)

    def block_stmt(slf, *body):
        return ASTBlockStatement(body=list(body))

    def assign_stmt(slf, target, op, expr):
        return ASTAssignStatement(target=target, op=op, expr=expr)

    def input_stmt(slf, dest):
        return ASTInputStatement(dest=dest)

    def output_stmt(slf, src):
        return ASTOutputStatement(src=src)

    def if_stmt(slf, cond, true, false=None):
        return ASTIfStatement(cond=cond, true=true, false=false)

    def while_stmt(slf, cond, body):
        return ASTWhileStatement(cond=cond, body=body)

    def continue_stmt(slf):
        return ASTContinueStatement()

    def break_stmt(slf):
        return ASTBreakStatement()

    def array_stmt(slf, target, size):
        return ASTArrayStatement(target=target, size=size)

    def binary_expr(slf, left, op, right):
        return ASTBinaryExpression(left=left, op=op, right=right)

    def unary_expr(slf, op, right):
        return ASTUnaryExpression(op=op, right=right)

    def subscript_expr(slf, addr, index):
        return ASTSubscriptExpression(addr=addr, index=index)

    def literal(slf, token):
        return ASTLiteral(value=int(token))

    def identifier(slf, token):
        return ASTIdentifer(value=str(token))


class ASTVisitor:
    def visit(slf, node):
        if node:
            visit = f'visit_{node._id}'
            if hasattr(slf, visit):
                return getattr(slf, visit)(node)


class IRLiteral:
    immediate = True

    def __init__(slf, value):
        slf.value = value

    def __hash__(slf):
        return hash(('IRLiteral', slf.value))

    def __eq__(slf, otr):
        return isinstance(otr, IRLiteral) and slf.value == otr.value

    def __str__(slf):
        return str(slf.value)

    def __repr__(slf):
        return f'IRLiteral({slf.value})'


class IRAddress:
    immediate = False

    def __init__(slf, value):
        slf.value = value

    def __str__(slf):
        return f'#{slf.value}'

    def __repr__(slf):
        return f'IRAddress({slf.value})'


class IRStorage:
    pass


class IRVariable(IRStorage):
    immediate = False
    size = 1

    def __init__(slf, name):
        slf.name = name

    def __hash__(slf):
        return hash(('IRVariable', slf.name))

    def __eq__(slf, otr):
        return isinstance(otr, IRVariable) and slf.name == otr.name

    def __str__(slf):
        return str(slf.name)

    def __repr__(slf):
        return f'IRVariable({slf.name})'


class IRArray(IRStorage):
    immediate = True

    def __init__(slf, size):
        if size < 1:
            raise Exception('Arrays must have at least one element!')
        slf.size = size

    def __str__(slf):
        return f'array[{slf.size}]'

    def __repr__(slf):
        return f'IRArray({slf.size})'


class IRElement:
    immediate = False

    def __init__(slf, addr, index):
        slf.addr = addr
        slf.index = index

    def __str__(slf):
        return f'{slf.addr}[{slf.index}]'

    def __repr__(slf):
        return f'IRElement({slf.addr}, {slf.index})'


class IRInstruction:
    _lhs = []
    _rhs = []

    def __init__(slf, *args):
        ln = len(slf._lhs)
        rn = len(slf._rhs)
        n = ln + rn
        if len(args) != n:
            raise Exception(
                f"'{type(slf).__name__}' expects {n} parameters"
            )

        slf.lhs = list(args[:ln])
        slf.rhs = list(args[ln:])


class IRLabel(IRInstruction):
    immediate = True
    _id = 'label'
    _rhs = ['name']

    def __str__(slf):
        return f'label {slf.rhs[0]}'


class IRInputAssignment(IRInstruction):
    _id = 'input'
    _lhs = ['dest']

    def __str__(slf):
        return f'{slf.lhs[0]} << input'


class IROutput(IRInstruction):
    _id = 'output'
    _rhs = ['src']

    def __str__(slf):
        return f'output << {slf.rhs[0]}'


class IRCopyAssignment(IRInstruction):
    _id = 'copy'
    _lhs = ['result']
    _rhs = ['right']

    def __str__(slf):
        return f'{slf.lhs[0]} := {slf.rhs[0]}'


class IRUnaryExprAssignment(IRInstruction):
    _id = 'unary'
    _lhs = ['result']
    _rhs = ['op', 'right']

    def __str__(slf):
        return f'{slf.lhs[0]} := {slf.rhs[0]} {slf.rhs[1]}'


class IRBinaryExprAssignment(IRInstruction):
    _id = 'binary'
    _lhs = ['result']
    _rhs = ['left', 'op', 'right']

    def __str__(slf):
        return f'{slf.lhs[0]} := {slf.rhs[0]} {slf.rhs[1]} {slf.rhs[2]}'


class IRGoto(IRInstruction):
    _id = 'goto'
    _rhs = ['label']

    def __str__(slf):
        return f'goto {slf.rhs[0].rhs[0]}'


class IRGotoIfFalse(IRInstruction):
    _id = 'goto_if_false'
    _rhs = ['cond', 'label']

    def __str__(slf):
        return f'if(!{slf.rhs[0]}) goto {slf.rhs[1].rhs[0]}'


class IRArrayAssignment(IRInstruction):
    _id = 'array'
    _lhs = ['result']
    _rhs = ['array']

    def __str__(slf):
        return f'{slf.lhs[0]} = {slf.rhs[0]}'


class IRScope:
    CONTINUE = 'continue'
    BREAK = 'break'

    def __init__(slf):
        slf.variables = {}
        slf.labels = {}


class IRScopeManager:
    def __init__(slf):
        slf.scopes = []

    def __enter__(slf):
        scope = IRScope()
        slf.scopes.append(scope)
        return scope

    def __exit__(slf, *args):
        slf.scopes.pop()

    def map_variable(slf, v):
        scope = slf.scopes[-1]
        if v.name in scope.variables:
            raise Exception(
                f"There is already a variable named '{v.name}' in the scope"
            )
        else:
            scope.variables[v.name] = v

    def map_label(slf, name, label):
        slf.scopes[-1].labels[name] = label

    def find_variable(slf, name):
        for scope in reversed(slf.scopes):
            if name in scope.variables:
                return scope.variables[name]

        raise KeyError(
            f"There is no variable named '{name}' in the current scope"
        )

    def find_label(slf, name):
        for scope in reversed(slf.scopes):
            if name in scope.labels:
                return scope.labels[name]
        raise KeyError()


class IRGenerator(ASTVisitor):
    def __init__(slf):
        slf.scope = IRScopeManager()
        slf.next_label = 0
        slf.next_variable = 0
        slf.instrs = []

    def emit(slf, instr):
        slf.instrs.append(instr)

    def label(slf):
        label = IRLabel(f'L{slf.next_label}')
        slf.next_label += 1
        return label

    def temp(slf):
        temp = IRVariable(f't{slf.next_variable}')
        slf.next_variable += 1
        return temp

    def visit_program(slf, node):
        with slf.scope:
            for stmt in node.stmts:
                slf.visit(stmt)
        return slf.instrs

    def visit_block_stmt(slf, node):
        with slf.scope:
            for stmt in node.body:
                slf.visit(stmt)

    def find_or_create_variable(slf, name):
        try:
            return slf.scope.find_variable(name)
        except KeyError:
            variable = IRVariable(name)
            slf.scope.map_variable(variable)
            return variable

    def visit_assign_stmt(slf, node):
        # Assignments to a variable automatically instantiates that variable,
        # if it doesn't already exist.
        if isinstance(node.target, ASTIdentifer):
            target = slf.find_or_create_variable(node.target.value)
        else:
            target = slf.visit(node.target)
        expr = slf.visit(node.expr)

        if node.op == OP.ASSIGN:
            slf.emit(IRCopyAssignment(target, expr))
        else:
            op = assign_ops[node.op]
            slf.emit(IRBinaryExprAssignment(target, target, op, expr))

    def visit_input_stmt(slf, node):
        if isinstance(node.dest, ASTIdentifer):
            dest = slf.find_or_create_variable(node.dest.value)
        else:
            dest = slf.visit(node.dest)
        slf.emit(IRInputAssignment(dest))

    def visit_output_stmt(slf, node):
        src = slf.visit(node.src)
        slf.emit(IROutput(src))

    def visit_if_stmt(slf, node):
        label_false = slf.label()
        label_end = slf.label()

        # Condition
        cond = slf.visit(node.cond)
        slf.emit(IRGotoIfFalse(cond, label_false))

        # True
        slf.visit(node.true)
        slf.emit(IRGoto(label_end))

        # False
        slf.emit(label_false)
        if node.false:
            slf.visit(node.false)

        # End
        slf.emit(label_end)

    def visit_while_stmt(slf, node):
        label_loop = slf.label()
        label_end = slf.label()

        # Condition
        slf.emit(label_loop)
        cond = slf.visit(node.cond)
        slf.emit(IRGotoIfFalse(cond, label_end))

        # Body
        with slf.scope:
            slf.scope.map_label(IRScope.CONTINUE, label_loop)
            slf.scope.map_label(IRScope.BREAK, label_end)
            slf.visit(node.body)
            slf.emit(IRGoto(label_loop))

        # End
        slf.emit(label_end)

    def visit_continue_stmt(slf, node):
        try:
            label = slf.scope.find_label(IRScope.CONTINUE)
        except KeyError:
            raise Exception('Continue statement outside loop')
        slf.emit(IRGoto(label))

    def visit_break_stmt(slf, node):
        try:
            label = slf.scope.find_label(IRScope.BREAK)
        except KeyError:
            raise SyntaxError('Break statement outside loop')
        slf.emit(IRGoto(label))

    def visit_array_stmt(slf, node):
        target = slf.find_or_create_variable(node.target.value)
        array = IRArray(node.size.value)
        slf.emit(IRArrayAssignment(target, array))

    def visit_binary_expr(slf, node):
        left = slf.visit(node.left)
        right = slf.visit(node.right)
        result = slf.temp()
        slf.emit(IRBinaryExprAssignment(result, left, node.op.value, right))
        return result

    def visit_unary_expr(slf, node):
        right = slf.visit(node.right)
        result = slf.temp()
        slf.emit(IRUnaryExprAssignment(result, node.op.value, right))
        return result

    def visit_subscript_expr(slf, node):
        addr = slf.visit(node.addr)
        index = slf.visit(node.index)
        return IRElement(addr, index)

    def visit_identifier(slf, node):
        return slf.scope.find_variable(node.value)

    def visit_literal(slf, node):
        return IRLiteral(node.value)


class IntcodeGenerator:
    def generate(slf, ir):
        slf.memory = []
        slf.labels = {}
        slf.free_scratches = set()
        slf.used_scratches = set()

        # Process each instruction.
        for instr in ir:
            visit = f'visit_{instr._id}'
            assert hasattr(slf, visit)
            getattr(slf, visit)(*chain(instr.lhs, instr.rhs))

        # Insert a stop instruction at the end of the program.
        slf.memory.extend([OPCODE.STOP])

        # Allocate storage for variables and arrays.
        offset = len(slf.memory)
        storage = {}
        for v in slf.memory:
            if isinstance(v, IRStorage) and v not in storage:
                storage[v] = len(slf.memory)
                slf.memory.extend(repeat(0, v.size))

        # Replace label and storage placeholders.
        for i, cell in enumerate(slf.memory):
            if isinstance(cell, IRStorage):
                slf.memory[i] = storage[cell]
            elif isinstance(cell, IRLabel):
                slf.memory[i] = slf.labels[cell.rhs[0]]

        return slf.memory

    @contextmanager
    def scratch(slf):
        if len(slf.free_scratches) == 0:
            n = len(slf.used_scratches)
            slf.free_scratches.add(IRVariable(f'__scratch{n}__'))

        scratch = slf.free_scratches.pop()
        slf.used_scratches.add(scratch)

        yield scratch

        slf.used_scratches.remove(scratch)
        slf.free_scratches.add(scratch)

    def emit(slf, opcode, *params):
        # To handle indexing, we emit a prelude which modifies the instruction
        # with the address of the element we are reading or writing.
        elements = [e for e in params if isinstance(e, IRElement)]
        preludes = {}
        for element in elements:
            slf.emit(OPCODE.ADD, element.addr, element.index, IRAddress(0))
            preludes[element] = len(slf.memory) - 1

        # Now emit the actual instruction.
        flags = ['1' if p.immediate else '0' for p in params]
        head = ''.join(reversed(flags)) + f'{opcode:02d}'
        slf.memory.append(int(head))
        for param in params:
            if isinstance(param, IRLiteral) or isinstance(param, IRAddress):
                slf.memory.append(param.value)
            elif isinstance(param, IRElement):
                # Write the address of this parameter into the prelude
                # instruction which will modify it.
                slf.memory[preludes[param]] = len(slf.memory)
                slf.memory.append(0)
            else:
                slf.memory.append(param)

    def visit_label(slf, name):
        slf.labels[name] = len(slf.memory)

    def visit_input(slf, dest):
        slf.emit(OPCODE.INPUT, dest)

    def visit_output(slf, src):
        slf.emit(OPCODE.OUTPUT, src)

    def visit_copy(slf, result, right):
        slf.visit_binary(result, IRLiteral(0), OP.ADD, right)

    def visit_unary(slf, result, op, right):
        if op == OP.NOT:
            slf.visit_binary(result, IRLiteral(0), OP.EQ, right)
        elif op == OP.SUB:
            slf.visit_binary(result, IRLiteral(-1), OP.MUL, right)
        elif op == OP.ADD:
            slf.visit_copy(result, right)
        else:
            raise Exception(f"Unknown unary op '{op}'")

    def visit_binary(slf, result, left, op, right):
        if op == OP.OR:
            with slf.scratch() as scratch0, slf.scratch() as scratch1:
                slf.visit_binary(scratch0, left, OP.EQ, IRLiteral(0))
                slf.visit_binary(scratch1, right, OP.EQ, IRLiteral(0))
                slf.visit_binary(scratch0, scratch0, OP.ADD, scratch1)
                slf.visit_binary(result, scratch0, OP.LT, IRLiteral(2))
        elif op == OP.AND:
            with slf.scratch() as scratch0, slf.scratch() as scratch1:
                slf.visit_binary(scratch0, left, OP.EQ, IRLiteral(0))
                slf.visit_binary(scratch1, right, OP.EQ, IRLiteral(0))
                slf.visit_binary(scratch0, scratch0, OP.ADD, scratch1)
                slf.visit_binary(result, scratch0, OP.EQ, IRLiteral(0))
        elif op == OP.LT:
            slf.emit(OPCODE.LESS_THAN, left, right, result)
        elif op == OP.LE:
            with slf.scratch() as scratch0, slf.scratch() as scratch1:
                slf.visit_binary(scratch0, left, OP.LT, right)
                slf.visit_binary(scratch1, left, OP.EQ, right)
                slf.visit_binary(scratch0, scratch0, OP.ADD, scratch1)
                slf.visit_binary(result, scratch0, OP.EQ, IRLiteral(1))
        elif op == OP.GT:
            with slf.scratch() as scratch:
                slf.visit_binary(scratch, left, OP.LE, right)
                slf.visit_unary(result, OP.NOT, scratch)
        elif op == OP.GE:
            with slf.scratch() as scratch:
                slf.visit_binary(scratch, left, OP.LT, right)
                slf.visit_unary(result, OP.NOT, scratch)
        elif op == OP.EQ:
            slf.emit(OPCODE.EQUALS, left, right, result)
        elif op == OP.NE:
            with slf.scratch() as scratch:
                slf.emit(OPCODE.EQUALS, left, right, scratch)
                slf.visit_unary(result, OP.NOT, scratch)
        elif op == OP.ADD:
            slf.emit(OPCODE.ADD, left, right, result)
        elif op == OP.SUB:
            with slf.scratch() as scratch:
                slf.visit_unary(scratch, OP.SUB, right)
                slf.emit(OPCODE.ADD, left, scratch, result)
        elif op == OP.MUL:
            slf.emit(OPCODE.MULTIPLY, left, right, result)
        else:
            raise Exception(f"Unknown binary op '{op}'")

    def visit_goto(slf, label):
        slf.emit(OPCODE.JUMP_IF_TRUE, IRLiteral(1), label)

    def visit_goto_if_false(slf, cond, label):
        slf.emit(OPCODE.JUMP_IF_FALSE, cond, label)

    def visit_array(slf, result, array):
        slf.emit(OPCODE.ADD, IRLiteral(0), array, result)


def intscript(file):
    # Load grammar.
    grammar_path = Path(__file__).parent / 'intscript.lark'
    with open(grammar_path, mode='rt', encoding='UTF-8') as f:
        parser = Lark(f.read(), start='program', propagate_positions=True)

    # Parse tree.
    with open(file, mode='rt', encoding='UTF-8') as f:
        parse_tree = parser.parse(f.read())

    # Abstract syntax tree.
    ast = ASTTransformer().transform(parse_tree)

    # Intermediate representation.
    ir = IRGenerator().visit(ast)

    # Intcode.
    return IntcodeGenerator().generate(ir)


def main(argv):
    arg_parser = ArgumentParser(description='intscript compiler')
    arg_parser.add_argument(
        'file',
        help='intscript file'
    )
    args = arg_parser.parse_args(argv)
    intcode = intscript(args.file)
    print(','.join(map(str, intcode)))


if __name__ == "__main__":
    main(sys.argv[1:])
