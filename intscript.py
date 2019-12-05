#!/usr/bin/env python

"""a tiny language that compiles to intcode"""

from argparse import ArgumentParser
from lark import Lark, Transformer, v_args
from pathlib import Path
import sys


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


class ASTBinaryExpression(ASTNode):
    _id = 'binary_expr'
    _props = ['left', 'op', 'right']


class ASTUnaryExpression(ASTNode):
    _id = 'unary_expr'
    _props = ['op', 'right']


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

    def binary_expr(slf, left, op, right):
        return ASTBinaryExpression(left=left, op=op, right=right)

    def unary_expr(slf, op, right):
        return ASTUnaryExpression(op=op, right=right)

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
    def __init__(slf, value):
        slf.value = value
        slf.type = type

    def __hash__(slf):
        return hash(('IRLiteral', slf.value))

    def __eq__(slf, otr):
        return isinstance(otr, IRLiteral) and slf.value == otr.value

    def __str__(slf):
        return str(slf.value)

    def __repr__(slf):
        return f'IRLiteral({slf.value})'


class IRVariable:
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
    _id = 'ouptput'
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

    def map_variable(slf, name, variable):
        scope = slf.scopes[-1]
        if name in scope.variables:
            raise Exception(
                f"There is already a variable named '{name}' in the scope"
            )
        else:
            scope.variables[name] = variable

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
            slf.scope.map_variable(name, variable)
            return variable

    def visit_assign_stmt(slf, node):
        target = slf.find_or_create_variable(node.target.value)
        expr = slf.visit(node.expr)

        if node.op == OP.ASSIGN:
            slf.emit(IRCopyAssignment(target, expr))
        else:
            op = assign_ops[node.op]
            slf.emit(IRBinaryExprAssignment(target, target, op, expr))

    def visit_input_stmt(slf, node):
        dest = slf.find_or_create_variable(node.dest.value)
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

    def visit_binary_expr(slf, node):
        left = slf.visit(node.left)
        right = slf.visit(node.right)
        result = slf.temp()
        slf.emit(IRBinaryExprAssignment(result, left, node.op.value, right))
        return result

    def visit_unary_expr(slf, node):
        right = slf.visit(node.right)
        result = slf.temp()
        slf.emit(IRUnaryExprAssignment(result, op, right))
        return result

    def visit_identifier(slf, node):
        return slf.scope.find_variable(node.value)

    def visit_literal(slf, node):
        return IRLiteral(node.value)


def main(argv):
    arg_parser = ArgumentParser(description='intscript compiler')
    arg_parser.add_argument(
        'file',
        help='intscript file'
    )
    args = arg_parser.parse_args(argv)

    # Load grammar.
    grammar_path = Path(__file__).parent / 'intscript.lark'
    with open(grammar_path, mode='rt', encoding='UTF-8') as f:
        parser = Lark(
            f.read(),
            start='program',
            debug=True,
            propagate_positions=True
        )

    # Parse tree.
    with open(args.file, mode='rt', encoding='UTF-8') as f:
        parse_tree = parser.parse(f.read())

    # Abstract syntax tree.
    ast = ASTTransformer().transform(parse_tree)

    # Intermediate representation.
    ir = IRGenerator().visit(ast)

    for instr in ir:
        print(instr)

if __name__ == "__main__":
    main(sys.argv[1:])
