"""format_plan in the format pinned by tests/test_ir_nodes.py.

Additionally prints Scan.columns and Scan.pushed_predicate when set. That
is the extension requested from Person A, so B2/B3 traces show their effect.
"""

from __future__ import annotations

from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort


def format_plan(plan) -> str:
    lines: list[str] = []
    _walk(plan, 0, lines)
    return "\n".join(lines)


def _walk(node, depth, lines):
    lines.append("  " * depth + _label(node))
    for child in node.children:
        _walk(child, depth + 1, lines)


def _label(node) -> str:
    if isinstance(node, Scan):
        parts = [node.table]
        if node.columns is not None:
            parts.append(f"columns=[{', '.join(node.columns)}]")
        if node.pushed_predicate is not None:
            parts.append(f"pushed={format_expr(node.pushed_predicate)}")
        return f"Scan[{', '.join(parts)}]"
    if isinstance(node, Filter):
        return f"Filter[{format_expr(node.predicate)}]"
    if isinstance(node, Project):
        return f"Project[{', '.join(_aliased(e, a) for e, a in node.exprs)}]"
    if isinstance(node, Join):
        return f"Join[kind={node.kind}, cond={format_expr(node.condition)}]"
    if isinstance(node, Aggregate):
        parts = []
        if node.group_keys:
            keys = ", ".join(format_expr(k) for k in node.group_keys)
            parts.append(f"group={keys}" if len(node.group_keys) == 1 else f"group=[{keys}]")
        if node.aggs:
            parts.append(f"aggs={', '.join(_aliased(c, a) for c, a in node.aggs)}")
        return f"Aggregate[{', '.join(parts)}]"
    if isinstance(node, Sort):
        keys = ", ".join(f"{format_expr(e)} {'DESC' if desc else 'ASC'}" for e, desc in node.keys)
        return f"Sort[keys={keys}]"
    if isinstance(node, Limit):
        return f"Limit[n={node.n}]"
    return type(node).__name__


def _aliased(expr, alias) -> str:
    text = format_expr(expr)
    return text if text == alias else f"{text} AS {alias}"


def format_expr(expr) -> str:
    if isinstance(expr, ColumnRef):
        return f"{expr.table}.{expr.name}" if expr.table else expr.name
    if isinstance(expr, Literal):
        if expr.value is None:
            return "NULL"
        if expr.dtype in (DType.STRING, DType.DATE):
            return f"'{expr.value}'"
        if expr.dtype == DType.BOOL:
            return "true" if expr.value else "false"
        return str(expr.value)
    if isinstance(expr, BinaryOp):
        return f"{_operand(expr.left)} {expr.op} {_operand(expr.right)}"
    if isinstance(expr, UnaryOp):
        if expr.op.upper().replace("_", " ") in ("IS NULL", "IS NOT NULL"):
            return f"{_operand(expr.operand)} {expr.op}"
        sep = " " if expr.op.isalpha() else ""
        return f"{expr.op}{sep}{_operand(expr.operand)}"
    if isinstance(expr, AggCall):
        return f"{expr.func}({'*' if expr.arg is None else format_expr(expr.arg)})"
    return repr(expr)


def _operand(expr) -> str:
    text = format_expr(expr)
    return f"({text})" if isinstance(expr, BinaryOp) else text
