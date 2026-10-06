from __future__ import annotations

from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Expr, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, PlanNode, Project, Scan, Sort


def format_expr(expr: Expr) -> str:
    """Formats an expression node into a clean string representation."""
    if isinstance(expr, ColumnRef):
        if expr.table:
            return f"{expr.table}.{expr.name}"
        return expr.name

    if isinstance(expr, Literal):
        if expr.dtype in (DType.STRING, DType.DATE):
            return f"'{expr.value}'"
        return str(expr.value)

    if isinstance(expr, BinaryOp):
        return f"{format_expr(expr.left)} {expr.op} {format_expr(expr.right)}"

    if isinstance(expr, UnaryOp):
        op_norm = " ".join(expr.op.strip().upper().replace("_", " ").split())
        if op_norm == "NOT":
            return f"NOT {format_expr(expr.operand)}"
        if op_norm == "IS NULL":
            return f"{format_expr(expr.operand)} IS NULL"
        if op_norm == "IS NOT NULL":
            return f"{format_expr(expr.operand)} IS NOT NULL"
        return f"{expr.op}{format_expr(expr.operand)}"

    if isinstance(expr, AggCall):
        if expr.arg is None:
            return f"{expr.func}(*)"
        return f"{expr.func}({format_expr(expr.arg)})"

    return str(expr)


def _format_node_header(node: PlanNode) -> str:
    """Formats the node type and attributes for a single line."""
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
        formatted_exprs = []
        for e, alias in node.exprs:
            formatted_e = format_expr(e)
            if formatted_e == alias:
                formatted_exprs.append(alias)
            else:
                formatted_exprs.append(f"{formatted_e} AS {alias}")
        return f"Project[{', '.join(formatted_exprs)}]"

    if isinstance(node, Join):
        return f"Join[kind={node.kind}, cond={format_expr(node.condition)}]"

    if isinstance(node, Aggregate):
        parts = []
        if node.group_keys:
            group_str = ", ".join(format_expr(k) for k in node.group_keys)
            parts.append(f"group={group_str}")
        if node.aggs:
            aggs_str = ", ".join(f"{format_expr(a)} AS {alias}" for a, alias in node.aggs)
            parts.append(f"aggs={aggs_str}")
        return f"Aggregate[{', '.join(parts)}]"

    if isinstance(node, Sort):
        keys_str = ", ".join(
            f"{format_expr(k)} {'DESC' if desc else 'ASC'}" for k, desc in node.keys
        )
        return f"Sort[keys={keys_str}]"

    if isinstance(node, Limit):
        return f"Limit[n={node.n}]"

    return f"{node.__class__.__name__}[]"


def format_plan(node: PlanNode, depth: int = 0) -> str:
    """Renders a plan tree as a deterministic indented string representation."""
    indent = " " * (depth * 2)
    line = f"{indent}{_format_node_header(node)}"
    child_lines = [format_plan(child, depth + 1) for child in node.children]
    if child_lines:
        return f"{line}\n" + "\n".join(child_lines)
    return line


def print_plan(node: PlanNode) -> None:
    """Prints the plan tree to stdout."""
    print(format_plan(node))