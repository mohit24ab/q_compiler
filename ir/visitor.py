from __future__ import annotations

from typing import Callable

from ir.expr import AggCall, BinaryOp, Expr, UnaryOp
from ir.nodes import PlanNode


def transform_post_order(node: PlanNode, fn: Callable[[PlanNode], PlanNode]) -> PlanNode:
    """Recursively transforms a plan node tree in post-order (bottom-up)."""
    if not node.children:
        return fn(node)

    new_children = tuple(transform_post_order(child, fn) for child in node.children)
    new_node = node.replace_children(new_children)
    return fn(new_node)


def transform_expr_post_order(expr: Expr, fn: Callable[[Expr], Expr]) -> Expr:
    """Recursively transforms an expression node tree in post-order (bottom-up)."""
    if isinstance(expr, BinaryOp):
        new_left = transform_expr_post_order(expr.left, fn)
        new_right = transform_expr_post_order(expr.right, fn)
        new_expr = BinaryOp(op=expr.op, left=new_left, right=new_right)
        return fn(new_expr)
    elif isinstance(expr, UnaryOp):
        new_operand = transform_expr_post_order(expr.operand, fn)
        new_expr = UnaryOp(op=expr.op, operand=new_operand)
        return fn(new_expr)
    elif isinstance(expr, AggCall):
        if expr.arg is not None:
            new_arg = transform_expr_post_order(expr.arg, fn)
            new_expr = AggCall(func=expr.func, arg=new_arg)
            return fn(new_expr)
        return fn(expr)
    else:
        return fn(expr)