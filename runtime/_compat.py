"""Single import point for IR classes used by Person C's code and tests.

Uses Person A's real `ir` package when it exists; otherwise falls back to the
temporary stand-in. When `ir/` lands on main, nothing else needs to change.
"""
try:
    from ir.dtype import DType
    from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp
    from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort
    USING_STANDIN = False
except ImportError:  # Person A's IR is not on this branch yet
    from runtime._ir_standin import (  # noqa: F401
        AggCall, Aggregate, BinaryOp, ColumnRef, DType, Filter, Join, Limit,
        Literal, Project, Scan, Sort, UnaryOp,
    )
    USING_STANDIN = True

__all__ = [
    "DType", "AggCall", "BinaryOp", "ColumnRef", "Literal", "UnaryOp",
    "Aggregate", "Filter", "Join", "Limit", "Project", "Scan", "Sort",
    "USING_STANDIN",
]
