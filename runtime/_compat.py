"""Single import point for IR classes used by Person C's code and tests.

Re-exports Person A's `ir` package. (Until `ir/` landed on main this fell back to a
stand-in copy, runtime/_ir_standin.py; both are gone now that the real IR is complete.)
"""
from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort
from ir.printer import format_plan

__all__ = [
    "DType", "AggCall", "BinaryOp", "ColumnRef", "Literal", "UnaryOp",
    "Aggregate", "Filter", "Join", "Limit", "Project", "Scan", "Sort",
    "format_plan",
]
