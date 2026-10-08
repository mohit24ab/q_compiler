from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal as TypingLiteral

from ir.dtype import DType


class Expr:
    """Base class for all relational algebra expression nodes."""

    pass


@dataclass(frozen=True)
class ColumnRef(Expr):
    table: str | None
    name: str


@dataclass(frozen=True)
class Literal(Expr):
    value: Any
    dtype: DType


@dataclass(frozen=True)
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    operand: Expr


@dataclass(frozen=True)
class AggCall(Expr):
    func: TypingLiteral["sum", "count", "avg", "min", "max"]
    arg: Expr | None