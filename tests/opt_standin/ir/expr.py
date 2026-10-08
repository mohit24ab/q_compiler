from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ir.dtype import DType


class Expr:
    """Base of all expression nodes. Person A's ir/nodes.py and ir/printer.py import it."""


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
    left: Any
    right: Any


@dataclass(frozen=True)
class UnaryOp(Expr):
    op: str
    operand: Any


@dataclass(frozen=True)
class AggCall(Expr):
    func: str
    arg: Any | None
