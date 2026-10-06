from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ir.dtype import DType


@dataclass(frozen=True)
class ColumnRef:
    table: str | None
    name: str


@dataclass(frozen=True)
class Literal:
    value: Any
    dtype: DType


@dataclass(frozen=True)
class BinaryOp:
    op: str
    left: Any
    right: Any


@dataclass(frozen=True)
class UnaryOp:
    op: str
    operand: Any


@dataclass(frozen=True)
class AggCall:
    func: str
    arg: Any | None
