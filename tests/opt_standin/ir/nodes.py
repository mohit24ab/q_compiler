"""Plan nodes per Contract §4, with the Scan.table_schema extension pinned by tests/test_catalog.py."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp

Schema = list[tuple[str, DType]]


class PlanNode:
    @property
    def children(self) -> tuple:
        raise NotImplementedError

    def replace_children(self, new_children) -> "PlanNode":
        raise NotImplementedError

    def schema(self) -> Schema:
        raise NotImplementedError


class _Unary(PlanNode):
    @property
    def children(self) -> tuple:
        return (self.child,)

    def replace_children(self, new_children):
        (child,) = new_children
        return dataclasses.replace(self, child=child)


@dataclass(frozen=True)
class Scan(PlanNode):
    table: str
    columns: list[str] | None
    pushed_predicate: Any | None
    table_schema: Schema | None = field(default=None, repr=False, compare=True)

    @property
    def children(self) -> tuple:
        return ()

    def replace_children(self, new_children):
        assert not tuple(new_children)
        return dataclasses.replace(self)

    def schema(self) -> Schema:
        if self.table_schema is None:
            raise NotImplementedError(f"Scan[{self.table}] has no table_schema")
        if self.columns is None:
            return list(self.table_schema)
        types = dict(self.table_schema)
        missing = [c for c in self.columns if c not in types]
        if missing:
            raise ValueError(f"unknown columns {missing} in table {self.table}")
        return [(c, types[c]) for c in self.columns]


@dataclass(frozen=True)
class Filter(_Unary):
    child: PlanNode
    predicate: Any

    def schema(self):
        return self.child.schema()


@dataclass(frozen=True)
class Project(_Unary):
    child: PlanNode
    exprs: list[tuple[Any, str]]

    def schema(self):
        child = self.child.schema()
        return [(alias, _infer(expr, child)) for expr, alias in self.exprs]


@dataclass(frozen=True)
class Join(PlanNode):
    left: PlanNode
    right: PlanNode
    condition: Any
    kind: str

    @property
    def children(self) -> tuple:
        return (self.left, self.right)

    def replace_children(self, new_children):
        left, right = new_children
        return dataclasses.replace(self, left=left, right=right)

    def schema(self):
        return self.left.schema() + self.right.schema()


@dataclass(frozen=True)
class Aggregate(_Unary):
    child: PlanNode
    group_keys: list[Any]
    aggs: list[tuple[AggCall, str]]

    def schema(self):
        child = self.child.schema()
        keys = [(k.name if isinstance(k, ColumnRef) else repr(k), _infer(k, child)) for k in self.group_keys]
        return keys + [(alias, _infer(call, child)) for call, alias in self.aggs]


@dataclass(frozen=True)
class Sort(_Unary):
    child: PlanNode
    keys: list[tuple[Any, bool]]

    def schema(self):
        return self.child.schema()


@dataclass(frozen=True)
class Limit(_Unary):
    child: PlanNode
    n: int

    def schema(self):
        return self.child.schema()


_COMPARISONS = {"=", "!=", "<>", "<", "<=", ">", ">=", "AND", "OR"}


def _infer(expr, schema: Schema) -> DType:
    if isinstance(expr, ColumnRef):
        return dict(schema)[expr.name]
    if isinstance(expr, Literal):
        return expr.dtype
    if isinstance(expr, UnaryOp):
        return DType.BOOL if expr.op.upper() == "NOT" else _infer(expr.operand, schema)
    if isinstance(expr, BinaryOp):
        if expr.op.upper() in _COMPARISONS:
            return DType.BOOL
        sides = {_infer(expr.left, schema), _infer(expr.right, schema)}
        return DType.FLOAT if DType.FLOAT in sides or expr.op == "/" else DType.INT
    if isinstance(expr, AggCall):
        if expr.func == "count":
            return DType.INT
        if expr.func == "avg":
            return DType.FLOAT
        return _infer(expr.arg, schema)
    raise TypeError(f"cannot infer type of {expr!r}")
