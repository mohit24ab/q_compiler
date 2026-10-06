"""TEMPORARY stand-in for Person A's `ir` package.

Person C must not write to `ir/` (Contract §3), but Contract §9 says C builds against
hand-written IR trees until the real IR lands. This module mirrors the node shapes in
Contract §4 exactly as Person A's tests (tests/test_ir_nodes.py, tests/test_catalog.py)
describe them, including the `Scan.table_schema` extension.

It is only used when `import ir` fails — see runtime/_compat.py.
DELETE THIS FILE once `ir/` is on main.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class DType(Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    STRING = "STRING"
    BOOL = "BOOL"
    DATE = "DATE"


# ---------------------------------------------------------------- expressions

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


# ---------------------------------------------------------------- plan nodes

@dataclass(frozen=True)
class Scan:
    table: str
    columns: list[str] | None
    pushed_predicate: Any | None
    table_schema: list | None = field(default=None, compare=False)

    @property
    def children(self):
        return ()

    def replace_children(self, new_children):
        return replace(self)

    def schema(self):
        if self.table_schema is None:
            raise NotImplementedError("Scan.schema() needs table_schema")
        if self.columns is None:
            return list(self.table_schema)
        lookup = dict(self.table_schema)
        return [(c, lookup[c]) for c in self.columns]


@dataclass(frozen=True)
class Filter:
    child: Any
    predicate: Any

    @property
    def children(self):
        return (self.child,)

    def replace_children(self, new_children):
        return replace(self, child=new_children[0])


@dataclass(frozen=True)
class Project:
    child: Any
    exprs: list

    @property
    def children(self):
        return (self.child,)

    def replace_children(self, new_children):
        return replace(self, child=new_children[0])


@dataclass(frozen=True)
class Join:
    left: Any
    right: Any
    condition: Any
    kind: str

    @property
    def children(self):
        return (self.left, self.right)

    def replace_children(self, new_children):
        return replace(self, left=new_children[0], right=new_children[1])


@dataclass(frozen=True)
class Aggregate:
    child: Any
    group_keys: list
    aggs: list

    @property
    def children(self):
        return (self.child,)

    def replace_children(self, new_children):
        return replace(self, child=new_children[0])


@dataclass(frozen=True)
class Sort:
    child: Any
    keys: list

    @property
    def children(self):
        return (self.child,)

    def replace_children(self, new_children):
        return replace(self, child=new_children[0])


@dataclass(frozen=True)
class Limit:
    child: Any
    n: int

    @property
    def children(self):
        return (self.child,)

    def replace_children(self, new_children):
        return replace(self, child=new_children[0])
