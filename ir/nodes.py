from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Literal as TypingLiteral

from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Expr, Literal, UnaryOp


def _infer_expr_dtype(expr: Expr | None, child_schema: list[tuple[str, DType]]) -> DType:
    """Minimal structural type inference for Phase A1/A2 schema derivation."""
    if expr is None:
        return DType.STRING

    if isinstance(expr, ColumnRef):
        if expr.table:
            qualified_name = f"{expr.table}.{expr.name}"
            for col_name, dtype in child_schema:
                if col_name == qualified_name:
                    return dtype
            for col_name, dtype in child_schema:
                if "." not in col_name and col_name == expr.name:
                    return dtype
        else:
            for col_name, dtype in child_schema:
                if col_name == expr.name:
                    return dtype
            for col_name, dtype in child_schema:
                if "." in col_name and col_name.split(".")[-1] == expr.name:
                    return dtype
        return DType.STRING

    if isinstance(expr, Literal):
        if expr.dtype is not None:
            return expr.dtype
        return DType.STRING

    if isinstance(expr, BinaryOp):
        op = expr.op.strip().upper()
        if op == "||":
            return DType.STRING
        if op in ("=", "!=", "<", ">", "<=", ">=", "AND", "OR"):
            return DType.BOOL
        left_dt = _infer_expr_dtype(expr.left, child_schema)
        right_dt = _infer_expr_dtype(expr.right, child_schema)

        if op == "/":
            return DType.FLOAT

        if op in ("+", "-", "*", "%"):
            if left_dt == DType.FLOAT or right_dt == DType.FLOAT:
                return DType.FLOAT
            if left_dt == DType.INT and right_dt == DType.INT:
                return DType.INT
            if left_dt == right_dt:
                return left_dt
            return DType.FLOAT

        if left_dt == right_dt:
            return left_dt
        return DType.FLOAT

    if isinstance(expr, UnaryOp):
        op_norm = " ".join(expr.op.strip().upper().replace("_", " ").split())
        if op_norm in ("NOT", "IS NULL", "IS NOT NULL"):
            return DType.BOOL
        return _infer_expr_dtype(expr.operand, child_schema)

    if isinstance(expr, AggCall):
        func = expr.func.lower()
        if func == "count":
            return DType.INT
        if func == "avg":
            return DType.FLOAT
        if expr.arg is not None:
            return _infer_expr_dtype(expr.arg, child_schema)
        return DType.INT

    return DType.STRING


class PlanNode(ABC):
    """Abstract base class for all relational algebra plan nodes."""

    @property
    @abstractmethod
    def children(self) -> tuple[PlanNode, ...]:
        """Returns child plan nodes as an immutable tuple."""
        pass

    @abstractmethod
    def schema(self) -> list[tuple[str, DType]]:
        """Returns output schema as a list of (column_name, DType) tuples."""
        pass

    @abstractmethod
    def replace_children(self, new_children: Sequence[PlanNode]) -> PlanNode:
        """Returns a NEW plan node with updated child nodes."""
        pass


@dataclass(frozen=True)
class Scan(PlanNode):
    table: str
    columns: list[str] | None
    pushed_predicate: Expr | None
    table_schema: list[tuple[str, DType]] | None = None

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return ()

    def schema(self) -> list[tuple[str, DType]]:
        if self.table_schema is None:
            raise NotImplementedError("Scan schema requires table_schema to be populated.")

        if self.columns is None:
            return self.table_schema

        schema_dict = dict(self.table_schema)
        result: list[tuple[str, DType]] = []
        for col in self.columns:
            if col not in schema_dict:
                raise ValueError(f"Column '{col}' requested in Scan projection does not exist in table_schema.")
            result.append((col, schema_dict[col]))
        return result

    def replace_children(self, new_children: Sequence[PlanNode]) -> Scan:
        if len(new_children) != 0:
            raise ValueError(f"Scan expects 0 children, got {len(new_children)}")
        return Scan(
            table=self.table,
            columns=self.columns,
            pushed_predicate=self.pushed_predicate,
            table_schema=self.table_schema,
        )


@dataclass(frozen=True)
class Filter(PlanNode):
    child: PlanNode
    predicate: Expr

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def schema(self) -> list[tuple[str, DType]]:
        return self.child.schema()

    def replace_children(self, new_children: Sequence[PlanNode]) -> Filter:
        if len(new_children) != 1:
            raise ValueError(f"Filter expects 1 child, got {len(new_children)}")
        return Filter(child=new_children[0], predicate=self.predicate)


@dataclass(frozen=True)
class Project(PlanNode):
    child: PlanNode
    exprs: list[tuple[Expr, str]]

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def schema(self) -> list[tuple[str, DType]]:
        child_schema = self.child.schema()
        result: list[tuple[str, DType]] = []
        for expr, alias in self.exprs:
            dtype = _infer_expr_dtype(expr, child_schema)
            result.append((alias, dtype))
        return result

    def replace_children(self, new_children: Sequence[PlanNode]) -> Project:
        if len(new_children) != 1:
            raise ValueError(f"Project expects 1 child, got {len(new_children)}")
        return Project(child=new_children[0], exprs=self.exprs)


@dataclass(frozen=True)
class Join(PlanNode):
    left: PlanNode
    right: PlanNode
    condition: Expr
    kind: TypingLiteral["inner", "left"]

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.left, self.right)

    def schema(self) -> list[tuple[str, DType]]:
        return self.left.schema() + self.right.schema()

    def replace_children(self, new_children: Sequence[PlanNode]) -> Join:
        if len(new_children) != 2:
            raise ValueError(f"Join expects 2 children, got {len(new_children)}")
        return Join(
            left=new_children[0],
            right=new_children[1],
            condition=self.condition,
            kind=self.kind,
        )


@dataclass(frozen=True)
class Aggregate(PlanNode):
    child: PlanNode
    group_keys: list[Expr]
    aggs: list[tuple[AggCall, str]]

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def schema(self) -> list[tuple[str, DType]]:
        child_schema = self.child.schema()
        result: list[tuple[str, DType]] = []
        for key in self.group_keys:
            if isinstance(key, ColumnRef):
                name = key.name
            else:
                name = str(key)
            dtype = _infer_expr_dtype(key, child_schema)
            result.append((name, dtype))
        for agg_call, alias in self.aggs:
            dtype = _infer_expr_dtype(agg_call, child_schema)
            result.append((alias, dtype))
        return result

    def replace_children(self, new_children: Sequence[PlanNode]) -> Aggregate:
        if len(new_children) != 1:
            raise ValueError(f"Aggregate expects 1 child, got {len(new_children)}")
        return Aggregate(
            child=new_children[0],
            group_keys=self.group_keys,
            aggs=self.aggs,
        )


@dataclass(frozen=True)
class Sort(PlanNode):
    child: PlanNode
    keys: list[tuple[Expr, bool]]

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def schema(self) -> list[tuple[str, DType]]:
        return self.child.schema()

    def replace_children(self, new_children: Sequence[PlanNode]) -> Sort:
        if len(new_children) != 1:
            raise ValueError(f"Sort expects 1 child, got {len(new_children)}")
        return Sort(child=new_children[0], keys=self.keys)


@dataclass(frozen=True)
class Limit(PlanNode):
    child: PlanNode
    n: int

    @property
    def children(self) -> tuple[PlanNode, ...]:
        return (self.child,)

    def schema(self) -> list[tuple[str, DType]]:
        return self.child.schema()

    def replace_children(self, new_children: Sequence[PlanNode]) -> Limit:
        if len(new_children) != 1:
            raise ValueError(f"Limit expects 1 child, got {len(new_children)}")
        return Limit(child=new_children[0], n=self.n)