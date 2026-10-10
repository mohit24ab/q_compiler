from __future__ import annotations

from typing import Any, Sequence
from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Expr, Literal, UnaryOp
from ir.nodes import PlanNode
from ir.printer import format_expr
from frontend.resolver import Resolver, SemanticError


class SemanticTypeError(SemanticError, TypeError):
    """Exception raised for static typing errors during semantic analysis."""

    pass


class TypeChecker:
    """Enforces static relational typing and relational semantic rules."""

    def is_numeric(self, dtype: DType) -> bool:
        """Checks if a DType is numeric (INT or FLOAT)."""
        return dtype in (DType.INT, DType.FLOAT)

    def are_compatible_for_comparison(self, left: DType, right: DType) -> bool:
        """Determines if two DTypes can be compared with comparison operators."""
        if left == right:
            return True
        # Numeric coercion compatibility
        if {left, right} <= {DType.INT, DType.FLOAT}:
            return True
        # Date and string compatibility
        if {left, right} <= {DType.DATE, DType.STRING}:
            return True
        return False

    def find_aggregates(self, expr: Expr) -> list[AggCall]:
        """Finds all AggCall nodes recursively within an expression."""
        aggs: list[AggCall] = []

        def _walk(e: Expr) -> None:
            if isinstance(e, AggCall):
                aggs.append(e)
                if e.arg is not None:
                    _walk(e.arg)
            elif isinstance(e, BinaryOp):
                _walk(e.left)
                _walk(e.right)
            elif isinstance(e, UnaryOp):
                _walk(e.operand)

        _walk(expr)
        return aggs

    def get_non_aggregated_columns(self, expr: Expr) -> list[ColumnRef]:
        """Extracts all ColumnRef nodes that are NOT enclosed inside an AggCall."""
        cols: list[ColumnRef] = []

        def _walk(e: Expr) -> None:
            if isinstance(e, AggCall):
                return
            if isinstance(e, ColumnRef):
                cols.append(e)
            elif isinstance(e, BinaryOp):
                _walk(e.left)
                _walk(e.right)
            elif isinstance(e, UnaryOp):
                _walk(e.operand)

        _walk(expr)
        return cols

    def _lookup_column_type(self, col: ColumnRef, context: Any) -> DType:
        """Looks up the DType of a ColumnRef across provided contexts."""
        if isinstance(context, tuple) and len(context) == 2:
            ctx1, ctx2 = context
            try:
                return self._lookup_column_type(col, ctx1)
            except Exception:
                return self._lookup_column_type(col, ctx2)

        if isinstance(context, PlanNode):
            context = context.schema()

        if isinstance(context, (list, tuple)):
            for name, dt in context:
                if name == col.name or (col.table and name == f"{col.table}.{col.name}"):
                    return dt
                if "." in name and name.split(".")[-1] == col.name:
                    return dt

        if isinstance(context, Resolver):
            try:
                return context.get_column_type(col)
            except Exception:
                pass

        if isinstance(context, dict):
            if col.name in context:
                return context[col.name]
            if col.table and f"{col.table}.{col.name}" in context:
                return context[f"{col.table}.{col.name}"]

        raise SemanticError(f"Cannot determine type for column '{col.name}'.")

    def infer_type(
        self,
        expr: Expr,
        context: Any = None,
    ) -> DType:
        """Recursively infers DType and validates static typing for an expression."""
        if isinstance(expr, Literal):
            return expr.dtype

        if isinstance(expr, ColumnRef):
            return self._lookup_column_type(expr, context)

        if isinstance(expr, AggCall):
            func = expr.func.lower()
            if func == "count":
                if expr.arg is not None:
                    self.infer_type(expr.arg, context)
                return DType.INT
            elif func in ("sum", "avg"):
                if expr.arg is None:
                    raise SemanticError(f"Aggregate '{func}' requires an argument.")
                arg_type = self.infer_type(expr.arg, context)
                if not self.is_numeric(arg_type):
                    raise SemanticTypeError(
                        f"Aggregate {func.upper()} requires numeric argument, got {arg_type} in '{format_expr(expr)}'."
                    )
                if func == "avg":
                    return DType.FLOAT
                return arg_type
            elif func in ("min", "max"):
                if expr.arg is None:
                    raise SemanticError(f"Aggregate '{func}' requires an argument.")
                return self.infer_type(expr.arg, context)
            else:
                raise SemanticError(f"Unsupported aggregate function: '{func}'.")

        if isinstance(expr, UnaryOp):
            op_norm = " ".join(expr.op.strip().upper().replace("_", " ").split())
            if op_norm in ("IS NULL", "IS NOT NULL"):
                self.infer_type(expr.operand, context)
                return DType.BOOL
            elif op_norm == "NOT":
                operand_type = self.infer_type(expr.operand, context)
                if operand_type != DType.BOOL:
                    raise SemanticTypeError(
                        f"Logical operator 'NOT' requires boolean operand, got {operand_type} in '{format_expr(expr)}'."
                    )
                return DType.BOOL
            elif op_norm == "-":
                operand_type = self.infer_type(expr.operand, context)
                if not self.is_numeric(operand_type):
                    raise SemanticTypeError(
                        f"Unary operator '-' requires numeric operand, got {operand_type} in '{format_expr(expr)}'."
                    )
                return operand_type
            else:
                raise SemanticError(f"Unsupported unary operator: '{expr.op}'.")

        if isinstance(expr, BinaryOp):
            left_type = self.infer_type(expr.left, context)
            right_type = self.infer_type(expr.right, context)
            op = expr.op.strip().upper()

            # Arithmetic operators: +, -, *, /, %
            if op in ("+", "-", "*", "/", "%"):
                if not (self.is_numeric(left_type) and self.is_numeric(right_type)):
                    raise SemanticTypeError(
                        f"Offending expression '{format_expr(expr)}': operator '{expr.op}' "
                        f"requires numeric operands, got {left_type} and {right_type}."
                    )
                if op == "/":
                    return DType.FLOAT
                if left_type == DType.FLOAT or right_type == DType.FLOAT:
                    return DType.FLOAT
                return DType.INT

            # Logical operators: AND, OR
            elif op in ("AND", "OR"):
                if left_type != DType.BOOL or right_type != DType.BOOL:
                    raise SemanticTypeError(
                        f"Offending expression '{format_expr(expr)}': logical operator '{expr.op}' "
                        f"requires boolean operands, got {left_type} and {right_type}."
                    )
                return DType.BOOL

            # Comparison operators: =, !=, <, >, <=, >=
            elif op in ("=", "!=", "<", ">", "<=", ">="):
                if not self.are_compatible_for_comparison(left_type, right_type):
                    raise SemanticTypeError(
                        f"Offending expression '{format_expr(expr)}': comparison operator '{expr.op}' "
                        f"cannot compare incompatible types {left_type} and {right_type}."
                    )
                return DType.BOOL

            # String concatenation operator: ||
            elif op == "||":
                valid_types = (DType.STRING, DType.INT, DType.FLOAT, DType.BOOL, DType.DATE)
                if left_type not in valid_types or right_type not in valid_types:
                    raise SemanticTypeError(
                        f"Offending expression '{format_expr(expr)}': operator '{expr.op}' "
                        f"requires string or coercible operands, got {left_type} and {right_type}."
                    )
                return DType.STRING

            else:
                raise SemanticError(f"Unsupported binary operator: '{expr.op}'.")

        raise SemanticError(f"Unsupported expression node: {type(expr).__name__}")

    def check_boolean_condition(
        self,
        expr: Expr,
        context: Any = None,
        context_name: str = "Filter",
    ) -> None:
        """Validates that a predicate or condition expression evaluates to DType.BOOL."""
        dtype = self.infer_type(expr, context)
        if dtype != DType.BOOL:
            raise SemanticTypeError(
                f"{context_name} condition must evaluate to BOOL, got {dtype} in '{format_expr(expr)}'."
            )

    def check_no_aggregates(
        self,
        expr: Expr,
        context_name: str = "WHERE",
    ) -> None:
        """Ensures that no aggregate functions appear in clauses where they are disallowed."""
        aggs = self.find_aggregates(expr)
        if aggs:
            agg_names = ", ".join(f"{a.func.upper()}()" for a in aggs)
            raise SemanticError(
                f"Aggregate function {agg_names} is not allowed in {context_name} clause: '{format_expr(expr)}'."
            )

    def is_column_in_group_keys(
        self,
        col: ColumnRef,
        group_keys: Sequence[Expr],
        agg_aliases: set[str] | None = None,
        proj_aliases: dict[str, Expr] | None = None,
    ) -> bool:
        """Checks if a column reference is covered by GROUP BY keys or aggregate aliases."""
        # 1. Direct match in group keys
        for gk in group_keys:
            if isinstance(gk, ColumnRef):
                if col.name == gk.name:
                    if col.table is None or gk.table is None or col.table == gk.table:
                        return True
            elif col == gk:
                return True

        # 2. Match in aggregate aliases
        if agg_aliases and col.name in agg_aliases and col.table is None:
            return True

        # 3. Match in projection aliases
        if proj_aliases and col.name in proj_aliases and col.table is None:
            target = proj_aliases[col.name]
            if isinstance(target, AggCall):
                return True
            if isinstance(target, ColumnRef) and target.name == col.name:
                return False
            non_aggs = self.get_non_aggregated_columns(target)
            if not non_aggs:
                return True
            return all(
                self.is_column_in_group_keys(c, group_keys, agg_aliases, None)
                for c in non_aggs
            )

        return False

    def check_group_by_projections(
        self,
        project_exprs: Sequence[tuple[Expr, str]],
        group_keys: Sequence[Expr],
        agg_aliases: set[str] | None = None,
        proj_aliases: dict[str, Expr] | None = None,
    ) -> None:
        """Ensures that every non-aggregated column in SELECT projections exists in group_keys."""
        for expr, alias in project_exprs:
            if isinstance(expr, ColumnRef) and agg_aliases and expr.name in agg_aliases:
                continue
            non_agg_cols = self.get_non_aggregated_columns(expr)
            for col in non_agg_cols:
                if not self.is_column_in_group_keys(col, group_keys, agg_aliases, proj_aliases):
                    raise SemanticError(
                        f"Column '{col.name}' must appear in the GROUP BY clause or be used in an aggregate function."
                    )

    def check_group_by_order_by(
        self,
        sort_keys: Sequence[tuple[Expr, bool]],
        group_keys: Sequence[Expr],
        agg_aliases: set[str] | None = None,
        proj_aliases: dict[str, Expr] | None = None,
    ) -> None:
        """Ensures that every non-aggregated column in ORDER BY exists in group_keys."""
        for expr, _ in sort_keys:
            if isinstance(expr, ColumnRef) and agg_aliases and expr.name in agg_aliases:
                continue
            non_agg_cols = self.get_non_aggregated_columns(expr)
            for col in non_agg_cols:
                if not self.is_column_in_group_keys(col, group_keys, agg_aliases, proj_aliases):
                    raise SemanticError(
                        f"Column '{col.name}' in ORDER BY must appear in the GROUP BY clause or be used in an aggregate function."
                    )
