from __future__ import annotations

from typing import Any, Sequence
import sqlglot
from sqlglot import exp

from catalog.catalog import Catalog
from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Expr, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, PlanNode, Project, Scan, Sort
from ir.printer import format_expr


class Scope:
    """Manages active relation schemas and resolves column references."""

    def __init__(self) -> None:
        self.schemas: dict[str, list[tuple[str, DType]]] = {}
        self.table_to_alias: dict[str, str] = {}
        self.relation_order: list[str] = []

    def add_relation(
        self, table_name: str, schema: list[tuple[str, DType]], alias: str | None = None
    ) -> None:
        rel_key = alias if alias else table_name
        self.schemas[rel_key] = schema
        self.relation_order.append(rel_key)
        if alias:
            self.table_to_alias[table_name] = alias

    def resolve_column(self, name: str, table: str | None = None) -> ColumnRef:
        if table is not None:
            target_rel = table
            if target_rel not in self.schemas:
                if table in self.table_to_alias and self.table_to_alias[table] in self.schemas:
                    target_rel = self.table_to_alias[table]
                else:
                    raise ValueError(f"Table '{table}' not found in active scope.")

            rel_schema = self.schemas[target_rel]
            col_names = [col_name for col_name, _ in rel_schema]
            if name not in col_names:
                raise ValueError(f"Column '{name}' not found in table '{table}'.")
            return ColumnRef(table=target_rel, name=name)
        else:
            matches: list[str] = []
            for rel_name, rel_schema in self.schemas.items():
                col_names = [col_name for col_name, _ in rel_schema]
                if name in col_names:
                    matches.append(rel_name)

            if len(matches) == 0:
                raise ValueError(f"Unknown column '{name}'.")
            if len(matches) > 1:
                raise ValueError(
                    f"Ambiguous column reference '{name}' found across active tables: {matches}."
                )
            return ColumnRef(table=matches[0], name=name)


_BINARY_OPS = {
    exp.Add: "+",
    exp.Sub: "-",
    exp.Mul: "*",
    exp.Div: "/",
    exp.Mod: "%",
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.GT: ">",
    exp.GTE: ">=",
    exp.And: "AND",
    exp.Or: "OR",
}

_AGG_FUNCS = {"sum", "count", "avg", "min", "max"}


def _bind_agg(e: exp.AggFunc, scope: Scope) -> AggCall:
    func_name = e.key.lower()
    if func_name not in _AGG_FUNCS:
        raise ValueError(f"Unsupported aggregate function: '{e.key}'.")

    if func_name == "count":
        if isinstance(e.this, exp.Star) or e.this is None:
            return AggCall(func="count", arg=None)
        arg = _bind_expr(e.this, scope)
        return AggCall(func="count", arg=arg)

    if e.this is None:
        raise ValueError(f"Aggregate '{func_name}' requires an argument.")

    arg = _bind_expr(e.this, scope)
    return AggCall(func=func_name, arg=arg)  # type: ignore[arg-type]


def _bind_expr(e: exp.Expression, scope: Scope) -> Expr:
    if isinstance(e, exp.Paren):
        return _bind_expr(e.this, scope)

    if isinstance(e, exp.Column):
        table_name = e.table if e.table else None
        col_name = e.name
        return scope.resolve_column(name=col_name, table=table_name)

    if isinstance(e, exp.Boolean):
        return Literal(value=bool(e.this), dtype=DType.BOOL)

    if isinstance(e, exp.Null):
        return Literal(value=None, dtype=DType.STRING)

    if isinstance(e, exp.Cast):
        to_type = str(e.to.this).upper()
        if to_type == "DATE":
            val = e.this.this if isinstance(e.this, exp.Literal) else str(e.this)
            return Literal(value=val, dtype=DType.DATE)
    if isinstance(e, exp.Date):
        val = e.this.this if isinstance(e.this, exp.Literal) else str(e.this)
        return Literal(value=val, dtype=DType.DATE)

    if isinstance(e, exp.Literal):
        if e.is_string:
            return Literal(value=e.this, dtype=DType.STRING)
        if e.is_number:
            if "." in e.this:
                return Literal(value=float(e.this), dtype=DType.FLOAT)
            return Literal(value=int(e.this), dtype=DType.INT)
        return Literal(value=e.this, dtype=DType.STRING)

    if isinstance(e, exp.Is):
        if isinstance(e.expression, exp.Null):
            return UnaryOp(op="IS NULL", operand=_bind_expr(e.this, scope))
    if isinstance(e, exp.Not):
        if isinstance(e.this, exp.Is) and isinstance(e.this.expression, exp.Null):
            return UnaryOp(op="IS NOT NULL", operand=_bind_expr(e.this.this, scope))
        return UnaryOp(op="NOT", operand=_bind_expr(e.this, scope))

    if isinstance(e, exp.Neg):
        return UnaryOp(op="-", operand=_bind_expr(e.this, scope))

    if type(e) in _BINARY_OPS:
        op = _BINARY_OPS[type(e)]
        left = _bind_expr(e.this, scope)
        right = _bind_expr(e.expression, scope)
        return BinaryOp(op=op, left=left, right=right)

    if isinstance(e, exp.AggFunc):
        return _bind_agg(e, scope)

    raise ValueError(f"Unsupported expression AST node: {type(e).__name__} ({e})")


def _get_agg_alias(agg: AggCall) -> str:
    if agg.arg is None:
        arg_str = "*"
    elif isinstance(agg.arg, ColumnRef):
        arg_str = agg.arg.name
    else:
        arg_str = format_expr(agg.arg)
    return f"{agg.func}({arg_str})"


def parse_and_bind(sql: str, catalog: Catalog) -> PlanNode:
    """Parses a SQL SELECT query into a canonical unoptimized IR plan node hierarchy."""
    try:
        ast = sqlglot.parse_one(sql)
    except Exception as exc:
        raise ValueError(f"SQL parsing error: {exc}") from exc

    if not isinstance(ast, exp.Select):
        raise ValueError(f"Expected a SELECT query, got: {type(ast).__name__}")

    from_clause = ast.args.get("from_")
    if from_clause is None or from_clause.this is None:
        raise ValueError("Query must specify a FROM clause.")

    from_table = from_clause.this
    if not isinstance(from_table, exp.Table):
        raise ValueError("Unsupported FROM expression: expected a table.")

    table_name = from_table.name
    alias = from_table.alias if from_table.alias else None

    scope = Scope()

    try:
        table_schema = catalog.schema(table_name)
    except KeyError as exc:
        raise ValueError(f"Table '{table_name}' not found in catalog.") from exc

    scope.add_relation(table_name=table_name, schema=table_schema, alias=alias)

    current_plan: PlanNode = Scan(
        table=table_name,
        columns=None,
        pushed_predicate=None,
        table_schema=table_schema,
    )

    for join_ast in ast.args.get("joins", []):
        join_table = join_ast.this
        if not isinstance(join_table, exp.Table):
            raise ValueError("Unsupported JOIN expression: expected a table.")

        joined_name = join_table.name
        joined_alias = join_table.alias if join_table.alias else None

        try:
            joined_schema = catalog.schema(joined_name)
        except KeyError as exc:
            raise ValueError(f"Table '{joined_name}' not found in catalog.") from exc

        scope.add_relation(table_name=joined_name, schema=joined_schema, alias=joined_alias)

        right_scan = Scan(
            table=joined_name,
            columns=None,
            pushed_predicate=None,
            table_schema=joined_schema,
        )

        on_ast = join_ast.args.get("on")
        if on_ast is not None:
            condition = _bind_expr(on_ast, scope)
        else:
            condition = Literal(value=True, dtype=DType.BOOL)

        side_str = (join_ast.side or "").upper()
        kind_str = (join_ast.kind or "").upper()
        kind = "left" if (side_str == "LEFT" or kind_str == "LEFT") else "inner"

        current_plan = Join(
            left=current_plan,
            right=right_scan,
            condition=condition,
            kind=kind,
        )

    where_clause = ast.args.get("where")
    if where_clause is not None:
        where_pred = _bind_expr(where_clause.this, scope)
        current_plan = Filter(child=current_plan, predicate=where_pred)

    group_clause = ast.args.get("group")
    having_clause = ast.args.get("having")
    select_items = ast.expressions

    has_group_by = group_clause is not None
    has_aggs = any(e.find(exp.AggFunc) is not None for e in select_items) or (having_clause is not None)

    if has_group_by or has_aggs:
        group_keys: list[Expr] = []
        if group_clause is not None:
            for ge in group_clause.expressions:
                group_keys.append(_bind_expr(ge, scope))

        aggs: list[tuple[AggCall, str]] = []
        registered_aggs: dict[str, str] = {}

        def register_agg_call(agg_node: exp.AggFunc, explicit_alias: str | None = None) -> str:
            bound_agg = _bind_agg(agg_node, scope)
            alias = explicit_alias if explicit_alias else _get_agg_alias(bound_agg)
            key = repr(agg_node)
            if key not in registered_aggs:
                registered_aggs[key] = alias
                aggs.append((bound_agg, alias))
            return registered_aggs[key]

        for se in select_items:
            if isinstance(se, exp.Alias) and isinstance(se.this, exp.AggFunc):
                register_agg_call(se.this, se.alias)
            else:
                for agg_node in se.find_all(exp.AggFunc):
                    register_agg_call(agg_node)

        if having_clause is not None:
            for agg_node in having_clause.this.find_all(exp.AggFunc):
                register_agg_call(agg_node)

        current_plan = Aggregate(
            child=current_plan,
            group_keys=group_keys,
            aggs=aggs,
        )

        if having_clause is not None:
            having_pred = _bind_expr(having_clause.this, scope)
            current_plan = Filter(child=current_plan, predicate=having_pred)

    project_exprs: list[tuple[Expr, str]] = []
    is_wildcard = any(isinstance(e, exp.Star) for e in select_items)

    if is_wildcard:
        for rel_name in scope.relation_order:
            for col_name, _ in scope.schemas[rel_name]:
                project_exprs.append((ColumnRef(table=rel_name, name=col_name), col_name))
    else:
        for se in select_items:
            if isinstance(se, exp.Alias):
                alias = se.alias
                expr_ast = se.this
            elif isinstance(se, exp.Column):
                alias = se.name
                expr_ast = se
            elif isinstance(se, exp.AggFunc):
                bound_agg = _bind_agg(se, scope)
                alias = _get_agg_alias(bound_agg)
                expr_ast = se
            else:
                alias = se.sql()
                expr_ast = se

            if (has_group_by or has_aggs) and isinstance(expr_ast, exp.AggFunc):
                col_ref = ColumnRef(table=None, name=alias)
                project_exprs.append((col_ref, alias))
            else:
                bound_e = _bind_expr(expr_ast, scope)
                project_exprs.append((bound_e, alias))

    current_plan = Project(child=current_plan, exprs=project_exprs)

    order_clause = ast.args.get("order")
    if order_clause is not None:
        sort_keys: list[tuple[Expr, bool]] = []
        proj_aliases = {alias: expr for expr, alias in project_exprs}
        for ordered in order_clause.expressions:
            is_desc = bool(ordered.args.get("desc", False))
            order_ast = ordered.this
            if isinstance(order_ast, exp.Column) and not order_ast.table and order_ast.name in proj_aliases:
                try:
                    sort_expr = scope.resolve_column(name=order_ast.name)
                except ValueError:
                    sort_expr = ColumnRef(table=None, name=order_ast.name)
            else:
                sort_expr = _bind_expr(order_ast, scope)
            sort_keys.append((sort_expr, is_desc))

        current_plan = Sort(child=current_plan, keys=sort_keys)

    limit_clause = ast.args.get("limit")
    if limit_clause is not None:
        limit_expr = limit_clause.expression
        if not isinstance(limit_expr, exp.Literal) or not limit_expr.is_number:
            raise ValueError(f"Expected integer numeric LIMIT, got: {limit_expr}")
        n = int(limit_expr.this)
        current_plan = Limit(child=current_plan, n=n)

    return current_plan
