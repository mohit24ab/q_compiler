from __future__ import annotations

from typing import Any, Sequence
import sqlglot
from sqlglot import exp

from catalog.catalog import Catalog
from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Expr, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, PlanNode, Project, Scan, Sort
from ir.printer import format_expr
from ir.visitor import transform_expr_post_order
from frontend.resolver import Resolver, SemanticError
from frontend.typecheck import TypeChecker, SemanticTypeError

Scope = Resolver

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
    exp.DPipe: "||",
}

_AGG_FUNCS = {"sum", "count", "avg", "min", "max"}


def _bind_agg(
    e: exp.AggFunc,
    scope: Resolver,
    typechecker: TypeChecker | None = None,
) -> AggCall:
    func_name = e.key.lower()
    if func_name not in _AGG_FUNCS:
        raise ValueError(f"Unsupported aggregate function: '{e.key}'.")

    if func_name == "count":
        if isinstance(e.this, exp.Star) or e.this is None:
            agg = AggCall(func="count", arg=None)
        else:
            arg = _bind_expr(e.this, scope, typechecker)
            agg = AggCall(func="count", arg=arg)
    else:
        if e.this is None:
            raise ValueError(f"Aggregate '{func_name}' requires an argument.")
        arg = _bind_expr(e.this, scope, typechecker)
        agg = AggCall(func=func_name, arg=arg)  # type: ignore[arg-type]

    if typechecker is not None:
        typechecker.infer_type(agg, scope)
    return agg


def _bind_expr(
    e: exp.Expression,
    scope: Resolver,
    typechecker: TypeChecker | None = None,
) -> Expr:
    if isinstance(e, exp.Paren):
        return _bind_expr(e.this, scope, typechecker)

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
        if "DATE" in to_type or (hasattr(e.to, "is_type") and e.to.is_type("date")):
            val = e.this.this if isinstance(e.this, exp.Literal) else str(e.this)
            return Literal(value=str(val), dtype=DType.DATE)
    if isinstance(e, exp.Date):
        val = e.this.this if isinstance(e.this, exp.Literal) else str(e.this)
        return Literal(value=str(val), dtype=DType.DATE)
    if isinstance(e, exp.Anonymous) and e.name.upper() == "DATE":
        if e.expressions:
            arg = e.expressions[0]
            val = arg.this if isinstance(arg, exp.Literal) else str(arg)
            return Literal(value=str(val), dtype=DType.DATE)

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
            operand = _bind_expr(e.this, scope, typechecker)
            res_op = UnaryOp(op="IS NULL", operand=operand)
            if typechecker is not None:
                typechecker.infer_type(res_op, scope)
            return res_op
    if isinstance(e, exp.Not):
        if isinstance(e.this, exp.Is) and isinstance(e.this.expression, exp.Null):
            operand = _bind_expr(e.this.this, scope, typechecker)
            res_op = UnaryOp(op="IS NOT NULL", operand=operand)
            if typechecker is not None:
                typechecker.infer_type(res_op, scope)
            return res_op
        operand = _bind_expr(e.this, scope, typechecker)
        res_op = UnaryOp(op="NOT", operand=operand)
        if typechecker is not None:
            typechecker.infer_type(res_op, scope)
        return res_op

    if isinstance(e, exp.Neg):
        operand = _bind_expr(e.this, scope, typechecker)
        res_op = UnaryOp(op="-", operand=operand)
        if typechecker is not None:
            typechecker.infer_type(res_op, scope)
        return res_op

    if isinstance(e, exp.DPipe):
        left = _bind_expr(e.this, scope, typechecker)
        right = _bind_expr(e.expression, scope, typechecker)
        bin_op = BinaryOp(op="||", left=left, right=right)
        if typechecker is not None:
            typechecker.infer_type(bin_op, scope)
        return bin_op

    if isinstance(e, exp.Concat):
        args = ([e.this] if e.this is not None else []) + (e.expressions or [])
        if not args:
            return Literal(value="", dtype=DType.STRING)
        res = _bind_expr(args[0], scope, typechecker)
        for arg in args[1:]:
            right = _bind_expr(arg, scope, typechecker)
            res = BinaryOp(op="||", left=res, right=right)
            if typechecker is not None:
                typechecker.infer_type(res, scope)
        return res

    if type(e) in _BINARY_OPS:
        op = _BINARY_OPS[type(e)]
        left = _bind_expr(e.this, scope, typechecker)
        right = _bind_expr(e.expression, scope, typechecker)
        bin_op = BinaryOp(op=op, left=left, right=right)
        if typechecker is not None:
            typechecker.infer_type(bin_op, scope)
        return bin_op

    if isinstance(e, exp.AggFunc):
        return _bind_agg(e, scope, typechecker)

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

    # Check for unsupported OFFSET modifier
    if ast.args.get("offset") is not None:
        raise SemanticError("OFFSET is not supported by the query compiler.")

    is_distinct = ast.args.get("distinct") is not None

    from_clause = ast.args.get("from_")
    if from_clause is None or from_clause.this is None:
        raise ValueError("Query must specify a FROM clause.")

    from_table = from_clause.this
    if not isinstance(from_table, exp.Table):
        raise ValueError("Unsupported FROM expression: expected a table.")

    table_name = from_table.name
    alias = from_table.alias if from_table.alias else None

    resolver = Resolver(catalog)
    typechecker = TypeChecker()

    try:
        table_schema = resolver.add_table(table_name=table_name, alias=alias)
    except KeyError as exc:
        raise ValueError(f"Table '{table_name}' not found in catalog.") from exc

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

        side_str = (join_ast.side or "").upper()
        kind_str = (join_ast.kind or "").upper()
        method_str = (join_ast.args.get("method") or "").upper()
        is_natural = method_str == "NATURAL" or bool(join_ast.args.get("natural"))
        using_ast = join_ast.args.get("using")

        # Validate unsupported join types explicitly
        if side_str == "RIGHT" or kind_str == "RIGHT":
            raise NotImplementedError("RIGHT JOIN is not supported by the query engine.")
        if side_str == "FULL" or kind_str == "FULL":
            raise NotImplementedError("FULL JOIN is not supported by the query engine.")
        if "SEMI" in kind_str or "SEMI" in side_str:
            raise NotImplementedError("SEMI JOIN is not supported by the query engine.")
        if "ANTI" in kind_str or "ANTI" in side_str:
            raise NotImplementedError("ANTI JOIN is not supported by the query engine.")

        if side_str == "LEFT" or kind_str == "LEFT":
            kind = "left"
        elif kind_str in ("", "INNER", "CROSS") and side_str in ("", "INNER"):
            kind = "inner"
        else:
            raise NotImplementedError(f"Join type '{side_str} {kind_str}'.strip() is not supported.")

        try:
            joined_schema = resolver.add_table(table_name=joined_name, alias=joined_alias)
        except KeyError as exc:
            raise ValueError(f"Table '{joined_name}' not found in catalog.") from exc

        right_scan = Scan(
            table=joined_name,
            columns=None,
            pushed_predicate=None,
            table_schema=joined_schema,
        )

        right_rel = joined_alias if joined_alias else joined_name

        on_ast = join_ast.args.get("on")
        if on_ast is not None:
            condition = _bind_expr(on_ast, resolver, typechecker)
        elif using_ast is not None:
            conjuncts: list[Expr] = []
            for item in using_ast:
                u_name = item.name if hasattr(item, "name") and item.name else str(item.this)
                right_ref = resolver.resolve_column(name=u_name, table=right_rel)
                left_matches = [
                    rel for rel in resolver.relation_order
                    if rel != right_rel and any(c == u_name for c, _ in resolver.schemas[rel])
                ]
                if len(left_matches) == 0:
                    raise SemanticError(f"Column '{u_name}' in USING clause not found in left tables.")
                if len(left_matches) > 1:
                    raise SemanticError(
                        f"Ambiguous column reference '{u_name}' in USING clause across left tables: {left_matches}."
                    )
                left_ref = resolver.resolve_column(name=u_name, table=left_matches[0])
                conjuncts.append(BinaryOp(op="=", left=left_ref, right=right_ref))

            condition = conjuncts[0]
            for conj in conjuncts[1:]:
                condition = BinaryOp(op="AND", left=condition, right=conj)
        elif is_natural:
            right_cols = [c for c, _ in joined_schema]
            conjuncts = []
            for c_name in right_cols:
                left_matches = [
                    rel for rel in resolver.relation_order
                    if rel != right_rel and any(c == c_name for c, _ in resolver.schemas[rel])
                ]
                if len(left_matches) == 1:
                    left_ref = resolver.resolve_column(name=c_name, table=left_matches[0])
                    right_ref = resolver.resolve_column(name=c_name, table=right_rel)
                    conjuncts.append(BinaryOp(op="=", left=left_ref, right=right_ref))
                elif len(left_matches) > 1:
                    raise SemanticError(f"Ambiguous common column reference '{c_name}' in NATURAL JOIN.")
            if conjuncts:
                condition = conjuncts[0]
                for conj in conjuncts[1:]:
                    condition = BinaryOp(op="AND", left=condition, right=conj)
            else:
                condition = Literal(value=True, dtype=DType.BOOL)
        else:
            condition = Literal(value=True, dtype=DType.BOOL)

        typechecker.check_no_aggregates(condition, context_name="JOIN ON")
        typechecker.check_boolean_condition(condition, resolver, context_name="JOIN ON")

        current_plan = Join(
            left=current_plan,
            right=right_scan,
            condition=condition,
            kind=kind,
        )

    where_clause = ast.args.get("where")
    if where_clause is not None:
        where_pred = _bind_expr(where_clause.this, resolver, typechecker)
        typechecker.check_no_aggregates(where_pred, context_name="WHERE")
        typechecker.check_boolean_condition(where_pred, resolver, context_name="WHERE")
        current_plan = Filter(child=current_plan, predicate=where_pred)

    group_clause = ast.args.get("group")
    having_clause = ast.args.get("having")
    order_clause = ast.args.get("order")
    select_items = ast.expressions

    has_group_by = group_clause is not None
    has_aggs = (
        any(e.find(exp.AggFunc) is not None for e in select_items)
        or (having_clause is not None)
        or (order_clause is not None and order_clause.find(exp.AggFunc) is not None)
    )

    group_keys: list[Expr] = []
    aggs: list[tuple[AggCall, str]] = []
    registered_aggs: dict[str, str] = {}
    agg_call_to_alias: dict[AggCall, str] = {}

    if has_group_by or has_aggs:
        if group_clause is not None:
            for ge in group_clause.expressions:
                gk = _bind_expr(ge, resolver, typechecker)
                typechecker.check_no_aggregates(gk, context_name="GROUP BY")
                group_keys.append(gk)

        def register_agg_call(agg_node: exp.AggFunc, explicit_alias: str | None = None) -> str:
            bound_agg = _bind_agg(agg_node, resolver, typechecker)
            if bound_agg in agg_call_to_alias:
                return agg_call_to_alias[bound_agg]
            alias = explicit_alias if explicit_alias else _get_agg_alias(bound_agg)
            key = repr(agg_node)
            if key not in registered_aggs:
                registered_aggs[key] = alias
                agg_call_to_alias[bound_agg] = alias
                aggs.append((bound_agg, alias))
            else:
                agg_call_to_alias[bound_agg] = registered_aggs[key]
            return agg_call_to_alias[bound_agg]

        for se in select_items:
            if isinstance(se, exp.Alias) and isinstance(se.this, exp.AggFunc):
                register_agg_call(se.this, se.alias)
            else:
                for agg_node in se.find_all(exp.AggFunc):
                    register_agg_call(agg_node)

        if having_clause is not None:
            for agg_node in having_clause.this.find_all(exp.AggFunc):
                register_agg_call(agg_node)

        synth_ord_aggs: list[str] = []
        if order_clause is not None:
            for ordered in order_clause.expressions:
                for agg_node in ordered.find_all(exp.AggFunc):
                    bound_agg = _bind_agg(agg_node, resolver, typechecker)
                    if bound_agg not in agg_call_to_alias:
                        synth_alias = f"__ord_agg_{len(synth_ord_aggs)}"
                        synth_ord_aggs.append(synth_alias)
                        key = repr(agg_node)
                        registered_aggs[key] = synth_alias
                        agg_call_to_alias[bound_agg] = synth_alias
                        aggs.append((bound_agg, synth_alias))

        current_plan = Aggregate(
            child=current_plan,
            group_keys=group_keys,
            aggs=aggs,
        )

        if having_clause is not None:
            having_pred = _bind_expr(having_clause.this, resolver, typechecker)
            typechecker.check_boolean_condition(having_pred, (current_plan, resolver), context_name="HAVING")
            current_plan = Filter(child=current_plan, predicate=having_pred)

    select_aliases = {se.alias for se in select_items if isinstance(se, exp.Alias)}
    user_project_exprs: list[tuple[Expr, str]] = []
    seen_aliases: dict[str, int] = {}

    def deduplicate_alias(cand: str) -> str:
        if cand not in seen_aliases:
            seen_aliases[cand] = 0
            return cand
        seen_aliases[cand] += 1
        return f"{cand}_{seen_aliases[cand]}"

    for se in select_items:
        if isinstance(se, exp.Star):
            for rel_name in resolver.relation_order:
                canonical_table = resolver.get_canonical_table(rel_name)
                for col_name, _ in resolver.schemas[rel_name]:
                    final_alias = deduplicate_alias(col_name)
                    user_project_exprs.append((ColumnRef(table=canonical_table, name=col_name), final_alias))
        else:
            is_explicit = isinstance(se, exp.Alias)
            if is_explicit:
                alias = se.alias
                expr_ast = se.this
            elif isinstance(se, exp.Column):
                alias = se.name
                expr_ast = se
            elif isinstance(se, exp.AggFunc):
                bound_agg = _bind_agg(se, resolver, typechecker)
                alias = _get_agg_alias(bound_agg)
                expr_ast = se
            else:
                alias = se.sql()
                expr_ast = se

            if not is_explicit:
                alias = deduplicate_alias(alias)
            else:
                seen_aliases[alias] = seen_aliases.get(alias, 0)

            if (has_group_by or has_aggs) and isinstance(expr_ast, exp.AggFunc):
                bound_agg = _bind_agg(expr_ast, resolver, typechecker)
                resolved_agg_alias = agg_call_to_alias.get(bound_agg, alias)
                col_ref = ColumnRef(table=None, name=resolved_agg_alias)
                user_project_exprs.append((col_ref, alias))
            else:
                bound_e = _bind_expr(expr_ast, resolver, typechecker)
                user_project_exprs.append((bound_e, alias))

    project_exprs = list(user_project_exprs)
    extra_order_proj_aliases: list[str] = []

    if order_clause is not None:
        if any(t.text.upper() == "NULLS" for t in sqlglot.tokenize(sql)):
            raise SemanticError("NULLS FIRST / NULLS LAST ordering is not supported by the IR.")

        if has_group_by or has_aggs:
            existing_proj_aliases = {alias for _, alias in project_exprs}
            for ordered in order_clause.expressions:
                for agg_node in ordered.find_all(exp.AggFunc):
                    bound_agg = _bind_agg(agg_node, resolver, typechecker)
                    if bound_agg in agg_call_to_alias:
                        agg_alias = agg_call_to_alias[bound_agg]
                        if agg_alias not in existing_proj_aliases:
                            project_exprs.append((ColumnRef(table=None, name=agg_alias), agg_alias))
                            existing_proj_aliases.add(agg_alias)
                            extra_order_proj_aliases.append(agg_alias)

    proj_aliases = {alias: expr for expr, alias in project_exprs}
    agg_aliases = {alias for _, alias in aggs}

    if has_group_by or has_aggs:
        typechecker.check_group_by_projections(
            project_exprs=project_exprs,
            group_keys=group_keys,
            agg_aliases=agg_aliases,
            proj_aliases=proj_aliases,
        )

    for p_expr, _ in project_exprs:
        typechecker.infer_type(p_expr, (current_plan, resolver))

    current_plan = Project(child=current_plan, exprs=project_exprs)

    # Lower SELECT DISTINCT to Aggregate with group_keys over projected columns
    if is_distinct:
        distinct_group_keys = [ColumnRef(table=None, name=alias) for _, alias in project_exprs]
        current_plan = Aggregate(
            child=current_plan,
            group_keys=distinct_group_keys,
            aggs=[],
        )

    if order_clause is not None:
        sort_keys: list[tuple[Expr, bool]] = []
        child_schema_dict = dict(current_plan.schema())
        existing_plan_aliases = set(child_schema_dict.keys())

        for ordered in order_clause.expressions:
            is_desc = bool(ordered.args.get("desc", False))
            order_ast = ordered.this

            # Positional ORDER BY (e.g. ORDER BY 2)
            if isinstance(order_ast, exp.Literal) and order_ast.is_number and not order_ast.is_string:
                pos = int(order_ast.this)
                if pos < 1 or pos > len(user_project_exprs):
                    raise SemanticError(
                        f"ORDER BY position {pos} is out of range (must be between 1 and {len(user_project_exprs)})."
                    )
                target_expr, target_alias = user_project_exprs[pos - 1]
                if target_alias in select_aliases:
                    sort_expr = ColumnRef(table=None, name=target_alias)
                else:
                    sort_expr = target_expr

            elif isinstance(order_ast, exp.Column) and not order_ast.table:
                if order_ast.name in select_aliases:
                    sort_expr = ColumnRef(table=None, name=order_ast.name)
                elif order_ast.name in proj_aliases:
                    try:
                        sort_expr = resolver.resolve_column(name=order_ast.name)
                    except (SemanticError, ValueError):
                        sort_expr = ColumnRef(table=None, name=order_ast.name)
                else:
                    sort_expr = _bind_expr(order_ast, resolver, typechecker)
            else:
                sort_expr = _bind_expr(order_ast, resolver, typechecker)

            def _map_agg_to_colref(e: Expr) -> Expr:
                if isinstance(e, AggCall) and e in agg_call_to_alias:
                    return ColumnRef(table=None, name=agg_call_to_alias[e])
                return e

            sort_expr = transform_expr_post_order(sort_expr, _map_agg_to_colref)

            # Ensure any column referenced by Sort is present in current_plan
            # If not present in current_plan (ORDER BY column not in SELECT list), pass it through
            def _rewrite_missing_colref(e: Expr) -> Expr:
                nonlocal current_plan
                if not (has_group_by or has_aggs) and isinstance(e, ColumnRef):
                    # Check if directly in current_plan schema
                    if e.name in existing_plan_aliases:
                        return e
                    if e.table:
                        qname = f"{e.table}.{e.name}"
                        if qname in existing_plan_aliases:
                            return e
                    # If current_plan is Project, check if already in exprs, else append as hidden projection
                    if isinstance(current_plan, Project):
                        for p_expr, p_alias in current_plan.exprs:
                            if p_expr == e:
                                if p_alias == e.name:
                                    return e
                                return ColumnRef(table=None, name=p_alias)
                        hidden_alias = f"__ord_hidden_{len(extra_order_proj_aliases)}"
                        new_exprs = list(current_plan.exprs) + [(e, hidden_alias)]
                        current_plan = Project(child=current_plan.child, exprs=new_exprs)
                        existing_plan_aliases.add(hidden_alias)
                        extra_order_proj_aliases.append(hidden_alias)
                        return ColumnRef(table=None, name=hidden_alias)
                return e

            sort_expr = transform_expr_post_order(sort_expr, _rewrite_missing_colref)
            sort_keys.append((sort_expr, is_desc))

        if has_group_by or has_aggs:
            typechecker.check_group_by_order_by(
                sort_keys=sort_keys,
                group_keys=group_keys,
                agg_aliases=agg_aliases,
                proj_aliases=proj_aliases,
            )

        for s_expr, _ in sort_keys:
            typechecker.infer_type(s_expr, (current_plan, resolver))

        current_plan = Sort(child=current_plan, keys=sort_keys)

        if extra_order_proj_aliases:
            final_project_exprs: list[tuple[Expr, str]] = [
                (ColumnRef(table=None, name=alias), alias)
                for _, alias in user_project_exprs
            ]
            for f_expr, _ in final_project_exprs:
                typechecker.infer_type(f_expr, (current_plan, resolver))
            current_plan = Project(child=current_plan, exprs=final_project_exprs)

    limit_clause = ast.args.get("limit")
    if limit_clause is not None:
        if limit_clause.args.get("offset") is not None:
            raise SemanticError("OFFSET is not supported by the query compiler.")
        limit_expr = limit_clause.expression
        if not isinstance(limit_expr, exp.Literal) or not limit_expr.is_number:
            raise ValueError(f"Expected integer numeric LIMIT, got: {limit_expr}")
        n = int(limit_expr.this)
        current_plan = Limit(child=current_plan, n=n)

    return current_plan
