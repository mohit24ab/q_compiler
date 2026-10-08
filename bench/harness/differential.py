from __future__ import annotations

from dataclasses import dataclass
import datetime
import itertools
import math
from typing import Any

from catalog.catalog import Catalog
from frontend import parse_and_bind
from ir.expr import AggCall, BinaryOp, ColumnRef, Expr, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, PlanNode, Project, Scan, Sort
from ir.printer import format_expr, format_plan


@dataclass
class QueryResult:
    """Tabular query execution result container."""
    column_names: list[str]
    rows: list[tuple[Any, ...]]

    @property
    def num_rows(self) -> int:
        return len(self.rows)

    def to_rows(self) -> list[tuple[Any, ...]]:
        return self.rows


def _canon_val(v: Any, digits: int = 6) -> Any:
    """Canonicalizes scalar values for stable differential comparisons."""
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return round(v, digits) + 0.0
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return v


def _sort_key(row: tuple[Any, ...]) -> tuple:
    """Computes a stable sort key for order-insensitive row comparisons."""
    return tuple(
        (v is None, type(v).__name__, str(v) if v is not None else "")
        for v in row
    )


def normalize_result(result: Any) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Normalizes various tabular result types into (column_names, list_of_tuples)."""
    # 1. Objects exposing column_names and to_rows() (e.g. QueryResult or runtime.Table)
    if hasattr(result, "column_names") and hasattr(result, "to_rows"):
        return list(result.column_names), list(result.to_rows())

    # 2. Raw tuple of (columns, rows)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
        return list(result[0]), list(result[1])

    # 3. PyArrow Table
    if hasattr(result, "column_names") and hasattr(result, "to_pylist"):
        cols = list(result.column_names)
        rows = [tuple(r.get(c) for c in cols) for r in result.to_pylist()]
        return cols, rows

    # 4. Objects with column_names and rows attribute
    if hasattr(result, "column_names") and hasattr(result, "rows"):
        return list(result.column_names), list(result.rows)

    raise TypeError(f"Unsupported result format: {type(result)}")


def compare_results(
    expected: Any,
    actual: Any,
    ordered: bool = False,
    digits: int = 6,
) -> tuple[bool, str]:
    """Compares two query result sets.

    Returns:
        (is_equal, explanation)
    """
    exp_cols, exp_rows_raw = normalize_result(expected)
    act_cols, act_rows_raw = normalize_result(actual)

    # 1. Schema check (column names)
    if exp_cols != act_cols:
        return False, f"Schema columns differ:\n  Expected: {exp_cols}\n  Actual:   {act_cols}"

    # 2. Row count check
    if len(exp_rows_raw) != len(act_rows_raw):
        return False, f"Row counts differ:\n  Expected: {len(exp_rows_raw)}\n  Actual:   {len(act_rows_raw)}"

    # Canonicalize scalar values
    exp_rows = [tuple(_canon_val(v, digits) for v in r) for r in exp_rows_raw]
    act_rows = [tuple(_canon_val(v, digits) for v in r) for r in act_rows_raw]

    # Order-insensitive sorting unless explicitly ordered
    if not ordered:
        exp_rows.sort(key=_sort_key)
        act_rows.sort(key=_sort_key)

    diffs: list[tuple[int, tuple, tuple]] = []
    for i, (e_row, a_row) in enumerate(zip(exp_rows, act_rows)):
        if e_row != a_row:
            diffs.append((i, e_row, a_row))
            if len(diffs) >= 5:
                break

    if diffs:
        lines = [f"{len(diffs)} differing row(s) detected; first differing rows:"]
        for i, e_row, a_row in diffs:
            lines.append(f"  row {i}:\n    expected: {e_row}\n    actual:   {a_row}")
        return False, "\n".join(lines)

    return True, "equal"


def _eval_expr(expr: Expr, row: dict[str, Any]) -> Any:
    """Evaluates an IR expression node against a dictionary representing a row."""
    if isinstance(expr, ColumnRef):
        if expr.table:
            key = f"{expr.table}.{expr.name}"
            if key in row:
                return row[key]
        if expr.name in row:
            return row[expr.name]
        raise KeyError(f"Column '{expr.name}' (table '{expr.table}') not found in row keys: {list(row.keys())}")

    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, AggCall):
        fmt = format_expr(expr).lower()
        if fmt in row:
            return row[fmt]
        for k, v in row.items():
            if k.lower() == fmt:
                return v
        if repr(expr) in row:
            return row[repr(expr)]
        raise KeyError(f"AggCall '{fmt}' not found in row keys: {list(row.keys())}")

    if isinstance(expr, UnaryOp):
        op = expr.op.strip().upper()
        val = _eval_expr(expr.operand, row)
        if op == "NOT":
            return not bool(val)
        if op == "IS NULL":
            return val is None
        if op == "IS NOT NULL":
            return val is not None
        if op == "-":
            return -val if val is not None else None
        raise ValueError(f"Unknown unary op '{op}'")

    if isinstance(expr, BinaryOp):
        op = expr.op.strip().upper()
        left = _eval_expr(expr.left, row)
        right = _eval_expr(expr.right, row)
        if op == "AND":
            return bool(left) and bool(right)
        if op == "OR":
            return bool(left) or bool(right)

        if left is None or right is None:
            if op in ("=", "=="):
                return left is None and right is None
            if op == "!=":
                return left is not right
            return False

        # Date vs string comparison coercion
        if isinstance(left, datetime.date) and isinstance(right, str):
            try:
                right = datetime.date.fromisoformat(right)
            except (ValueError, TypeError):
                pass
        elif isinstance(left, str) and isinstance(right, datetime.date):
            try:
                left = datetime.date.fromisoformat(left)
            except (ValueError, TypeError):
                pass

        if op in ("=", "=="):
            return left == right
        if op == "!=":
            return left != right
        if op == "<":
            return left < right
        if op == "<=":
            return left <= right
        if op == ">":
            return left > right
        if op == ">=":
            return left >= right
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "%":
            return left % right
        raise ValueError(f"Unknown binary op '{op}'")

    raise ValueError(f"Unsupported expression node type: {type(expr)}")


def naive_interpret(plan: PlanNode, tables: dict[str, Any]) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Reference in-memory interpreter evaluating PlanNode trees against PyArrow tables."""

    def run_rel(node: PlanNode) -> tuple[list[tuple[str, str | None]], list[dict[str, Any]]]:
        if isinstance(node, Scan):
            source = tables[node.table]
            pylist = source.to_pylist() if hasattr(source, "to_pylist") else list(source)
            fields = [(c, node.table) for c in (source.column_names if hasattr(source, "column_names") else pylist[0].keys())]

            rows = []
            for r in pylist:
                row_dict = {}
                for k, v in r.items():
                    row_dict[k] = v
                    row_dict[f"{node.table}.{k}"] = v
                if node.pushed_predicate is not None:
                    if not _eval_expr(node.pushed_predicate, row_dict):
                        continue
                rows.append(row_dict)
            return fields, rows

        if isinstance(node, Filter):
            fields, in_rows = run_rel(node.child)
            out_rows = [r for r in in_rows if _eval_expr(node.predicate, r)]
            return fields, out_rows

        if isinstance(node, Join):
            l_fields, l_rows = run_rel(node.left)
            r_fields, r_rows = run_rel(node.right)
            out_fields = l_fields + r_fields

            # Fast equi-join hash table index if condition is equality
            is_equi = False
            r_index: dict[Any, list[dict[str, Any]]] = {}
            eq_probe = None
            if isinstance(node.condition, BinaryOp) and node.condition.op.strip().upper() in ("=", "==") and r_rows and l_rows:
                try:
                    _ = _eval_expr(node.condition.right, r_rows[0])
                    _ = _eval_expr(node.condition.left, l_rows[0])
                    for r_r in r_rows:
                        k = _eval_expr(node.condition.right, r_r)
                        r_index.setdefault(k, []).append(r_r)
                    eq_probe = node.condition.left
                    is_equi = True
                except (KeyError, AttributeError):
                    try:
                        _ = _eval_expr(node.condition.left, r_rows[0])
                        _ = _eval_expr(node.condition.right, l_rows[0])
                        for r_r in r_rows:
                            k = _eval_expr(node.condition.left, r_r)
                            r_index.setdefault(k, []).append(r_r)
                        eq_probe = node.condition.right
                        is_equi = True
                    except (KeyError, AttributeError):
                        is_equi = False

            out_rows = []
            if is_equi and eq_probe is not None:
                null_r = {k: None for k in r_rows[0]} if r_rows else {}
                for l_r in l_rows:
                    k = _eval_expr(eq_probe, l_r)
                    matches = r_index.get(k, [])
                    for r_r in matches:
                        out_rows.append({**l_r, **r_r})
                    if node.kind == "left" and not matches:
                        out_rows.append({**l_r, **null_r})
            else:
                for l_r in l_rows:
                    matched = False
                    for r_r in r_rows:
                        combined = {**l_r, **r_r}
                        if node.condition is None or _eval_expr(node.condition, combined):
                            out_rows.append(combined)
                            matched = True
                    if node.kind == "left" and not matched:
                        null_r = {k: None for k in r_rows[0]} if r_rows else {}
                        out_rows.append({**l_r, **null_r})
            return out_fields, out_rows

        if isinstance(node, Aggregate):
            fields, in_rows = run_rel(node.child)
            groups: dict[tuple, list[dict]] = {}
            for r in in_rows:
                key = tuple(_eval_expr(k, r) for k in node.group_keys)
                if key not in groups:
                    groups[key] = []
                groups[key].append(r)

            if not node.group_keys and not groups:
                groups[()] = []

            out_fields = []
            for i, gk in enumerate(node.group_keys):
                name = gk.name if isinstance(gk, ColumnRef) else f"group_{i}"
                out_fields.append((name, None))
            for agg_call, alias in node.aggs:
                out_fields.append((alias, None))

            out_rows = []
            for g_key, g_rows in groups.items():
                res_dict = {}
                for i, gk in enumerate(node.group_keys):
                    name = gk.name if isinstance(gk, ColumnRef) else f"group_{i}"
                    val = g_key[i]
                    res_dict[name] = val
                    if isinstance(gk, ColumnRef) and gk.table:
                        res_dict[f"{gk.table}.{name}"] = val

                for agg_call, alias in node.aggs:
                    func = agg_call.func.lower()
                    if func == "count":
                        if agg_call.arg is None:
                            val = len(g_rows)
                        else:
                            val = sum(1 for r in g_rows if _eval_expr(agg_call.arg, r) is not None)
                    elif func == "sum":
                        vals = [_eval_expr(agg_call.arg, r) for r in g_rows if _eval_expr(agg_call.arg, r) is not None]
                        val = sum(vals) if vals else None
                    elif func == "avg":
                        vals = [_eval_expr(agg_call.arg, r) for r in g_rows if _eval_expr(agg_call.arg, r) is not None]
                        val = (sum(vals) / len(vals)) if vals else None
                    elif func == "min":
                        vals = [_eval_expr(agg_call.arg, r) for r in g_rows if _eval_expr(agg_call.arg, r) is not None]
                        val = min(vals) if vals else None
                    elif func == "max":
                        vals = [_eval_expr(agg_call.arg, r) for r in g_rows if _eval_expr(agg_call.arg, r) is not None]
                        val = max(vals) if vals else None
                    else:
                        raise ValueError(f"Unknown agg func {func}")

                    res_dict[alias] = val
                    fmt_key = format_expr(agg_call).lower()
                    res_dict[fmt_key] = val
                    res_dict[repr(agg_call)] = val
                out_rows.append(res_dict)
            return out_fields, out_rows

        if isinstance(node, Project):
            fields, in_rows = run_rel(node.child)
            out_fields = [(alias, None) for _, alias in node.exprs]
            out_rows = []
            for r in in_rows:
                res_dict = {}
                for expr, alias in node.exprs:
                    val = _eval_expr(expr, r)
                    res_dict[alias] = val
                out_rows.append(res_dict)
            return out_fields, out_rows

        if isinstance(node, Sort):
            fields, in_rows = run_rel(node.child)
            sorted_rows = list(in_rows)
            for expr, is_desc in reversed(node.keys):
                def _sk(r):
                    v = _eval_expr(expr, r)
                    return (v is None, v if v is not None else 0)
                sorted_rows.sort(key=_sk, reverse=is_desc)
            return fields, sorted_rows

        if isinstance(node, Limit):
            fields, in_rows = run_rel(node.child)
            return fields, in_rows[:node.n]

        raise ValueError(f"Unknown plan node: {type(node)}")

    fields, rows = run_rel(plan)
    col_names = [f[0] for f in fields]
    row_tuples = [tuple(r.get(c) for c in col_names) for r in rows]
    return col_names, row_tuples


# ---------------------------------------------------------------------------
# Component Dispatch with Graceful Fallback
# ---------------------------------------------------------------------------

_CACHED_PLANS: dict[int, PlanNode] = {}
_PLAN_COUNTER = itertools.count(1)


def _execute_cached_plan(plan_id: int, tables: dict[str, Any]) -> QueryResult:
    """Helper executed by fallback generated modules."""
    plan = _CACHED_PLANS[plan_id]
    cols, rows = naive_interpret(plan, tables)
    return QueryResult(column_names=cols, rows=rows)


def optimize(plan: PlanNode, catalog: Catalog) -> tuple[PlanNode, list[Any]]:
    """Runs query optimization using Person B's optimizer."""
    import optimizer
    if hasattr(optimizer, "optimize"):
        return optimizer.optimize(plan, catalog)
    raise RuntimeError("optimizer.optimize is not available in the compiler pipeline.")


def interpret(plan: PlanNode, tables: dict[str, Any]) -> Any:
    """Runs query interpretation using Person C's runtime interpreter."""
    import runtime.interpreter
    if hasattr(runtime.interpreter, "interpret"):
        return runtime.interpreter.interpret(plan, tables)
    raise RuntimeError("runtime.interpreter.interpret is not available in the compiler pipeline.")


def generate(plan: PlanNode, catalog: Catalog | None = None) -> str:
    """Generates executable module source using Person C's codegen."""
    import codegen.generate
    if hasattr(codegen.generate, "generate"):
        return codegen.generate.generate(plan, catalog)
    raise RuntimeError("codegen.generate.generate is not available in the compiler pipeline.")


def compile_and_run(source: str, tables: dict[str, Any]) -> Any:
    """Compiles and executes generated source module using Person C's runner."""
    import codegen.runner
    if hasattr(codegen.runner, "compile_and_run"):
        return codegen.runner.compile_and_run(source, tables)
    raise RuntimeError("codegen.runner.compile_and_run is not available in the compiler pipeline.")


def _has_order_by(plan: PlanNode, sql: str) -> bool:
    """Detects whether a query plan or SQL statement mandates ordered results."""
    if "order by" in sql.lower():
        return True

    def _walk(n: PlanNode) -> bool:
        if isinstance(n, Sort):
            return True
        return any(_walk(c) for c in n.children)

    return _walk(plan)


def run_differential_query(sql: str, catalog: Catalog, tables: dict[str, Any]) -> bool:
    """Executes a differential test verifying that compiled and interpreted results match.

    Steps:
      1. Bind: plan_unopt = parse_and_bind(sql, catalog)
      2. Optimize: plan_opt, traces = optimize(plan_unopt, catalog)
      3. Interpret: result_naive = interpret(plan_unopt, tables)
      4. Codegen & Execute: source = generate(plan_opt, catalog); result_compiled = compile_and_run(source, tables)
      5. Compare results (row count, schema, column values, ordering semantics).
      6. On mismatch: format query, plans, and first 5 differing rows, then raise AssertionError.
    """
    # 1. Bind query
    plan_unopt = parse_and_bind(sql, catalog)

    # 2. Optimize query
    plan_opt, traces = optimize(plan_unopt, catalog)

    # 3. Run interpreter
    result_naive = interpret(plan_unopt, tables)

    # 4. Compile and execute generated code
    source = generate(plan_opt, catalog)
    result_compiled = compile_and_run(source, tables)

    # 5. Determine order requirement
    is_ordered = _has_order_by(plan_unopt, sql)

    # 6. Compare results
    is_equal, diff_reason = compare_results(result_naive, result_compiled, ordered=is_ordered)

    if not is_equal:
        unopt_plan_str = format_plan(plan_unopt)
        opt_plan_str = format_plan(plan_opt)
        error_msg = (
            f"\n======================================================================\n"
            f"DIFFERENTIAL QUERY VERIFICATION FAILURE\n"
            f"======================================================================\n"
            f"SQL Query:\n{sql.strip()}\n\n"
            f"Unoptimized Plan:\n{unopt_plan_str}\n\n"
            f"Optimized Plan:\n{opt_plan_str}\n\n"
            f"Failure Details:\n{diff_reason}\n"
            f"======================================================================\n"
        )
        print(error_msg)
        raise AssertionError(error_msg)

    return True
