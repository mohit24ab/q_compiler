from __future__ import annotations

from dataclasses import dataclass
import datetime
import math
from typing import Any

from catalog.catalog import Catalog
from frontend import parse_and_bind
from ir.nodes import PlanNode, Sort
from ir.printer import format_plan


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


def _val_close(v1: Any, v2: Any, rel_tol: float = 1e-5, abs_tol: float = 1e-5) -> bool:
    """Checks whether two scalar values are equal, using math.isclose for floats."""
    if isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
        if isinstance(v1, float) or isinstance(v2, float):
            try:
                f1, f2 = float(v1), float(v2)
                if math.isnan(f1) and math.isnan(f2):
                    return True
                if math.isnan(f1) or math.isnan(f2):
                    return False
                return math.isclose(f1, f2, rel_tol=rel_tol, abs_tol=abs_tol)
            except (TypeError, ValueError):
                pass
    return v1 == v2


def _canon_val(v: Any, digits: int = 6) -> Any:
    """Canonicalizes scalar values for stable differential comparisons."""
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        return v
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return v


def _sort_key(row: tuple[Any, ...]) -> tuple:
    """Computes a stable sort key for order-insensitive row comparisons."""
    return tuple(
        (
            v is None,
            type(v).__name__,
            round(float(v), 4)
            if isinstance(v, (float, int)) and not (isinstance(v, float) and math.isnan(v))
            else str(v)
            if v is not None
            else "",
        )
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
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-5,
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
        if len(e_row) != len(a_row) or not all(
            _val_close(e, a, rel_tol=rel_tol, abs_tol=abs_tol) for e, a in zip(e_row, a_row)
        ):
            diffs.append((i, e_row, a_row))
            if len(diffs) >= 5:
                break

    if diffs and not ordered:
        # Multiset check fallback for float sorting ties
        matched_actual_indices: set[int] = set()
        all_matched = True
        for e_row in exp_rows:
            found = False
            for j, a_row in enumerate(act_rows):
                if j not in matched_actual_indices:
                    if len(e_row) == len(a_row) and all(
                        _val_close(e, a, rel_tol=rel_tol, abs_tol=abs_tol) for e, a in zip(e_row, a_row)
                    ):
                        matched_actual_indices.add(j)
                        found = True
                        break
            if not found:
                all_matched = False
                break
        if all_matched and len(matched_actual_indices) == len(act_rows):
            return True, "equal"

    if diffs:
        lines = [f"{len(diffs)} differing row(s) detected; first differing rows:"]
        for i, e_row, a_row in diffs:
            lines.append(f"  row {i}:\n    expected: {e_row}\n    actual:   {a_row}")
        return False, "\n".join(lines)

    return True, "equal"


# ---------------------------------------------------------------------------
# Component Dispatch
# ---------------------------------------------------------------------------
import optimizer
import optimizer.manager
if not hasattr(optimizer.manager, "optimize"):
    optimizer.manager.optimize = getattr(optimizer, "optimize", None)
from optimizer.manager import optimize
from runtime.interpreter import interpret
from codegen import generate
from codegen.runner import compile_and_run



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
