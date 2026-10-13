"""Result comparison used by the differential checks (Contract §7).

Two results are equal when they hold the same rows: as a multiset unless `ordered`
(the query has ORDER BY), in which case the row order must match too.
Floats compare with a relative tolerance (math.isclose), since generated code may sum in
a different order than the interpreter. A fixed number of decimal places would not do:
a 1M-row sum of about 1e9 carries order-dependent noise near the 6th decimal, and two
values either side of a rounding boundary would differ however close they are.
"""
from __future__ import annotations

import math

from runtime.table import Table

REL_TOL = 1e-9   # far above the ~1e-13 summation-order noise of a 1M-row float sum
ABS_TOL = 1e-9   # for results that should be 0 but carry cancellation noise


def _values_close(e, a, rel_tol: float, abs_tol: float) -> bool:
    if isinstance(e, float) or isinstance(a, float):
        if not (_is_number(e) and _is_number(a)):
            return False
        if math.isnan(e) or math.isnan(a):
            return math.isnan(e) and math.isnan(a)
        return math.isclose(e, a, rel_tol=rel_tol, abs_tol=abs_tol)
    return e == a


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _rows_close(e, a, rel_tol: float, abs_tol: float) -> bool:
    return len(e) == len(a) and all(_values_close(x, y, rel_tol, abs_tol) for x, y in zip(e, a))


def _sort_key(row):
    """Every non-float value first (in column order), then the floats: rows sort into the
    same order on both sides even when their floats differ by noise, as long as the other
    columns tell them apart. NULLs and mixed types must not break sorting."""
    exact = tuple((v is None, type(v).__name__, 0 if v is None else v)
                  for v in row if not isinstance(v, float))
    floats = tuple((math.isnan(v), 0.0 if math.isnan(v) else v)
                   for v in row if isinstance(v, float))
    return exact, floats


def compare_tables(expected: Table, actual: Table, ordered: bool = False,
                   rel_tol: float = REL_TOL, abs_tol: float = ABS_TOL) -> tuple[bool, str]:
    """Return (equal, explanation). Column names are compared, qualifiers are not."""
    if expected.column_names != actual.column_names:
        return False, f"columns differ: {expected.column_names} vs {actual.column_names}"
    if expected.num_rows != actual.num_rows:
        return False, f"row counts differ: {expected.num_rows} vs {actual.num_rows}"

    exp, act = expected.to_rows(), actual.to_rows()
    if not ordered:
        exp.sort(key=_sort_key)
        act.sort(key=_sort_key)

    diffs = [(i, e, a) for i, (e, a) in enumerate(zip(exp, act))
             if not _rows_close(e, a, rel_tol, abs_tol)]
    if not diffs:
        return True, "equal"
    lines = [f"{len(diffs)} differing rows; first five:"]
    lines += [f"  row {i}: expected {e} got {a}" for i, e, a in diffs[:5]]
    return False, "\n".join(lines)
