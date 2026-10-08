"""Result comparison used by the differential checks (Contract §7).

Two results are equal when they hold the same rows: as a multiset unless `ordered`
(the query has ORDER BY), in which case the row order must match too.
Floats compare with a relative tolerance, since generated code may sum in a different
order than the interpreter.
"""
from __future__ import annotations

import math

from runtime.table import Table


def _canon(value, digits: int):
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return round(value, digits) + 0.0  # +0.0 folds -0.0 into 0.0
    return value


def _sort_key(row):
    # NULLs and mixed types must not break sorting; compare by (is_null, type, value).
    return tuple((v is None, type(v).__name__, v if v is not None else 0) for v in row)


def compare_tables(expected: Table, actual: Table, ordered: bool = False,
                   digits: int = 6) -> tuple[bool, str]:
    """Return (equal, explanation). Column names are compared, qualifiers are not."""
    if expected.column_names != actual.column_names:
        return False, f"columns differ: {expected.column_names} vs {actual.column_names}"
    if expected.num_rows != actual.num_rows:
        return False, f"row counts differ: {expected.num_rows} vs {actual.num_rows}"

    exp = [tuple(_canon(v, digits) for v in r) for r in expected.to_rows()]
    act = [tuple(_canon(v, digits) for v in r) for r in actual.to_rows()]
    if not ordered:
        exp.sort(key=_sort_key)
        act.sort(key=_sort_key)

    diffs = [(i, e, a) for i, (e, a) in enumerate(zip(exp, act)) if e != a]
    if not diffs:
        return True, "equal"
    lines = [f"{len(diffs)} differing rows; first five:"]
    lines += [f"  row {i}: expected {e} got {a}" for i, e, a in diffs[:5]]
    return False, "\n".join(lines)
