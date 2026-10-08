"""A deliberately naive plan evaluator: the optimizer's correctness oracle.

Use it until Person C's ``runtime.interpret`` lands, then swap that in, in
``assert_equivalent``. It is written for obviousness, not speed: nested-loop
joins, Python lists of tuples.

Each column is tracked as ``(qualifier, name)``. A Scan's columns are
qualified with their table; Project and aggregate outputs are unqualified.
Name resolution is STRICT:

* ``ColumnRef(None, n)`` matches the one column named ``n``.
* ``ColumnRef(t, n)`` matches only the column named ``n`` qualified by ``t``.

Zero matches or several matches raise ``LookupError``. So a rewrite that
drops a needed column, or that breaks a qualified reference, fails loudly
here instead of returning something plausible.

SQL NULL semantics: NULL is ``None``. Comparisons and arithmetic involving
NULL yield NULL. AND, OR and NOT use three-valued logic. Filter keeps only
rows whose predicate is TRUE.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort

Column = tuple[str | None, str]


@dataclass
class Result:
    columns: list[Column]
    rows: list[tuple]
    # Aggregate results by call (repr of the AggCall -> column index), so a
    # HAVING or ORDER BY above can name them by repeating the call.
    computed: dict[str, int] = field(default_factory=dict)


def evaluate(plan, tables: dict[str, tuple[list[str], list[tuple]]]) -> Result:
    """Evaluate ``plan`` over ``tables``, which maps each table name to ``(column_names, rows)``."""
    if isinstance(plan, Scan):
        # Same order as Person C's Scan: filter the full table, then narrow to
        # `columns`. So pushed_predicate may read columns not in `columns`.
        names, rows = tables[plan.table]
        if plan.pushed_predicate is not None:
            full = [(plan.table, n) for n in names]
            rows = [r for r in rows if _eval(plan.pushed_predicate, full, r) is True]
        cols = list(plan.columns) if plan.columns is not None else list(names)
        idx = [names.index(c) for c in cols]
        return Result([(plan.table, c) for c in cols], [tuple(r[i] for i in idx) for r in rows])
    if isinstance(plan, Filter):
        child = evaluate(plan.child, tables)
        keep = [r for r in child.rows if _eval(plan.predicate, child.columns, r, child.computed) is True]
        return Result(child.columns, keep, child.computed)
    if isinstance(plan, Project):
        child = evaluate(plan.child, tables)
        return Result(
            [(None, alias) for _, alias in plan.exprs],
            [tuple(_eval(e, child.columns, r, child.computed) for e, _ in plan.exprs) for r in child.rows],
        )
    if isinstance(plan, Join):
        return _join(plan, evaluate(plan.left, tables), evaluate(plan.right, tables))
    if isinstance(plan, Aggregate):
        return _aggregate(plan, evaluate(plan.child, tables))
    if isinstance(plan, Sort):
        child = evaluate(plan.child, tables)
        rows = list(child.rows)
        for expr, desc in reversed(plan.keys):  # stable sort: least significant key first
            rows.sort(key=lambda r: _null_last_key(_eval(expr, child.columns, r, child.computed)), reverse=desc)
        return Result(child.columns, rows, child.computed)
    if isinstance(plan, Limit):
        child = evaluate(plan.child, tables)
        return Result(child.columns, child.rows[: plan.n], child.computed)
    raise TypeError(f"reference evaluator cannot run {type(plan).__name__}")


def _join(plan: Join, left: Result, right: Result) -> Result:
    columns = left.columns + right.columns
    null_right = (None,) * len(right.columns)
    rows = []
    for lrow in left.rows:
        matched = False
        for rrow in right.rows:
            if _eval(plan.condition, columns, lrow + rrow) is True:
                rows.append(lrow + rrow)
                matched = True
        if plan.kind == "left" and not matched:
            rows.append(lrow + null_right)
    return Result(columns, rows)


def _aggregate(plan: Aggregate, child: Result) -> Result:
    groups: dict[tuple, list[tuple]] = {}
    for row in child.rows:
        key = tuple(_eval(k, child.columns, row) for k in plan.group_keys)
        groups.setdefault(key, []).append(row)
    if not plan.group_keys and not groups:
        groups[()] = []  # a global aggregate returns one row even over no input
    key_cols = [
        (k.table, k.name) if isinstance(k, ColumnRef) else (None, f"key{i}")
        for i, k in enumerate(plan.group_keys)
    ]
    rows = [
        key + tuple(_agg(call, child.columns, members) for call, _ in plan.aggs)
        for key, members in groups.items()
    ]
    computed = {repr(call): len(key_cols) + i for i, (call, _) in enumerate(plan.aggs)}
    return Result(key_cols + [(None, alias) for _, alias in plan.aggs], rows, computed)


def _agg(call: AggCall, columns, rows) -> object:
    if call.arg is None:
        assert call.func == "count", call
        return len(rows)
    values = [v for v in (_eval(call.arg, columns, r) for r in rows) if v is not None]
    if call.func == "count":
        return len(values)
    if not values:
        return None
    if call.func == "sum":
        return sum(values)
    if call.func == "avg":
        return sum(values) / len(values)
    if call.func == "min":
        return min(values)
    if call.func == "max":
        return max(values)
    raise ValueError(f"unknown aggregate {call.func!r}")


def resolve(columns: list[Column], ref: ColumnRef) -> int:
    hits = [
        i for i, (q, n) in enumerate(columns)
        if n == ref.name and (ref.table is None or q == ref.table)
    ]
    if len(hits) != 1:
        what = "ambiguous" if hits else "unresolved"
        raise LookupError(f"{what} column reference {ref} against {columns}")
    return hits[0]


_COMPARE = {
    "=": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<>": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b,
}


def _eval(expr, columns: list[Column], row: tuple, computed: dict[str, int] | None = None):
    if isinstance(expr, ColumnRef):
        return row[resolve(columns, expr)]
    if isinstance(expr, AggCall):  # names a result of the Aggregate below
        if not computed or repr(expr) not in computed:
            raise LookupError(f"aggregate {expr!r} is not produced by an Aggregate below")
        return row[computed[repr(expr)]]
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, UnaryOp):
        value = _eval(expr.operand, columns, row, computed)
        op = expr.op.upper()
        if op == "NOT":
            return None if value is None else not value
        if op == "-":
            return None if value is None else -value
        if op == "IS NULL":
            return value is None
        if op == "IS NOT NULL":
            return value is not None
        raise ValueError(f"unknown unary operator {expr.op!r}")
    if isinstance(expr, BinaryOp):
        op = expr.op.upper()
        left = _eval(expr.left, columns, row, computed)
        right = _eval(expr.right, columns, row, computed)
        if op == "AND":
            if left is False or right is False:
                return False
            return None if left is None or right is None else True
        if op == "OR":
            if left is True or right is True:
                return True
            return None if left is None or right is None else False
        if left is None or right is None:
            return None
        return _COMPARE[op](left, right)
    raise TypeError(f"reference evaluator cannot evaluate {expr!r}")


def _null_last_key(value):
    return (value is None, value if value is not None else 0)


# --------------------------------------------------------------------------
# Comparing results
# --------------------------------------------------------------------------


def _is_ordered(plan) -> bool:
    """Return True if row order is part of the answer: a Sort below only Limit/Project/Filter nodes."""
    while isinstance(plan, (Limit, Project, Filter)):
        plan = plan.child
    return isinstance(plan, Sort)


def _normalise(rows: list[tuple]) -> list[tuple]:
    # Summation order may legitimately change under optimization.
    return [tuple(round(v, 9) if isinstance(v, float) else v for v in r) for r in rows]


def assert_equivalent(original, optimized, tables) -> Result:
    """Contract §7 check: same output columns and same rows (order-insensitive unless the query sorts)."""
    expected = evaluate(original, tables)
    actual = evaluate(optimized, tables)
    assert actual.columns == expected.columns, (
        f"output columns changed: {expected.columns} -> {actual.columns}"
    )
    exp_rows, act_rows = _normalise(expected.rows), _normalise(actual.rows)
    if _is_ordered(original):
        assert act_rows == exp_rows, "ordered rows differ"
    else:
        assert Counter(act_rows) == Counter(exp_rows), "rows differ (as multisets)"
    return actual
