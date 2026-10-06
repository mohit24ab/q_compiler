"""Cardinality estimation (optimizer/stats.py).

The first part checks each rule against hand-computed values on a table
whose statistics are known exactly. The second part checks invariants on
every plan the suites have, original and optimized. The last part checks
accuracy: where the data meets the estimator's assumptions, the estimate
must be close to the actual row count.
"""

import datetime
import math
import warnings

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
import opt_stats_workload as W
import optimizer
from ir.dtype import DType
from ir.expr import Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Scan
from opt_query_suite import SuiteCatalog, agg, col, date, join, lit, op, scan
from opt_reference_eval import evaluate
from optimizer.cardinality_report import NodeEstimate
from optimizer.stats import (
    DEFAULT_EQ_SEL, DEFAULT_RANGE_SEL, DEFAULT_ROW_COUNT, CardinalityEstimator, annotate, estimate,
    estimate_rows,
)

# t: 100 rows. x = 0..99 (ndv 100); y = x % 10, NULL in rows 0-9 and 50-59 (ndv 10,
# 20% NULL); s = 'a'..'j'; d = 100 consecutive days of 2024; k = x % 20.
# u: 40 rows. k = 0..39 (ndv 40); g = k % 4.
_DAY0 = datetime.date(2024, 1, 1)
SCHEMAS = {
    "t": [("x", DType.INT), ("y", DType.INT), ("s", DType.STRING), ("d", DType.DATE), ("k", DType.INT)],
    "u": [("k", DType.INT), ("g", DType.INT)],
}
TABLES = {
    "t": (["x", "y", "s", "d", "k"], [
        (i, None if (i // 10) % 5 == 0 else i % 10, "abcdefghij"[i % 10],
         (_DAY0 + datetime.timedelta(days=i)).isoformat(), i % 20)
        for i in range(100)
    ]),
    "u": (["k", "g"], [(i, i % 4) for i in range(40)]),
}
CAT = SuiteCatalog(TABLES, SCHEMAS)


def rows(plan, catalog=CAT):
    return estimate_rows(plan, catalog)


def where(predicate, table="t"):
    return Filter(child=scan(table), predicate=predicate)


def actual(plan, tables=TABLES):
    return len(evaluate(plan, tables).rows)


# --------------------------------------------------------------------------
# Rules, one by one
# --------------------------------------------------------------------------


def test_scan_is_the_row_count():
    assert rows(scan("t")) == 100
    assert rows(scan("u")) == 40


def test_equality_is_one_over_ndv():
    assert rows(where(op("=", col("x"), lit(5)))) == pytest.approx(1.0)


def test_equality_excludes_nulls():
    # 80 non-NULL rows, 10 distinct values
    assert rows(where(op("=", col("y"), lit(3)))) == pytest.approx(8.0)


def test_equality_outside_min_max_is_empty():
    assert rows(where(op("=", col("x"), lit(500)))) == 0.0
    assert rows(where(op("=", col("x"), lit(-1)))) == 0.0


def test_comparison_with_null_is_empty():
    null = Literal(value=None, dtype=DType.INT)
    assert rows(where(op("=", col("x"), null))) == 0.0
    assert rows(where(op("<", col("x"), null))) == 0.0


def test_not_equal():
    assert rows(where(op("<>", col("x"), lit(5)))) == pytest.approx(99.0)
    assert rows(where(op("!=", col("y"), lit(3)))) == pytest.approx(72.0)


@pytest.mark.parametrize("pred, expected", [
    (op("<", col("x"), lit(0)), 0.0),        # below min: nothing
    (op("<=", col("x"), lit(0)), 1.0),       # min itself: one value
    (op("<=", col("x"), lit(99)), 100.0),    # up to max: everything
    (op("<", col("x"), lit(99)), 99.0),      # all but max
    (op(">=", col("x"), lit(99)), 1.0),      # max itself
    (op(">", col("x"), lit(99)), 0.0),       # above max
    (op(">", col("x"), lit(-5)), 100.0),     # below min: everything
    (op("<", col("x"), lit(1000)), 100.0),   # above max: everything
])
def test_range_endpoints_are_exact(pred, expected):
    assert rows(where(pred)) == pytest.approx(expected)


def test_range_interpolates_linearly():
    # P(x < 50) = (1 - 1/100) * (50 - 0) / (99 - 0)
    assert rows(where(op("<", col("x"), lit(50)))) == pytest.approx(100 * 0.99 * 50 / 99)


def test_lt_and_le_differ_by_one_value():
    lt = rows(where(op("<", col("x"), lit(30))))
    le = rows(where(op("<=", col("x"), lit(30))))
    eq = rows(where(op("=", col("x"), lit(30))))
    assert le - lt == pytest.approx(eq)


def test_literal_on_the_left_is_flipped():
    assert rows(where(op(">", lit(50), col("x")))) == rows(where(op("<", col("x"), lit(50))))
    assert rows(where(op("=", lit(7), col("x")))) == rows(where(op("=", col("x"), lit(7))))


def test_range_pair_is_one_interval_not_a_product():
    pair = rows(where(op("AND", op(">=", col("x"), lit(20)), op("<", col("x"), lit(30)))))
    assert pair == pytest.approx(10.0, rel=0.05)
    lower = rows(where(op(">=", col("x"), lit(20)))) / 100
    upper = rows(where(op("<", col("x"), lit(30)))) / 100
    assert pair != pytest.approx(100 * lower * upper)  # the independence answer, ~24


def test_tightest_bound_wins():
    assert rows(where(op("AND", op(">", col("x"), lit(10)), op(">", col("x"), lit(80))))) == \
        pytest.approx(rows(where(op(">", col("x"), lit(80)))))


def test_contradictory_bounds_are_empty():
    assert rows(where(op("AND", op(">", col("x"), lit(50)), op("<", col("x"), lit(10))))) == 0.0
    assert rows(where(op("AND", op("=", col("x"), lit(1)), op("=", col("x"), lit(2))))) == 0.0
    assert rows(where(op("AND", op("=", col("x"), lit(1)), op(">", col("x"), lit(2))))) == 0.0
    assert rows(where(op("AND", op("=", col("x"), lit(1)), op("<>", col("x"), lit(1))))) == 0.0


def test_equality_inside_bounds_is_one_over_ndv():
    assert rows(where(op("AND", op("=", col("x"), lit(5)), op("<", col("x"), lit(10))))) == pytest.approx(1.0)


def test_not_equal_inside_a_range_removes_one_value():
    # x < 10 keeps ~10 rows; excluding the value 3 removes one more
    base = rows(where(op("<", col("x"), lit(10))))
    both = rows(where(op("AND", op("<", col("x"), lit(10)), op("<>", col("x"), lit(3)))))
    assert base - both == pytest.approx(1.0)


def test_and_of_different_columns_multiplies():
    a = rows(where(op("=", col("x"), lit(5)))) / 100
    b = rows(where(op("<", col("k"), lit(10)))) / 100
    both = rows(where(op("AND", op("=", col("x"), lit(5)), op("<", col("k"), lit(10)))))
    assert both == pytest.approx(100 * a * b)


def test_or_is_inclusion_exclusion():
    a = rows(where(op("<", col("x"), lit(30)))) / 100
    b = rows(where(op("<", col("k"), lit(5)))) / 100
    either = rows(where(op("OR", op("<", col("x"), lit(30)), op("<", col("k"), lit(5)))))
    assert either == pytest.approx(100 * (a + b - a * b))


@pytest.mark.parametrize("inner, equivalent", [
    (op("<", col("x"), lit(30)), op(">=", col("x"), lit(30))),
    (op("=", col("x"), lit(30)), op("<>", col("x"), lit(30))),
    (op(">=", col("y"), lit(5)), op("<", col("y"), lit(5))),  # NULLs fail both: not 1 - sel
    (UnaryOp(op="IS NULL", operand=col("y")), UnaryOp(op="IS NOT NULL", operand=col("y"))),
    (UnaryOp(op="NOT", operand=op("<", col("x"), lit(7))), op("<", col("x"), lit(7))),
])
def test_not_is_pushed_into_the_predicate(inner, equivalent):
    assert rows(where(UnaryOp(op="NOT", operand=inner))) == pytest.approx(rows(where(equivalent)))


def test_not_of_a_comparison_respects_nulls():
    # 80 non-NULL y rows: y >= 5 keeps 40 of them, NOT (y >= 5) keeps the other 40, not 60
    assert rows(where(UnaryOp(op="NOT", operand=op(">=", col("y"), lit(5))))) == pytest.approx(40.0, rel=0.15)


def test_de_morgan():
    a, b = op("<", col("x"), lit(30)), op("=", col("k"), lit(4))
    not_and = rows(where(UnaryOp(op="NOT", operand=op("AND", a, b))))
    or_of_nots = rows(where(op("OR", op(">=", col("x"), lit(30)), op("<>", col("k"), lit(4)))))
    assert not_and == pytest.approx(or_of_nots)


def test_not_of_something_opaque_is_one_minus():
    opaque = op("LIKE", col("s"), lit("a%"))
    assert rows(where(UnaryOp(op="NOT", operand=opaque))) == pytest.approx(100 - rows(where(opaque)))


def test_is_null_uses_the_null_count():
    assert rows(where(UnaryOp(op="IS NULL", operand=col("y")))) == pytest.approx(20.0)
    assert rows(where(UnaryOp(op="is_not_null", operand=col("y")))) == pytest.approx(80.0)
    assert rows(where(UnaryOp(op="IS NULL", operand=col("x")))) == 0.0


def test_is_null_and_a_comparison_is_empty():
    pred = op("AND", UnaryOp(op="IS NULL", operand=col("y")), op("=", col("y"), lit(3)))
    assert rows(where(pred)) == 0.0


def test_is_null_of_an_expression_is_null_if_any_input_is():
    assert rows(where(UnaryOp(op="IS NULL", operand=op("+", col("x"), col("y"))))) == pytest.approx(20.0)


def test_two_columns_equal_is_one_over_the_larger_ndv():
    # x (ndv 100) = k (ndv 20)
    assert rows(where(op("=", col("x"), col("k")))) == pytest.approx(1.0)
    assert rows(where(op("=", col("x"), col("x")))) == pytest.approx(100.0)
    assert rows(where(op("<", col("x"), col("x")))) == 0.0


def test_defaults_where_statistics_cannot_answer():
    computed = op("+", col("x"), col("k"))
    assert rows(where(op("=", computed, lit(5)))) == pytest.approx(100 * DEFAULT_EQ_SEL)
    assert rows(where(op(">", computed, lit(5)))) == pytest.approx(100 * DEFAULT_RANGE_SEL)


def test_string_ranges_are_exact_outside_min_max_and_default_inside():
    assert rows(where(op("<", col("s"), lit("a")))) == 0.0
    assert rows(where(op("<=", col("s"), lit("j")))) == pytest.approx(100.0)
    assert rows(where(op(">", col("s"), lit("z")))) == 0.0
    assert rows(where(op("<", col("s"), lit("e")))) == pytest.approx(100 * DEFAULT_RANGE_SEL)
    assert rows(where(op("=", col("s"), lit("e")))) == pytest.approx(10.0)
    assert rows(where(op("=", col("s"), lit("zzz")))) == 0.0


def test_dates_interpolate_as_day_numbers():
    # d covers 2024-01-01 .. 2024-04-09, one row per day
    assert rows(where(op("<", col("d"), date("2024-01-01")))) == 0.0
    assert rows(where(op("<", col("d"), date("2024-02-20")))) == pytest.approx(50.0, rel=0.02)
    assert rows(where(op(">", col("d"), date("2025-01-01")))) == 0.0


class _DateObjectCatalog(SuiteCatalog):
    """A catalog that reports DATE min/max as ``datetime.date``, not ISO strings."""

    def stats(self, table, column):
        s = super().stats(table, column)
        if column != "d":
            return s
        return type(s)(s.ndv, datetime.date.fromisoformat(s.min), datetime.date.fromisoformat(s.max), s.null_count)


def test_dates_work_as_strings_or_date_objects():
    plan = where(op("<", col("d"), date("2024-02-20")))
    as_objects = _DateObjectCatalog(TABLES, SCHEMAS)
    assert rows(plan, as_objects) == pytest.approx(rows(plan))
    literal_object = where(op("<", col("d"), Literal(value=datetime.date(2024, 2, 20), dtype=DType.DATE)))
    assert rows(literal_object) == pytest.approx(rows(plan))


def test_constant_folding_is_applied_first():
    assert rows(where(op("<", col("x"), op("+", lit(40), lit(10))))) == rows(where(op("<", col("x"), lit(50))))
    assert rows(where(op("=", lit(1), lit(0)))) == 0.0
    assert rows(where(op("OR", op("=", col("x"), lit(5)), lit(True)))) == 100.0


def test_literal_predicates():
    assert rows(where(lit(True))) == 100.0
    assert rows(where(lit(False))) == 0.0
    assert rows(where(Literal(value=None, dtype=DType.BOOL))) == 0.0


def test_pushed_predicate_may_read_unselected_columns():
    pushed = Scan(table="t", columns=["s"], pushed_predicate=op("=", col("x"), lit(5)))
    assert rows(pushed) == pytest.approx(1.0)
    assert [ref for ref, _ in estimate(pushed, CAT).columns] == [("t", "s")]


def test_inner_join_is_product_over_larger_ndv():
    # |t| * |u| / max(ndv t.k = 20, ndv u.k = 40)
    plan = join(scan("t"), scan("u"), op("=", col("k", "t"), col("k", "u")))
    assert rows(plan) == pytest.approx(100 * 40 / 40)
    assert rows(plan) == actual(plan)


def test_join_excludes_null_keys():
    # y: 20% NULL, ndv 10; u.g: ndv 4
    plan = join(scan("t"), scan("u"), op("=", col("y"), col("g")))
    assert rows(plan) == pytest.approx(80 * 40 / 10)


def test_cross_join_is_the_product():
    assert rows(join(scan("t"), scan("u"), lit(True))) == 4000.0
    assert rows(join(scan("t"), scan("u"), lit(False))) == 0.0


def test_left_join_keeps_every_left_row():
    filtered = where(op("<", col("k"), lit(5)), "u")
    plan = join(scan("t"), filtered, op("=", col("k", "t"), col("k", "u")), kind="left")
    inner = join(scan("t"), filtered, op("=", col("k", "t"), col("k", "u")))
    assert rows(plan) >= 100
    assert rows(plan) >= rows(inner)
    # t.k = 0..19 and only 5 keys survive: about 3/4 of t's rows are unmatched
    est = estimate(plan, CAT)
    assert est.column(("u", "g")).null_frac == pytest.approx(0.75, abs=0.05)
    assert rows(plan) == pytest.approx(actual(plan), rel=0.1)


def test_left_join_without_matches_is_the_left_side():
    plan = join(scan("u"), scan("t"), lit(False), kind="left")
    assert rows(plan) == 40.0


def test_global_aggregate_is_one_row_even_over_nothing():
    assert rows(Aggregate(child=scan("t"), group_keys=[], aggs=[(agg("count"), "n")])) == 1.0
    empty = where(lit(False))
    assert rows(Aggregate(child=empty, group_keys=[], aggs=[(agg("count"), "n")])) == 1.0


def test_grouped_aggregate_is_the_product_of_ndv_capped_by_rows():
    assert rows(Aggregate(child=scan("t"), group_keys=[col("k")], aggs=[])) == pytest.approx(20.0)
    assert rows(Aggregate(child=scan("t"), group_keys=[col("x"), col("k")], aggs=[])) == pytest.approx(100.0)


def test_null_is_a_group_of_its_own():
    plan = Aggregate(child=scan("t"), group_keys=[col("y")], aggs=[])
    assert rows(plan) == pytest.approx(11.0)
    assert rows(plan) == actual(plan)


def test_limit():
    assert rows(Limit(child=scan("t"), n=7)) == 7.0
    assert rows(Limit(child=scan("t"), n=500)) == 100.0
    assert rows(Limit(child=scan("t"), n=0)) == 0.0


def test_filter_narrows_column_statistics():
    after_eq = estimate(where(op("=", col("k"), lit(3))), CAT).column(("t", "k"))
    assert (after_eq.ndv, after_eq.min, after_eq.max, after_eq.null_frac) == (1.0, 3, 3, 0.0)
    after_range = estimate(where(op("<", col("x"), lit(10))), CAT).column(("t", "x"))
    assert after_range.max == 10 and after_range.ndv == pytest.approx(10.0, rel=0.1)
    after_cmp = estimate(where(op(">", col("y"), lit(2))), CAT).column(("t", "y"))
    assert after_cmp.null_frac == 0.0


def test_group_by_after_a_filter_uses_the_narrowed_column():
    plan = Aggregate(child=where(op("=", col("k"), lit(3))), group_keys=[col("k")], aggs=[])
    assert rows(plan) == pytest.approx(1.0)
    # (An unrelated filter is covered by the workload's group_by_after_filter.)


def test_unconstrained_columns_lose_values_by_the_urn_model():
    # keeping ~10 of 100 rows: of k's 20 values (5 rows each), 1 - 0.9^5 = 41% survive
    est = estimate(where(op("<", col("x"), lit(10))), CAT).column(("t", "k"))
    sel = rows(where(op("<", col("x"), lit(10)))) / 100
    assert est.ndv == pytest.approx(20 * (1 - (1 - sel) ** 5))


def test_ndv_never_exceeds_rows():
    est = estimate(Limit(child=scan("t"), n=3), CAT)
    assert all(c.ndv <= 3 for _, c in est.columns)


def test_qualified_refs_resolve_strictly_across_join_sides():
    # dept.id is not emp.id: the estimate must use u.k's statistics, not t.k's
    plan = join(scan("t"), scan("u"), op("=", col("x", "t"), col("k", "u")))
    assert rows(plan) == pytest.approx(100 * 40 / 100)


def test_unknown_catalog_falls_back_to_defaults():
    class NoStats:
        def schema(self, table):
            return SCHEMAS[table]

        def row_count(self, table):
            return 100

    assert rows(where(op("=", col("x"), lit(5))), NoStats()) == pytest.approx(100 * DEFAULT_EQ_SEL)
    assert rows(Scan(table="nowhere", columns=None, pushed_predicate=None), CAT) == DEFAULT_ROW_COUNT


def test_estimator_does_not_mutate_and_memoizes():
    plan = S.QUERIES[0].plan
    before = repr(plan)
    est = CardinalityEstimator(S.CATALOG)
    first = est.estimate(plan)
    assert est.estimate(plan) is first
    assert repr(plan) == before


def test_annotate_lists_every_node_in_preorder():
    plan = join(where(op("<", col("x"), lit(5))), scan("u"), op("=", col("k", "t"), col("k", "u")))
    paths = [path for path, _, _ in annotate(plan, CAT)]
    assert paths == [(), (0,), (0, 0), (1,)]


# --------------------------------------------------------------------------
# Invariants, on every plan in both suites
# --------------------------------------------------------------------------


def _plans():
    for q in S.QUERIES:
        yield f"suite:{q.name}", q.plan, S.CATALOG
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield f"suite-optimized:{q.name}", optimizer.optimize(q.plan, S.CATALOG)[0], S.CATALOG
    for q in W.QUERIES:
        yield f"workload:{q.name}", q.plan, W.CATALOG


PLANS = list(_plans())


@pytest.mark.parametrize("name, plan, catalog", PLANS, ids=[p[0] for p in PLANS])
def test_estimates_are_consistent(name, plan, catalog):
    est = CardinalityEstimator(catalog)
    for _, node, e in annotate(plan, catalog):
        assert math.isfinite(e.rows) and e.rows >= 0, node
        for _, c in e.columns:
            assert 0.0 <= c.null_frac <= 1.0 + 1e-9, node
            assert c.ndv is None or c.ndv <= e.rows + 1e-9, node
        kids = [est.rows(child) for child in node.children]
        if isinstance(node, Scan):
            assert e.rows <= catalog.row_count(node.table) + 1e-9
        elif isinstance(node, (Filter, Limit)):
            assert e.rows <= kids[0] + 1e-9
        elif isinstance(node, Limit):
            assert e.rows <= node.n
        elif isinstance(node, Aggregate):
            assert e.rows == 1.0 if not node.group_keys else e.rows <= kids[0] + 1e-9
        elif isinstance(node, Join):
            assert e.rows <= kids[0] * kids[1] + 1e-9 or node.kind == "left"
            if node.kind == "left":
                assert e.rows >= kids[0] - 1e-9


# --------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------


def _q_error(q):
    plan = q.plan
    return NodeEstimate(q.name, (), "", "", rows(plan, W.CATALOG), actual(plan, W.TABLES)).q_error


@pytest.mark.parametrize("q", [q for q in W.QUERIES if q.assumption_holds], ids=lambda q: q.name)
def test_accurate_where_the_assumptions_hold(q):
    assert _q_error(q) <= 1.5


@pytest.mark.parametrize("q", [q for q in W.QUERIES if not q.assumption_holds], ids=lambda q: q.name)
def test_workload_really_breaks_the_assumption(q):
    # A negative control: if this fails, the query no longer demonstrates a failure mode.
    assert _q_error(q) >= 1.4


def test_suite_catalog_stats_match_person_a_catalog():
    pa = pytest.importorskip("pyarrow")
    try:
        from catalog.catalog import Catalog
    except ImportError as e:  # Person A's catalog needs the complete IR
        pytest.skip(f"catalog not importable: {e}")
    real = Catalog()
    for table, (names, data) in S.TABLES.items():
        arrays = {}
        for i, (name, dtype) in enumerate(S.SCHEMAS[table]):
            values = [r[i] for r in data]
            if dtype == DType.DATE:
                values = [None if v is None else datetime.date.fromisoformat(v) for v in values]
            arrays[name] = pa.array(values)
        real.register_table(table, pa.table(arrays))
    for table, schema in S.SCHEMAS.items():
        assert real.row_count(table) == S.CATALOG.row_count(table)
        for name, _ in schema:
            ours, theirs = S.CATALOG.stats(table, name), real.stats(table, name)
            assert (ours.ndv, ours.min, ours.max, ours.null_count) == \
                (theirs.ndv, theirs.min, theirs.max, theirs.null_count), (table, name)
