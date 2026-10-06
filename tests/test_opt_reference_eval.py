"""Sanity tests for the reference evaluator. If the oracle is wrong, every correctness test is worthless."""

import pytest

import opt_ir  # noqa: F401
from ir.dtype import DType
from ir.expr import Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Limit, Scan, Sort
from opt_query_suite import TABLES, agg, col, join, keep, lit, op, project, scan
from opt_reference_eval import assert_equivalent, evaluate

NULL = Literal(value=None, dtype=DType.BOOL)


def rows(plan):
    return evaluate(plan, TABLES).rows


def test_scan_reads_requested_columns_in_order():
    result = evaluate(Scan("emp", ["salary", "id"], None), TABLES)
    assert result.columns == [("emp", "salary"), ("emp", "id")]
    assert result.rows[0] == (120.0, 1)


def test_inner_join_drops_dangling_keys_and_left_join_pads_nulls():
    cond = op("=", col("c_id", "customer"), col("o_custkey", "orders"))
    inner = rows(join(scan("customer"), scan("orders"), cond))
    left = rows(join(scan("customer"), scan("orders"), cond, kind="left"))

    custkeys = {r[1] for r in TABLES["orders"][1]}
    no_orders = {r[0] for r in TABLES["customer"][1]} - custkeys
    assert {11, 12} <= no_orders  # by construction; random draws may add more
    assert {13, 14} & custkeys  # orders pointing at customers that don't exist

    assert {r[0] for r in inner}.isdisjoint(no_orders)
    assert len(inner) == sum(1 for r in TABLES["orders"][1] if r[1] <= 12)
    padded = [r for r in left if r[0] in no_orders]
    assert len(padded) == len(no_orders) and all(r[5:] == (None,) * 5 for r in padded)
    assert len(left) == len(inner) + len(no_orders)


def test_qualified_reference_does_not_resolve_through_a_project():
    plan = project(project(scan("emp"), *keep("name", table="emp")), (col("name", "emp"), "x"))
    with pytest.raises(LookupError, match="unresolved"):
        evaluate(plan, TABLES)


def test_unqualified_reference_to_a_colliding_name_is_ambiguous():
    j = join(scan("emp"), scan("dept"), op("=", col("dept_id", "emp"), col("id", "dept")))
    with pytest.raises(LookupError, match="ambiguous"):
        evaluate(project(j, *keep("name")), TABLES)


def test_filter_keeps_only_true_rows_under_three_valued_logic():
    amounts = [r[3] for r in TABLES["sales"][1]]
    assert None in amounts
    over = rows(Filter(scan("sales"), op(">", col("amount"), lit(100.0))))
    not_over = rows(Filter(scan("sales"), UnaryOp("NOT", op(">", col("amount"), lit(100.0)))))
    # NULL amounts satisfy neither the predicate nor its negation.
    assert len(over) + len(not_over) == len(amounts) - amounts.count(None)


@pytest.mark.parametrize("left, right, and_, or_", [
    (True, None, None, True),
    (False, None, False, None),
    (None, None, None, None),
])
def test_and_or_truth_tables_with_null(left, right, and_, or_):
    def value(v):
        return NULL if v is None else lit(v)

    base = Scan("dept", ["id"], None)
    for symbol, expected in (("AND", and_), ("OR", or_)):
        plan = project(Limit(base, 1), (op(symbol, value(left), value(right)), "v"))
        assert rows(plan) == [(expected,)]


def test_aggregates_skip_nulls_and_count_star_counts_rows():
    plan = Aggregate(scan("sales"), [], [(agg("count"), "n"), (agg("count", col("amount")), "n_amount"),
                                         (agg("sum", col("amount")), "s")])
    ((n, n_amount, s),) = rows(plan)
    amounts = [r[3] for r in TABLES["sales"][1]]
    assert n == 60 and n_amount == 60 - amounts.count(None)
    assert s == pytest.approx(sum(a for a in amounts if a is not None))


def test_global_aggregate_over_empty_input_returns_one_row():
    empty = Filter(scan("sales"), op("<", col("qty"), lit(0)))
    assert rows(Aggregate(empty, [], [(agg("count"), "n"), (agg("sum", col("amount")), "s")])) == [(0, None)]
    assert rows(Aggregate(empty, [col("region")], [(agg("count"), "n")])) == []


def test_sort_puts_nulls_last_ascending_and_first_descending():
    base = Scan("sales", ["amount"], None)
    asc = [r[0] for r in rows(Sort(base, [(col("amount"), False)]))]
    desc = [r[0] for r in rows(Sort(base, [(col("amount"), True)]))]
    assert asc[-1] is None and desc[0] is None
    present = [a for a in asc if a is not None]
    assert present == sorted(present)


def test_assert_equivalent_detects_changed_columns_and_rows():
    plan = project(scan("emp"), *keep("id", "name"))
    assert_equivalent(plan, plan, TABLES)
    with pytest.raises(AssertionError, match="output columns changed"):
        assert_equivalent(plan, project(scan("emp"), *keep("name", "id")), TABLES)
    with pytest.raises(AssertionError, match="rows differ"):
        assert_equivalent(plan, Filter(plan, op(">", col("id"), lit(1))), TABLES)


def test_assert_equivalent_respects_order_only_when_the_query_sorts():
    asc = Sort(scan("emp"), [(col("id"), False)])
    desc = Sort(scan("emp"), [(col("id"), True)])
    with pytest.raises(AssertionError, match="ordered rows differ"):
        assert_equivalent(asc, desc, TABLES)
    with pytest.raises(AssertionError, match="ordered rows differ"):
        assert_equivalent(project(asc, *keep("id")), project(desc, *keep("id")), TABLES)

    # Under an Aggregate the sort order does not matter: the same groups come
    # out in a different order, and that is still the same answer.
    by_asc = Aggregate(asc, [col("id")], [])
    by_desc = Aggregate(desc, [col("id")], [])
    assert evaluate(by_asc, TABLES).rows != evaluate(by_desc, TABLES).rows
    assert_equivalent(by_asc, by_desc, TABLES)
