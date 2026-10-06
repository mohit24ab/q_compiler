"""Plan tests for predicate pushdown: one per direction a conjunct can move, one per case where it must not.

Correctness for every query in the suite is in test_opt_differential.py,
including negative controls showing the must-not-push queries really give
a different answer if pushed.
"""

import dataclasses

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan
from opt_query_suite import CATALOG, col, join, keep, lit, op, project, scan
from optimizer.expressions import TRUE, split_conjuncts
from optimizer.predicate_pushdown import PredicatePushdown

PUSH = PredicatePushdown()


def push(plan, catalog=CATALOG):
    return PUSH.apply(plan, catalog)


def find(plan, kind):
    found = [plan] if isinstance(plan, kind) else []
    for child in plan.children:
        found += find(child, kind)
    return found


def pushed(plan) -> dict[str, list]:
    """Map each scanned table to the conjuncts in its pushed_predicate."""
    return {s.table: split_conjuncts(s.pushed_predicate) for s in find(plan, Scan)}


def filters(plan) -> list:
    """List every conjunct still held in a Filter."""
    return [c for f in find(plan, Filter) for c in split_conjuncts(f.predicate)]


# --------------------------------------------------------------------------
# Directions a conjunct moves
# --------------------------------------------------------------------------


def test_stacked_filters_land_in_the_scan_inner_conjunct_first():
    out = push(S.stacked_filters())
    assert out == dataclasses.replace(
        scan("sales"),
        pushed_predicate=op("AND", op(">", col("qty"), lit(2)), op("=", col("region"), lit("EU"))),
    )


def test_new_conjuncts_append_to_an_existing_pushed_predicate():
    out = push(S.filter_merges_with_pushed_predicate())
    assert pushed(out) == {"sales": [op(">", col("amount"), lit(50.0)), op(">", col("qty"), lit(5))]}
    assert not filters(out)


def test_through_project_rewrites_a_computed_alias():
    out = push(S.filter_through_computed_project())
    assert isinstance(out, Project)  # the Filter is gone, the Project stays on top
    assert pushed(out) == {"sales": [op(">", op("*", col("qty"), col("amount")), lit(500.0))]}


def test_through_project_rewrites_aliases_to_qualified_join_columns():
    out = push(S.filter_through_project_over_join())
    assert pushed(out) == {
        "orders": [op(">", col("o_id", "orders"), lit(10))],
        "customer": [op("=", col("c_segment", "customer"), lit("BUILDING"))],
    }


def test_through_sort():
    out = push(S.filter_through_sort())
    assert not filters(out)
    assert pushed(out) == {"sales": [op("<", col("qty"), lit(4))]}


def test_inner_join_splits_left_right_and_both():
    out = push(S.where_splits_across_inner_join())
    (j,) = find(out, Join)
    assert pushed(out) == {
        "orders": [op(">", col("o_total"), lit(100.0))],
        "customer": [op("=", col("c_segment"), lit("AUTO"))],
    }
    assert split_conjuncts(j.condition) == [
        op("=", col("o_custkey"), col("c_id")),
        op(">", op("+", col("o_custkey"), col("c_nationkey")), lit(4)),
    ]
    assert not filters(out)


def test_inner_join_on_conjuncts_reading_one_side_move_into_that_side():
    out = push(S.single_side_on_conjuncts_of_inner_join())
    (j,) = find(out, Join)
    assert j.condition == op("=", col("o_custkey"), col("c_id"))
    assert pushed(out) == {"orders": [op("=", col("o_status"), lit("F"))],
                           "customer": [op(">", col("c_balance"), lit(0.0))]}


def test_through_two_joins_to_the_far_table():
    out = push(S.filter_through_three_way_join())
    assert pushed(out)["nation"] == [op("=", col("n_region"), lit("AMERICA"))]
    assert not filters(out)


def test_through_aggregate_on_group_key_only():
    out = push(S.having_splits_on_group_key())
    assert pushed(out) == {"sales": [op("<>", col("region"), lit("AP"))]}
    assert filters(out) == [op(">", col("total"), lit(1500.0))]  # HAVING on SUM stays
    (f,) = find(out, Filter)
    assert isinstance(f.child, Aggregate)


def test_left_join_where_on_preserved_side_pushes_left():
    out = push(S.left_join_where_on_preserved_side())
    (j,) = find(out, Join)
    assert j.kind == "left"
    assert pushed(out) == {"customer": [op("=", col("c_segment"), lit("AUTO"))], "orders": []}


def test_left_join_on_conjunct_reading_null_producing_side_pushes_right():
    out = push(S.left_join_on_conjuncts())
    (j,) = find(out, Join)
    assert j.kind == "left"
    assert pushed(out)["orders"] == [op("=", col("o_status"), lit("F"))]


def test_null_rejecting_where_turns_left_join_inner_then_pushes():
    out = push(S.left_join_null_rejecting_where_becomes_inner())
    (j,) = find(out, Join)
    assert j.kind == "inner"
    assert pushed(out)["orders"] == [op(">", col("o_total"), lit(200.0))]
    assert not filters(out)


def test_join_with_every_on_conjunct_pushed_keeps_a_true_condition():
    plan = join(scan("orders"), scan("customer"), op("=", col("o_status"), lit("F")))
    out = push(plan)
    assert out.condition == TRUE
    assert pushed(out)["orders"] == [op("=", col("o_status"), lit("F"))]


# --------------------------------------------------------------------------
# Cases where the conjunct must NOT move
# --------------------------------------------------------------------------


def test_not_through_limit():
    plan = S.filter_above_limit_must_not_push()
    out = push(plan)
    assert out == plan
    assert isinstance(out, Filter) and isinstance(out.child, Limit)


def test_not_through_aggregate_when_reading_an_aggregate_result():
    plan = project(Filter(child=S.aggregate_unused_aggs().child, predicate=op(">", col("total"), lit(0.0))),
                   *keep("region"))
    assert push(plan) == plan


def test_not_through_a_global_aggregate_even_a_constant():
    plan = S.constant_false_over_global_aggregate_must_not_push()
    assert push(plan) == plan


def test_left_join_is_null_on_null_producing_side_stays_above():
    plan = S.left_join_is_null_must_not_push()
    out = push(plan)
    assert out == plan
    (j,) = find(out, Join)
    assert j.kind == "left"


def test_left_join_or_that_can_accept_null_extended_rows_stays_above():
    plan = S.left_join_or_with_preserved_side_must_not_push()
    assert push(plan) == plan


def test_left_join_on_conjunct_reading_preserved_side_stays_in_condition():
    out = push(S.left_join_on_conjuncts())
    (j,) = find(out, Join)
    assert split_conjuncts(j.condition) == [op("=", col("c_id"), col("o_custkey")),
                                            op("=", col("c_segment"), lit("AUTO"))]
    assert pushed(out)["customer"] == []


def test_not_through_project_when_a_reference_is_qualified():
    # Above a Project, `t.value` does not resolve to the alias `value`, so the pass can't rewrite it.
    plan = Filter(child=project(scan("sales"), (col("qty"), "value")), predicate=op(">", col("value", "t"), lit(1)))
    assert push(plan) == plan


def test_unroutable_conjunct_joins_an_inner_condition_but_stays_above_a_left_join():
    # `x.c_name` uses a qualifier neither side has, so its side is unknown.
    pred = op("=", col("c_name", "x"), lit("cust#1"))
    inner = Filter(child=join(scan("customer"), scan("orders"), op("=", col("c_id"), col("o_custkey"))),
                   predicate=pred)
    left = Filter(child=dataclasses.replace(inner.child, kind="left"), predicate=pred)

    (j,) = find(push(inner), Join)
    assert split_conjuncts(j.condition)[-1] == pred
    assert push(left) == left


def test_join_side_with_unknown_columns_blocks_routing():
    j = join(scan("orders"), scan("customer"), op("=", col("o_custkey"), col("c_id")))
    plan = Filter(child=j, predicate=op(">", col("o_total"), lit(1.0)))
    out = push(plan, catalog=None)  # no schemas, so the pass can't tell which side o_total is on
    (jn,) = find(out, Join)
    assert pushed(out) == {"orders": [], "customer": []}
    assert split_conjuncts(jn.condition)[-1] == op(">", col("o_total"), lit(1.0))


@dataclasses.dataclass(frozen=True)
class Opaque:
    child: object

    @property
    def children(self):
        return (self.child,)

    def replace_children(self, new_children):
        (child,) = new_children
        return Opaque(child)


def test_not_through_an_unknown_node_but_its_subtree_is_still_optimized():
    inner = Filter(child=scan("sales"), predicate=op(">", col("qty"), lit(1)))
    plan = Filter(child=Opaque(inner), predicate=op("=", col("region"), lit("EU")))
    out = push(plan)
    assert isinstance(out, Filter) and isinstance(out.child, Opaque)
    assert out.child.child == dataclasses.replace(scan("sales"), pushed_predicate=op(">", col("qty"), lit(1)))


# --------------------------------------------------------------------------
# Fixed point
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_pushdown_is_idempotent(query):
    once = push(query.plan)
    assert push(once) == once


def test_untouched_plan_is_returned_as_the_same_object():
    plan = S.two_of_twenty()
    assert push(plan) is plan
