"""Plan tests for column pruning: the tree changes exactly the way the pass claims.

Correctness (the answer does not change) is in test_opt_differential.py.
"""

import dataclasses

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
from ir.nodes import Aggregate, Filter, Join, Project, Scan
from opt_query_suite import CATALOG, agg, col, join, keep, lit, op, project, scan
from optimizer.column_pruning import ColumnPruning

PRUNE = ColumnPruning()


def prune(plan, catalog=CATALOG):
    return PRUNE.apply(plan, catalog)


def scans(plan) -> dict[str, list[str] | None]:
    """Map each scanned table to its Scan.columns."""
    if isinstance(plan, Scan):
        return {plan.table: plan.columns}
    out = {}
    for child in plan.children:
        out.update(scans(child))
    return out


def find(plan, kind):
    """Return every node of type ``kind`` in the plan, pre-order."""
    found = [plan] if isinstance(plan, kind) else []
    for child in plan.children:
        found += find(child, kind)
    return found


def aliases(node) -> list[str]:
    pairs = node.exprs if isinstance(node, Project) else node.aggs
    return [alias for _, alias in pairs]


# --------------------------------------------------------------------------
# Scan narrowing
# --------------------------------------------------------------------------


def test_selecting_two_of_twenty_columns_reads_two():
    assert len(S.SCHEMAS["sales"]) == 20
    assert scans(prune(S.two_of_twenty())) == {"sales": ["sale_id", "amount"]}


def test_filter_only_column_is_still_read():
    # The classic bug: `region` never reaches the output, but the Filter needs it.
    assert scans(prune(S.filter_only_column())) == {"sales": ["sale_id", "region"]}


def test_join_condition_only_columns_are_still_read():
    assert scans(prune(S.join_condition_only_columns())) == {
        "orders": ["o_id", "o_custkey"],
        "customer": ["c_id", "c_name"],
    }


def test_sort_key_only_column_is_still_read():
    assert scans(prune(S.sort_key_only_column())) == {"sales": ["sale_id", "amount", "qty"]}


def test_pushed_predicate_columns_are_still_read():
    assert scans(prune(S.pushed_predicate_column())) == {"sales": ["sale_id", "amount"]}


def test_qualifiers_separate_tables_with_colliding_column_names():
    # Both tables have `id` and `name`; each keeps only what is qualified with its own name.
    assert scans(prune(S.colliding_names())) == {
        "emp": ["name", "dept_id"],
        "dept": ["id", "name"],
    }


def test_kept_columns_follow_table_order_not_reference_order():
    # Referenced as qty, amount, region; the table order is region, amount, qty.
    assert scans(prune(S.computed_projection())) == {"sales": ["region", "amount", "qty"]}


def test_explicit_scan_columns_keep_their_own_order():
    plan = project(
        Scan(table="sales", columns=["qty", "sale_id", "region"], pushed_predicate=None),
        *keep("region", "sale_id"),
    )
    assert scans(prune(plan)) == {"sales": ["sale_id", "region"]}


def test_count_star_keeps_one_narrow_column_so_rows_survive():
    out = prune(S.count_star())
    (scan_node,) = find(out, Scan)
    assert scan_node.columns == ["sale_id"]  # the first INT column, not a STRING


def test_unknown_qualifier_is_treated_conservatively():
    # "o" is not a scanned table (it might be an alias), so the name alone decides.
    plan = project(scan("orders"), (col("o_total", "o"), "t"))
    assert scans(prune(plan)) == {"orders": ["o_total"]}


@pytest.mark.parametrize("catalog", [None, type("Empty", (), {"schema": lambda s, t: {}[t]})()])
def test_scan_without_known_schema_is_left_alone(catalog):
    plan = project(scan("sales"), *keep("sale_id"))
    assert prune(plan, catalog) is plan


def test_bound_scan_needs_no_catalog():
    plan = project(scan("sales", bound=True), *keep("sale_id"))
    assert scans(prune(plan, catalog=None)) == {"sales": ["sale_id"]}


# --------------------------------------------------------------------------
# Project and Aggregate tightening
# --------------------------------------------------------------------------


def test_unused_project_expressions_are_dropped():
    out = prune(S.nested_projects())
    outer, inner = find(out, Project)
    assert aliases(outer) == ["a"]
    assert aliases(inner) == ["a"]
    assert scans(out) == {"sales": ["sale_id"]}


def test_unused_aggregates_are_dropped_but_used_ones_stay():
    (aggregate,) = find(prune(S.aggregate_unused_aggs()), Aggregate)
    assert aliases(aggregate) == ["total"]
    assert scans(aggregate) == {"sales": ["region", "amount"]}


def test_aggregate_used_only_by_having_is_kept():
    (aggregate,) = find(prune(S.having_on_aggregate()), Aggregate)
    assert aliases(aggregate) == ["total"]


def test_aggregate_named_by_its_call_in_having_is_kept():
    # The binder writes HAVING COUNT(*) > 12 with the call itself, not the alias
    (aggregate,) = find(prune(S.having_names_an_aggregate_the_select_list_drops()), Aggregate)
    assert aggregate.aggs == [(agg("count"), "count(*)")]


def test_aggregate_named_by_its_call_in_order_by_is_kept():
    (aggregate,) = find(prune(S.order_by_an_aggregate_the_select_list_drops()), Aggregate)
    assert aggregate.aggs == [(agg("sum", col("amount")), "sum(amount)")]
    assert scans(aggregate) == {"sales": ["region", "amount"]}


def test_aggregates_nothing_names_are_still_dropped():
    count, top = agg("count"), agg("max", col("qty"))
    grouped = Aggregate(child=scan("sales"), group_keys=[col("region")],
                        aggs=[(count, "count(*)"), (top, "max(qty)")])
    plan = project(Filter(child=grouped, predicate=op(">", count, lit(12))), *keep("region"))
    (aggregate,) = find(prune(plan), Aggregate)
    assert aggregate.aggs == [(count, "count(*)")]


def test_group_keys_are_never_dropped():
    (aggregate,) = find(prune(S.group_by_without_aggs()), Aggregate)
    assert aggregate.group_keys == [col("region")]
    assert aggregate.aggs == []


def test_global_aggregate_keeps_one_aggregate_preferring_count_star():
    (aggregate,) = find(prune(S.constant_over_global_aggregate()), Aggregate)
    assert aggregate.aggs == [(agg("count"), "n")]


def test_root_output_is_never_pruned():
    root_project = project(scan("sales"), *keep("sale_id", "amount", "region"))
    root_aggregate = dataclasses.replace(S.aggregate_unused_aggs().child)  # unwrap the Project
    assert aliases(prune(root_project)) == ["sale_id", "amount", "region"]
    assert aliases(prune(root_aggregate)) == ["total", "n", "max_qty"]


def test_select_star_is_untouched():
    plan = S.select_star()
    assert prune(plan) is plan


# --------------------------------------------------------------------------
# Project insertion above joins
# --------------------------------------------------------------------------


def test_project_inserted_between_joins_drops_inner_join_keys():
    out = prune(S.three_way_join_unqualified())
    outer = out.child
    assert isinstance(outer, Join)
    inserted = outer.left
    assert isinstance(inserted, Project) and isinstance(inserted.child, Join)
    assert inserted.exprs == [(col("o_total", "orders"), "o_total"),
                              (col("c_nationkey", "customer"), "c_nationkey")]


def test_project_inserted_between_filter_and_join():
    out = prune(S.filter_over_join_unqualified())
    filt = out.child
    assert isinstance(filt, Filter)
    assert isinstance(filt.child, Project) and isinstance(filt.child.child, Join)
    assert aliases(filt.child) == ["o_id", "c_segment"]


def test_no_insertion_when_qualified_references_would_break():
    # Above a Project, `customer.c_nationkey` would no longer resolve (its
    # output is unqualified), so the pass must leave the join alone.
    out = prune(S.three_way_join_qualified())
    assert isinstance(out.child.left, Join)
    assert not find(out, Project)


def test_no_insertion_when_kept_names_would_collide():
    # Unqualified `name` matches both emp.name and dept.name. A Project would
    # need two outputs named `name`, so the pass declines.
    j = join(scan("emp"), scan("dept"), op("=", col("dept_id", "emp"), col("id", "dept")))
    plan = project(Filter(child=j, predicate=op("=", col("name"), lit("x"))), *keep("salary"))
    assert isinstance(prune(plan).child.child, Join)


def test_no_insertion_directly_under_project_or_aggregate():
    # Those already narrow their input themselves.
    out = prune(S.join_condition_only_columns())
    assert isinstance(out.child, Join)


# --------------------------------------------------------------------------
# Fixed point and unknown nodes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_pruning_is_idempotent(query):
    once = prune(query.plan)
    assert prune(once) is once


@dataclasses.dataclass(frozen=True)
class Opaque:
    """A plan node kind the pass has never heard of."""

    child: object

    @property
    def children(self):
        return (self.child,)

    def replace_children(self, new_children):
        (child,) = new_children
        return Opaque(child)


def test_below_an_unknown_node_everything_is_kept():
    plan = project(Opaque(scan("sales")), *keep("sale_id"))
    assert prune(plan) is plan
