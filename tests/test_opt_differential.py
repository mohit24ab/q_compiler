"""Correctness tests: optimized plans return the same answer as the original (Contract §7).

Every query in the suite runs through the full default pipeline and through
each default pass on its own, and is compared against the reference
evaluator. Add each new pass to the default pipeline and this file covers it
with no edits.

The negative controls at the bottom show that this harness actually
catches the classic pruning bugs, rather than passing everything.
"""

import warnings

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
import optimizer
from ir.nodes import Join, Project, Scan
from opt_query_suite import CATALOG, TABLES, col
from opt_reference_eval import assert_equivalent
from optimizer.column_pruning import ColumnPruning


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_full_pipeline_preserves_results(query):
    plan = query.plan
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # oscillation or the iteration cap fails the test
        final, traces = optimizer.optimize(plan, CATALOG)
    assert_equivalent(plan, final, TABLES)
    assert not any(t.changed for t in traces if t.iteration == traces[-1].iteration)


PASSES = optimizer.default_passes()


@pytest.mark.parametrize("opt_pass", PASSES, ids=lambda p: p.name)
@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_each_pass_alone_preserves_results(query, opt_pass):
    plan = query.plan
    assert_equivalent(plan, opt_pass.apply(plan, CATALOG), TABLES)


# --------------------------------------------------------------------------
# Negative controls: the harness must catch these broken rewrites
# --------------------------------------------------------------------------


def _replace_scan(plan, table, columns):
    if isinstance(plan, Scan):
        return Scan(plan.table, columns, plan.pushed_predicate, plan.table_schema) if plan.table == table else plan
    return plan.replace_children(tuple(_replace_scan(c, table, columns) for c in plan.children))


def test_harness_catches_dropping_a_filter_only_column():
    plan = S.filter_only_column()
    broken = _replace_scan(ColumnPruning().apply(plan, CATALOG), "sales", ["sale_id"])
    with pytest.raises(LookupError, match="unresolved column reference"):
        assert_equivalent(plan, broken, TABLES)


def test_harness_catches_dropping_a_join_key():
    plan = S.join_condition_only_columns()
    broken = _replace_scan(ColumnPruning().apply(plan, CATALOG), "orders", ["o_id"])
    with pytest.raises(LookupError, match="o_custkey"):
        assert_equivalent(plan, broken, TABLES)


def test_harness_catches_a_project_that_breaks_qualified_references():
    # This is the insertion the pass refuses for three_way_join_qualified.
    plan = S.three_way_join_qualified()
    outer = plan.child
    inner = outer.left
    narrowed = Project(child=inner, exprs=[(col("o_total", "orders"), "o_total"),
                                           (col("c_nationkey", "customer"), "c_nationkey")])
    broken = plan.replace_children((outer.replace_children((narrowed, outer.right)),))
    assert isinstance(broken.child.left, Project) and isinstance(broken.child, Join)
    with pytest.raises(LookupError, match="customer"):
        assert_equivalent(plan, broken, TABLES)
