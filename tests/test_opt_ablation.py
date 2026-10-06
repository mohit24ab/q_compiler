"""The ablation runner and the pass order study (optimizer/ablation.py)."""

from pathlib import Path

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
import optimizer
from ir.nodes import Filter, Scan
from opt_query_suite import col, join, lit, op, scan
from opt_reference_eval import evaluate
from optimizer.ablation import (
    Config, default_configs, optimize_with, order_table, pass_order_study, rank_correlation, run_ablation,
    scan_metrics, summary_table, to_csv,
)

PASSES = [p.name for p in optimizer.default_passes()]


def reference(plan):
    return lambda: evaluate(plan, S.TABLES).rows


def test_default_order():
    # docs/optimizer.md justifies this order with the study below
    assert PASSES == ["predicate_pushdown", "join_reordering", "column_pruning", "constant_folding"]


def test_default_configs():
    configs = default_configs(PASSES)
    assert [c.name for c in configs] == ["all passes"] + [f"without {p}" for p in PASSES] + ["no passes"]
    assert configs[0].passes == tuple(PASSES)
    assert configs[-1].passes == ()
    assert all(p not in c.passes for p, c in zip(PASSES, configs[1:-1]))


def test_optimize_with_runs_only_the_named_passes():
    plan = S.filter_only_column()
    assert optimize_with(plan, S.CATALOG, ()) is plan
    pushed_only = optimize_with(plan, S.CATALOG, ["predicate_pushdown"])
    assert isinstance(pushed_only.child, Scan) and pushed_only.child.pushed_predicate is not None
    assert pushed_only.child.columns is None  # pruning did not run
    pruned_only = optimize_with(plan, S.CATALOG, ["column_pruning"])
    assert isinstance(pruned_only.child, Filter)  # pushdown did not run


def test_scan_metrics():
    plain = Scan(table="sales", columns=["sale_id"], pushed_predicate=None)
    assert scan_metrics(plain, S.CATALOG) == (60, 60)
    pushed = Scan(table="sales", columns=["sale_id"], pushed_predicate=op("=", col("region"), lit("EU")))
    assert scan_metrics(pushed, S.CATALOG) == (60, 120)  # sale_id and region are read
    everything = scan("sales")
    assert scan_metrics(everything, S.CATALOG) == (60, 60 * 20)
    both = join(plain, scan("emp"), lit(True))
    assert scan_metrics(both, S.CATALOG) == (65, 60 + 5 * 4)


def test_scan_metrics_skip_subtrees_that_never_run():
    empty = Filter(child=join(scan("orders"), scan("customer"), lit(True)), predicate=lit(False))
    assert scan_metrics(empty, S.CATALOG) == (0, 0)
    never_read = Scan(table="sales", columns=None, pushed_predicate=lit(False))
    assert scan_metrics(never_read, S.CATALOG) == (0, 0)


QUERIES = [(q.name, q.plan) for q in S.QUERIES
           if q.name in ("filter_only_column", "comma_join_with_a_cross_product", "two_of_twenty",
                         "contradiction_on_range", "five_way_join_in_a_bad_order")]


def test_run_ablation_measures_every_query_under_every_configuration():
    ms = run_ablation(QUERIES, S.CATALOG, reference, repeat=1)
    configs = default_configs(PASSES)
    assert len(ms) == len(QUERIES) * len(configs)
    assert all(m.same_result for m in ms)
    assert all(m.runtime_ms > 0 for m in ms)
    by = {(m.query, m.config): m for m in ms}
    for name, plan in QUERIES:
        assert by[(name, "no passes")].rows_scanned == scan_metrics(plan, S.CATALOG)[0]
        assert by[(name, "no passes")].result_rows == len(evaluate(plan, S.TABLES).rows)
    # each pass's own effect shows in its column
    assert by[("two_of_twenty", "without column_pruning")].values_read > by[("two_of_twenty", "all passes")].values_read
    assert by[("contradiction_on_range", "all passes")].rows_scanned == 0
    assert by[("contradiction_on_range", "without constant_folding")].rows_scanned == 60
    comma = [by[("comma_join_with_a_cross_product", c.name)].estimated_cost_us for c in configs]
    assert comma[0] == min(comma)


def test_run_ablation_catches_a_wrong_result():
    # a negative control: an executor that loses a row whenever the plan was rewritten
    original = S.filter_only_column()  # every configuration with pushdown or pruning rewrites it

    def lossy(plan):
        rows = evaluate(plan, S.TABLES).rows
        return lambda: rows if plan is original else rows[1:]

    ms = run_ablation([("q", original)], S.CATALOG, lossy, repeat=1)
    assert [m.config for m in ms if not m.same_result] == ["all passes"] + [f"without {p}" for p in PASSES]
    assert next(m for m in ms if m.config == "no passes").same_result


def test_csv_and_summary():
    ms = run_ablation(QUERIES[:2], S.CATALOG, reference, repeat=1,
                      configs=[Config("all passes", tuple(PASSES)), Config("no passes", ())])
    lines = to_csv(ms).splitlines()
    assert lines[0] == ("query,config,runtime_ms,result_rows,rows_scanned,values_read,"
                        "estimated_cost_us,same_result")
    assert len(lines) == 1 + 4
    table = summary_table(ms).splitlines()
    assert [row.split(" | ")[0] for row in table[2:]] == ["| all passes", "| no passes"]


def test_rank_correlation():
    assert rank_correlation([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert rank_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert rank_correlation([1, 1, 2], [5, 5, 9]) == pytest.approx(1.0)  # ties share a rank
    assert rank_correlation([1, 2, 3], [7, 7, 7]) == 0.0


# --------------------------------------------------------------------------
# Pass order
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def study():
    return pass_order_study([(q.name, q.plan) for q in S.QUERIES], S.CATALOG)


def test_study_covers_every_order(study):
    assert len(study) == 24
    assert len({r.order for r in study}) == 24
    assert all(r.unstable == 0 for r in study)  # no order oscillates or hits the cap


def test_default_order_is_cheapest_and_needs_the_fewest_applications_among_the_cheapest(study):
    default = next(r for r in study if r.order == tuple(PASSES))
    assert default.differs == 0
    cheapest = min(r.total_cost_us for r in study)
    assert default.total_cost_us == pytest.approx(cheapest)
    tied = [r for r in study if r.total_cost_us == pytest.approx(cheapest)]
    assert default.applications == min(r.applications for r in tied)


def test_pruning_before_reordering_costs_more(study):
    # pruning's Projects split the join trees that reordering needs whole
    default = next(r for r in study if r.order == tuple(PASSES))
    for r in study:
        if r.order.index("column_pruning") < r.order.index("join_reordering"):
            assert r.total_cost_us > default.total_cost_us * 1.2


def test_committed_pass_order_table_is_current(study):
    committed = (Path(__file__).parent.parent / "docs" / "ablation" / "pass_order.md").read_text()
    assert order_table(study) in committed, "stale: run `python tests/opt_ablation_run.py --order`"
