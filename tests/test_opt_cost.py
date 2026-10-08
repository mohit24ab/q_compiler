"""The cost model (optimizer/cost.py).

Component formulas are checked with unit weights, so each expected cost is
a row count you can work out by hand. Then the model is checked against the
optimizer: no rewrite may make a suite query more expensive, and the model
must prefer the join order that keeps intermediate results small.
"""

import dataclasses
import math
import warnings

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
import optimizer
from ir.dtype import DType
from ir.expr import Literal
from ir.nodes import Aggregate, Filter, Limit, Project, Scan, Sort
from opt_query_suite import agg, col, join, lit, op, scan
from optimizer.cost import COMPONENTS, DEFAULT_WEIGHTS, CostModel, CostWeights, explain, plan_cost
from optimizer.stats import estimate_rows
from test_opt_stats import CAT

ONES = CostWeights(**{f.name: 1.0 for f in dataclasses.fields(CostWeights)})
NULL = Literal(value=None, dtype=DType.BOOL)


def cost(plan, weights=ONES, catalog=CAT):
    return CostModel(catalog, weights).cost(plan)


def own(node, weights=ONES, catalog=CAT):
    return CostModel(catalog, weights).node_cost(node).components


# --------------------------------------------------------------------------
# Components
# --------------------------------------------------------------------------


def test_scan_reads_every_column_of_every_row():
    # t has 4 fixed-width columns and 1 string: 100 rows * 5 columns
    assert own(scan("t")) == {"scan_io": 500.0}


def test_strings_cost_more_to_read():
    weights = dataclasses.replace(ONES, read_string=10.0)
    strings = Scan(table="t", columns=["s"], pushed_predicate=None)
    ints = Scan(table="t", columns=["x"], pushed_predicate=None)
    assert own(strings, weights)["scan_io"] == 10 * own(ints, weights)["scan_io"]


def test_column_pruning_reduces_scan_io():
    narrow = Scan(table="t", columns=["x"], pushed_predicate=None)
    assert own(narrow)["scan_io"] == 100.0


def test_pushed_predicate_columns_are_read_too():
    pushed = Scan(table="t", columns=["x"], pushed_predicate=op("=", col("k"), lit(3)))
    assert own(pushed) == {"scan_io": 200.0, "predicate": 100.0, "gather": 100.0}


def test_filter_evaluates_its_predicate_and_gathers_every_column():
    plan = Filter(child=scan("t"), predicate=op("AND", op("<", col("x"), lit(5)), op("=", col("k"), lit(1))))
    assert own(plan) == {"predicate": 100 * 3, "gather": 100 * 5}


def test_project_pays_only_for_computed_columns():
    plan = Project(child=scan("t"), exprs=[(col("x"), "x"), (op("*", col("x"), col("k")), "xk")])
    assert own(plan) == {"predicate": 100.0}


def test_hash_join_builds_on_the_smaller_input_and_probes_with_the_larger():
    plan = join(scan("t"), scan("u"), op("=", col("k", "t"), col("k", "u")))
    out = estimate_rows(plan, CAT)  # 100 rows; the candidates are the same rows
    assert own(plan) == {
        "join_build": 40.0, "join_probe": 100.0, "join_output": out,
        "predicate": 0.0, "gather": out * 7,
    }


def test_join_orientation_does_not_change_the_cost():
    # codegen picks the build side at run time
    cond = op("=", col("k", "t"), col("k", "u"))
    assert cost(join(scan("t"), scan("u"), cond)).total == cost(join(scan("u"), scan("t"), cond)).total


def test_residual_conditions_are_checked_on_candidate_pairs():
    cond = op("AND", op("=", col("k", "t"), col("k", "u")), op("<", col("x"), col("g")))
    assert own(join(scan("t"), scan("u"), cond))["predicate"] == pytest.approx(100.0)


def test_join_without_equality_keys_is_a_nested_loop():
    plan = join(scan("t"), scan("u"), op("<", col("x"), col("g")))
    components = own(plan)
    assert components["nested_loop"] == 100 * 40 * (1 + 1)
    assert "join_build" not in components


class _BigCatalog:
    """Two tables whose statistics say 100,000 and 10,000 rows, 10,000 keys each."""

    def schema(self, table):
        return [("k", DType.INT), ("v", DType.INT)]

    def row_count(self, table):
        return {"big": 100_000, "small": 10_000}[table]

    def stats(self, table, column):
        return S.ColumnStats(ndv=10_000, min=0, max=9_999, null_count=0)


def test_hash_join_beats_a_nested_loop_on_large_inputs():
    cat = _BigCatalog()
    equi = join(scan("big"), scan("small"), op("=", col("k", "big"), col("k", "small")))
    pairs = join(scan("big"), scan("small"), op("<", col("k", "big"), col("k", "small")))
    assert plan_cost(equi, cat) < plan_cost(pairs, cat) / 10  # ~88 ms against ~7 s


def test_nested_loop_beats_the_hash_join_on_small_inputs():
    # codegen's nested loop is vectorized, its hash join is a Python loop: on
    # 100 x 40 rows, 4,000 vectorized pairs are cheaper than 140 dict operations
    equi = join(scan("t"), scan("u"), op("=", col("k", "t"), col("k", "u")))
    pairs = join(scan("t"), scan("u"), op("<", col("k", "t"), col("k", "u")))
    assert plan_cost(pairs, CAT) < plan_cost(equi, CAT)


def test_grouped_aggregate_pays_per_row_per_aggregate():
    plan = Aggregate(child=scan("t"), group_keys=[col("k")], aggs=[(agg("sum", col("x")), "s"), (agg("count"), "n")])
    assert own(plan) == {"predicate": 0.0, "aggregation": 100 * (1 + 2), "gather": 20.0}


def test_global_aggregate_is_a_vectorized_reduction():
    weights = dataclasses.replace(ONES, reduce=0.5)
    plan = Aggregate(child=scan("t"), group_keys=[], aggs=[(agg("sum", col("x")), "s")])
    assert own(plan, weights) == {"predicate": 0.0, "aggregation": 50.0}


def test_sort_is_n_log_n_per_key():
    plan = Sort(child=scan("u"), keys=[(col("k"), False), (col("g"), True)])
    assert own(plan) == {"sort": 40 * math.log2(40) * 2, "gather": 40 * 2}


def test_limit_is_free():
    assert own(Limit(child=scan("t"), n=5)) == {}


@pytest.mark.parametrize("plan", [
    Filter(child=join(scan("t"), scan("u"), op("=", col("k", "t"), col("k", "u"))), predicate=lit(False)),
    Filter(child=scan("t"), predicate=NULL),
    Scan(table="t", columns=None, pushed_predicate=lit(False)),
    join(scan("t"), scan("u"), lit(False)),
], ids=["false-filter", "null-filter", "false-pushed", "false-inner-join"])
def test_known_empty_results_cost_nothing(plan):
    # codegen returns an empty table without running anything below these
    assert cost(plan).total == 0.0


def test_left_join_that_cannot_match_still_runs_both_sides():
    plan = join(scan("u"), scan("t"), lit(False), kind="left")
    assert cost(plan).total >= cost(scan("u")).total + cost(scan("t")).total


def test_total_is_the_sum_of_the_nodes():
    plan = S.QUERIES[0].plan
    model = CostModel(S.CATALOG)
    nodes = []

    def walk(n):
        nodes.append(n)
        for c in n.children:
            walk(c)

    walk(plan)
    assert model.cost(plan).total == pytest.approx(sum(model.node_cost(n).total for n in nodes))
    assert set(model.cost(plan).components) <= set(COMPONENTS)


def test_cost_does_not_mutate_the_plan():
    plan = S.QUERIES[0].plan
    before = repr(plan)
    plan_cost(plan, S.CATALOG)
    assert repr(plan) == before


def test_explain_prints_one_line_per_node_and_the_total():
    plan = join(scan("t"), scan("u"), op("=", col("k", "t"), col("k", "u")))
    lines = explain(plan, CAT).splitlines()
    assert len(lines) == 4
    assert lines[0].startswith("Join[") and "rows≈" in lines[0]
    assert lines[-1].startswith("total ") and "join_build" in lines[-1]


# --------------------------------------------------------------------------
# The cost model against the optimizer
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_optimizing_never_makes_a_query_more_expensive(query):
    # The end result only. A single pass can raise the cost on the way: pushdown
    # moves a filter into a Scan that still reads all 20 columns, and only the
    # column pruning that follows makes the move pay.
    plan = query.plan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimized, _ = optimizer.optimize(plan, S.CATALOG)
    model = CostModel(S.CATALOG)
    assert model.cost(optimized).total <= model.cost(plan).total * (1 + 1e-9)


def test_cost_prefers_joining_the_filtered_table_first():
    """A preview of B6: the order that keeps intermediate results small must be cheaper."""
    orders = Scan(table="orders", columns=["o_id", "o_custkey"], pushed_predicate=None)
    customer = Scan(table="customer", columns=["c_id", "c_nationkey"], pushed_predicate=None)
    nation = Scan(table="nation", columns=["n_id"], pushed_predicate=op("=", col("n_name"), lit("JAPAN")))
    oc = op("=", col("o_custkey"), col("c_id"))
    cn = op("=", col("c_nationkey"), col("n_id"))
    late = join(join(orders, customer, oc), nation, cn)   # (orders ⋈ customer) ⋈ nation
    early = join(orders, join(customer, nation, cn), oc)  # orders ⋈ (customer ⋈ nation)
    # the same result rows either way
    assert estimate_rows(late, S.CATALOG) == pytest.approx(estimate_rows(early, S.CATALOG))
    assert plan_cost(early, S.CATALOG) < plan_cost(late, S.CATALOG)


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibration_measures_every_weight():
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    from optimizer.calibration import measure

    measured = measure(n=4_000, repeat=1)
    assert set(measured) == {f.name for f in dataclasses.fields(CostWeights)}
    assert all(v > 0 for v in measured.values())


def test_default_weights_keep_the_measured_ordering():
    w = DEFAULT_WEIGHTS
    # the Python-level loops dominate the vectorized work
    assert w.hash_build > w.hash_probe > w.gather_mask > w.predicate
    assert w.group > w.aggregate > w.reduce
    assert w.read_string > w.read_fixed
