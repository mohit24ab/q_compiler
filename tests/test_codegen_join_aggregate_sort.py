"""Phase C4: Join / Aggregate / Sort / Limit compile to real code and match the interpreter."""
import random

import pytest

from codegen import compile_and_run, generate
from runtime import compare_tables, interpret
from runtime._compat import AggCall, Aggregate, DType, Filter, Join, Limit, Project, Sort, UnaryOp

from codegen_fixtures import (
    CATALOG, JOIN_AGG_PLANS, SAMPLE_PLANS, SINGLE_TABLE_PLANS, TABLES, col, lit, op, scan,
)

ALL_PLANS = {**{k: v for k, v in SAMPLE_PLANS.items()}, **JOIN_AGG_PLANS,
             **{k: (p, False) for k, (p, _) in SINGLE_TABLE_PLANS.items()}}


def compiled(plan):
    return generate(plan, CATALOG, mode="compiled")


def assert_matches_interpreter(plan, ordered):
    src = compiled(plan)
    ok, why = compare_tables(interpret(plan, TABLES), compile_and_run(src, TABLES), ordered=ordered)
    assert ok, f"{why}\n\n{src}"
    return src


# ------------------------------------------------------------------ full differential suite

@pytest.mark.parametrize("name", sorted(ALL_PLANS))
def test_every_plan_compiles_with_no_interpreter_calls_and_matches(name):
    plan, ordered = ALL_PLANS[name]
    src = assert_matches_interpreter(plan, ordered)
    assert "interpret" not in src


# ------------------------------------------------------------------ hand-computed answers
# The differential tests above can't catch a bug shared by both engines; these can.

def run(name):
    return compile_and_run(compiled(JOIN_AGG_PLANS[name][0]), TABLES)


def test_three_way_join_rows():
    # orders 100,102 -> ann; 101 -> bob; 103 has NULL cust_id. payments: 100 twice, 101 once.
    out = run("three_way_join")
    pairs = sorted((r[0], r[-2]) for r in out.to_rows())   # (orders.id, payments.amt)
    assert pairs == [(100, 4), (100, 5), (101, 7)]


def test_multi_key_join_and_residual():
    assert sorted(r[0] for r in run("multi_key_join").to_rows()) == [100, 101, 103]
    assert [r[0] for r in run("join_residual_inner").to_rows()] == [100]   # amt 4 < total 5


def test_left_join_pads_rows_whose_only_matches_fail_the_residual():
    out = run("join_residual_left_pads_when_residual_fails")
    by_order = {r[0]: r[3:] for r in out.to_rows()}
    assert by_order[100] == (100, 4, "cash")
    assert by_order[101] == (None, None, None)      # key matched, residual 7 < 7 failed
    assert by_order[102] == (None, None, None)      # no key match at all
    assert out.num_rows == 4


def test_aggregate_over_left_join_counts_padding_correctly():
    rows = {r[0]: r[1:] for r in run("aggregate_over_left_join_counts").to_rows()}
    assert rows == {100: (2, 2, 9, "cash"), 101: (1, 1, 7, "card"),
                    102: (1, 0, None, None), 103: (1, 1, 11, None)}


def test_global_aggregates_with_and_without_rows():
    assert run("global_aggregates_all_kinds").to_rows() == [
        (6, 5, 21, 3.5, "AP", __import__("datetime").date(2024, 6, 1), 170.0)]
    assert run("global_aggregate_all_null_input").to_rows() == [(None, None, None, 0)]
    assert run("global_aggregate_over_false_filter").to_rows() == [(0, None)]


def test_multi_key_group_by_with_null_key_and_avg():
    rows = {(r[0], r[1]): r[2:] for r in run("multi_key_group_by").to_rows()}
    assert rows[("US", False)][:3] == (2, 4, 20.0)     # ids 1,3: qty 1+3, avg(10,30)
    assert rows[("AP", True)][2] is None               # avg of only-NULL amount
    assert rows[(None, True)][:2] == (1, 6)            # NULL region is its own group


def test_sort_desc_nulls_last_and_stable_ties():
    ids = run("sort_mixed_directions_with_nulls").column("id").to_pylist()
    assert ids == [1, 3, 2, 5, 4, 6]                   # US, US, EU, EU, AP, NULL
    ties = run("sort_on_computed_expression_stable_ties").column("id").to_pylist()
    assert ties == [3, 6, 1, 4, 2, 5]                  # qty%3 = 0,0,1,1,2,2 in input order


def test_limit_edges():
    assert run("limit_zero_and_overflow").num_rows == 0
    assert run("join_limit_without_sort_keeps_interpreter_order").column("id").to_pylist() == \
        [100, 100, 101]


# ------------------------------------------------------------------ shape of the emitted code

def test_hash_join_emits_build_and_probe_as_code():
    src = compiled(JOIN_AGG_PLANS["three_way_join"][0])
    assert src.count("hash_table_") >= 4 and ".setdefault(key, []).append(i)" in src
    assert "build_left_1 = len(left_keys_1) <= len(right_keys_1)" in src  # smaller side builds
    assert "np.repeat" not in src                                         # no cross product


def test_join_without_equality_keys_falls_back_to_all_pairs():
    src = compiled(JOIN_AGG_PLANS["non_equi_only_join"][0])
    assert "np.repeat" in src and "hash_table" not in src


def test_avg_is_sum_and_count_divided_at_the_end():
    src = compiled(JOIN_AGG_PLANS["multi_key_group_by"][0])
    assert "np.add.at(a_sum_1" in src and "a_1 = a_sum_1 / np.where(a_count_1 == 0, 1, a_count_1)" in src


def test_global_aggregate_has_its_own_simpler_path():
    src = compiled(JOIN_AGG_PLANS["global_aggregates_all_kinds"][0])
    assert "no GROUP BY: one output row" in src
    assert "gid" not in src and "groups_" not in src


def test_sort_is_one_lexsort():
    src = compiled(JOIN_AGG_PLANS["sort_mixed_directions_with_nulls"][0])
    assert src.count("np.lexsort(") == 1


def test_inner_join_with_false_condition_runs_neither_side():
    plan = Join(left=scan("orders"), right=scan("payments"), condition=lit(False, DType.BOOL),
                kind="inner")
    src = compiled(plan)
    assert "as_table(" not in src
    assert compile_and_run(src, {}).num_rows == 0


# ------------------------------------------------------------------ randomized plans

class _PlanFuzzer:
    """Random join -> filter -> aggregate -> having -> sort -> limit pipelines."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def join(self):
        r = self.r
        kind = r.choice(["inner", "left"])
        shape = r.choice(["key", "two_keys", "key_residual", "residual_only", "true"])
        key = op("=", col("id", "orders"), col("order_id", "payments"))
        cond = {
            "key": key,
            "two_keys": op("AND", key, op("=", col("total", "orders"), col("amt", "payments"))),
            "key_residual": op("AND", key, op(r.choice(["<", ">=", "<>"]),
                                              col("amt", "payments"), col("total", "orders"))),
            "residual_only": op(r.choice(["<", ">"]), col("amt", "payments"), col("total", "orders")),
            "true": lit(True, DType.BOOL),
        }[shape]
        left, right = scan("orders"), scan("payments")
        if r.random() < 0.3:
            left = Filter(child=left, predicate=op(">", col("total"), lit(r.choice([4, 8, 50]))))
        return Join(left=left, right=right, condition=cond, kind=kind)

    def pipeline(self):
        r = self.r
        plan = self.join()
        if r.random() < 0.4:
            plan = Filter(child=plan, predicate=UnaryOp(op=r.choice(["IS NULL", "IS NOT NULL"]),
                                                        operand=col("method", "payments")))
        ordered = False
        if r.random() < 0.7:
            keys = r.sample([col("method", "payments"), col("cust_id", "orders"),
                             op(">", col("total", "orders"), lit(6))], r.randint(0, 2))
            aggs = [(AggCall(f, a), f"a{i}") for i, (f, a) in enumerate(r.sample([
                ("count", None), ("count", col("amt", "payments")), ("sum", col("amt", "payments")),
                ("avg", col("total", "orders")), ("min", col("method", "payments")),
                ("max", col("amt", "payments"))], r.randint(1, 4)))]
            plan = Aggregate(child=plan, group_keys=keys, aggs=aggs)
            if r.random() < 0.3:  # HAVING on an aggregate the Aggregate actually computes
                plan = Filter(child=plan, predicate=UnaryOp(op="IS NOT NULL", operand=aggs[0][0]))
            sort_cols = [(col(a), r.random() < 0.5) for _, a in aggs]
            sort_cols += [(k, r.random() < 0.5) for k in keys]
        else:
            sort_cols = [(col("id", "orders"), r.random() < 0.5), (col("amt", "payments"), True),
                         (col("method", "payments"), False)]
        if r.random() < 0.6:
            plan = Sort(child=plan, keys=sort_cols)
            ordered = True
            if r.random() < 0.5:
                plan = Limit(child=plan, n=r.randint(0, 5))
        return plan, ordered


@pytest.mark.parametrize("seed", range(150))
def test_fuzz_join_aggregate_sort_pipelines_match_interpreter(seed):
    plan, ordered = _PlanFuzzer(seed).pipeline()
    assert_matches_interpreter(plan, ordered)
