"""Phase C5: operator fusion. [Project] <- Filter* <- Scan compiles to one pass.

Fusion must never change an answer, so most of this file is differential: every plan in
the corpus and every fuzzed plan runs fused, unfused and interpreted, and all three must
agree. The rest pins down the shape the phase asks for (one pass instead of three) and
checks that the pass really allocates less.
"""
import random
import re
import time
import tracemalloc

import numpy as np
import pytest

from codegen import compile_and_run, generate
from codegen.exprgen import column_refs
from codegen.operators import fusable_chain
from runtime import compare_tables, interpret
from runtime._compat import Aggregate, AggCall, DType, Filter, Join, Project, Sort, UnaryOp
from runtime.table import Column, Table

from codegen_fixtures import (
    CATALOG, JOIN_AGG_PLANS, SAMPLE_PLANS, SINGLE_TABLE_PLANS, TABLES, col, lit, op, scan,
)
from test_codegen_join_aggregate_sort import _PlanFuzzer
from test_codegen_scan_filter_project import _ExprFuzzer

ALL_PLANS = {**SAMPLE_PLANS, **JOIN_AGG_PLANS,
             **{k: (p, False) for k, (p, _) in SINGLE_TABLE_PLANS.items()}}

SECTION = re.compile(r"^\s*# (Scan|Filter|Project|Join|Aggregate|Sort|Limit)\[", re.M)
FUSED = re.compile(r"^\s*# Fused pipeline", re.M)
MASK = re.compile(r"^\s*(keep|pushed)_\d+ = ", re.M)


def src_of(plan, fuse):
    return generate(plan, CATALOG, mode="compiled", fuse=fuse)


def passes(src):
    """Operator passes in the generated code: one per section header."""
    return len(SECTION.findall(src)) + len(FUSED.findall(src))


def assert_three_way(plan, ordered=False, tables=TABLES):
    """interpreter == unfused == fused, including column names, types and qualifiers."""
    expected = interpret(plan, tables)
    results = {}
    for fuse in (False, True):
        src = src_of(plan, fuse)
        results[fuse] = out = compile_and_run(src, tables)
        ok, why = compare_tables(expected, out, ordered=ordered)
        assert ok, f"fuse={fuse}: {why}\n\n{src}"
    layout = [[(c.name, c.dtype, c.table) for c in results[f].columns] for f in (False, True)]
    assert layout[0] == layout[1]


# ------------------------------------------------------------------ fused == unfused == interpreter

@pytest.mark.parametrize("name", sorted(ALL_PLANS))
def test_every_corpus_plan_agrees_fused_and_unfused(name):
    plan, ordered = ALL_PLANS[name]
    assert_three_way(plan, ordered)


class _ChainFuzzer(_ExprFuzzer):
    """Random [Project] <- Filter* <- Scan chains over `sales`: pruned or full scans,
    pushed predicates (which may read columns the scan doesn't output), stacked filters,
    projections of computed and plain columns."""

    COLUMNS = ["id", "region", "amount", "qty", "day"]

    def boolean(self, depth):
        if self.r.random() < 0.15:  # row-at-a-time predicates take a different path
            return op(self.r.choice(["LIKE", "NOT LIKE"]), col("region"),
                      lit(self.r.choice(["E%", "%S", "_U", "%", "X%"]), DType.STRING))
        return super().boolean(depth)

    def valid(self, make, visible):
        for _ in range(50):  # resample until it only reads columns the operator can see
            expr = make()
            if {r.name for r in column_refs(expr)} <= set(visible):
                return expr
        return lit(True, DType.BOOL)

    def chain(self):
        r = self.r
        visible = None if r.random() < 0.4 else r.sample(self.COLUMNS, r.randint(1, 5))
        pushed = self.boolean(2) if r.random() < 0.5 else None
        plan = scan("sales", columns=visible, pred=pushed)
        visible = visible or self.COLUMNS
        for _ in range(r.randint(0, 3)):
            plan = Filter(child=plan, predicate=self.valid(lambda: self.boolean(2), visible))
        if r.random() < 0.7:
            exprs = [(self.valid(lambda: self.num(2), visible), "n"),
                     (self.valid(lambda: self.boolean(1), visible), "b")]
            exprs += [(col(c), c) for c in r.sample(visible, r.randint(0, len(visible)))]
            plan = Project(child=plan, exprs=exprs)
        return plan


@pytest.mark.parametrize("seed", range(300))
def test_fuzz_fusable_chains(seed):
    plan = _ChainFuzzer(seed).chain()
    assert_three_way(plan)
    if fusable_chain(plan) is not None:
        fused = src_of(plan, True)
        assert passes(fused) == 1
        assert len(MASK.findall(fused)) <= 1


@pytest.mark.parametrize("seed", range(150))
def test_fuzz_join_aggregate_pipelines_with_fused_inputs(seed):
    plan, ordered = _PlanFuzzer(seed).pipeline()
    assert_three_way(plan, ordered)


# ------------------------------------------------------------------ the shape the phase asks for

def _canonical_chain():
    """SELECT id, amount * 2 FROM sales WHERE amount > 15 AND region = 'US' AND qty < 6,
    with one predicate pushed into the scan and two left as stacked Filters."""
    return Project(
        child=Filter(child=Filter(child=scan("sales", pred=op(">", col("amount"), lit(15))),
                                  predicate=op("=", col("region"), lit("US", DType.STRING))),
                     predicate=op("<", col("qty"), lit(6))),
        exprs=[(col("id"), "id"), (op("*", col("amount"), lit(2)), "double_amount")])


def test_scan_filter_project_is_one_pass_instead_of_three():
    plan = Project(child=Filter(child=scan("sales"), predicate=op("<", col("qty"), lit(5))),
                   exprs=[(col("id"), "id"), (op("+", col("qty"), lit(1)), "next_qty")])
    unfused, fused = src_of(plan, False), src_of(plan, True)
    assert passes(unfused) == 3 and passes(fused) == 1
    assert SECTION.findall(fused) == []
    assert "Fused pipeline: 3 operators, one pass over sales" in fused
    assert_three_way(plan)


def test_stacked_filters_and_pushed_predicate_build_one_mask():
    plan = _canonical_chain()
    unfused, fused = src_of(plan, False), src_of(plan, True)
    assert len(MASK.findall(unfused)) == 3        # pushed_1, keep_1, keep_2
    assert len(MASK.findall(fused)) == 1          # keep_1 = <pushed> & <filter> & <filter>
    assert passes(unfused) == 4 and passes(fused) == 1
    assert_three_way(plan)


def test_fused_applies_the_mask_once_and_only_to_needed_columns():
    plan = _canonical_chain()
    unfused, fused = src_of(plan, False), src_of(plan, True)
    # unfused drags every column (day, region, qty...) through every Filter
    assert "'day'" in unfused and "'day'" not in fused
    gathers = re.compile(r"\[(keep|pushed)_\d+\]|rows=(keep|pushed)_\d+")
    assert len(gathers.findall(fused)) == 3       # id via rows=, amount values + ok
    assert len(gathers.findall(unfused)) > 20
    # the output's only columns: id is read already filtered, amount is gathered once
    assert "read_column(sales_1, 'id', rows=keep_1)" in fused
    assert fused.count("amount_1[keep_1]") == 1


def test_pushed_predicate_may_read_columns_the_scan_does_not_output():
    plan = Filter(child=scan("sales", columns=["id"], pred=op("=", col("region"),
                                                                lit("EU", DType.STRING))),
                  predicate=op(">", col("id"), lit(2)))
    assert compile_and_run(src_of(plan, True), TABLES).to_rows() == [(5,)]
    assert_three_way(plan)


def test_constant_false_anywhere_in_the_chain_reads_nothing():
    plans = [
        Project(child=Filter(child=scan("sales"), predicate=lit(False, DType.BOOL)),
                exprs=[(op("/", col("qty"), lit(2)), "half")]),
        Filter(child=Filter(child=scan("sales", pred=op(">", col("qty"), lit(1))),
                            predicate=lit(None, DType.BOOL)),
               predicate=op("<", col("qty"), lit(5))),
        Project(child=scan("sales", columns=["day", "id"], pred=lit(False, DType.BOOL)),
                exprs=[(col("day"), "d")]),
    ]
    for plan in plans:
        src = src_of(plan, True)
        assert "as_table(" not in src and "read_column(" not in src, src
        assert compile_and_run(src, TABLES).num_rows == 0
        assert_three_way(plan)


def test_project_over_scan_reads_only_referenced_columns():
    plan = Project(child=scan("sales"), exprs=[(op("*", col("qty"), lit(3)), "q3")])
    fused = src_of(plan, True)
    assert passes(fused) == 1 and fused.count("read_column(") == 1
    assert src_of(plan, False).count("read_column(") == 5
    assert_three_way(plan)


# ------------------------------------------------------------------ LIKE: a loop per row, so it goes last

def _like(pattern, negate=False):
    return op("NOT LIKE" if negate else "LIKE", col("region"), lit(pattern, DType.STRING))


def test_like_only_runs_on_rows_the_vectorized_predicates_kept():
    # LIKE sits BELOW the cheap filter in the plan, and is still evaluated after it
    plan = Filter(child=Filter(child=scan("sales"), predicate=_like("%S")),
                  predicate=op("<", col("qty"), lit(4)))
    src = src_of(plan, True)
    assert "alive_1 = np.flatnonzero(keep_1)" in src
    assert "read_column(sales_1, 'region', rows=alive_1)" in src
    assert "keep_1[alive_1] = " in src
    assert src.index("(qty_1 < 4)") < src.index("like(")
    assert_three_way(plan)
    assert compile_and_run(src, TABLES).column("id").to_pylist() == [1, 3]


def test_like_alone_or_twice():
    once = Filter(child=scan("sales"), predicate=_like("E%"))
    assert "alive_" not in src_of(once, True)          # nothing cheap to narrow with first
    twice = Filter(child=Filter(child=scan("sales"), predicate=_like("E%")),
                   predicate=_like("%U", negate=True))
    src = src_of(twice, True)
    assert src.count("np.flatnonzero(keep_1)") == 1     # the second one runs on survivors
    for plan in (once, twice):
        assert_three_way(plan)


def test_in_place_mask_update_never_writes_into_an_input_column():
    # IS NOT NULL compiles to the column's own validity array; keep must be a copy of it
    plan = Project(child=Filter(child=Filter(child=scan("sales"),
                                             predicate=UnaryOp(op="IS NOT NULL", operand=col("amount"))),
                                predicate=_like("%S")),
                   exprs=[(col("id"), "id"), (col("amount"), "amount")])
    src = src_of(plan, True)
    assert ".copy()" in src
    before = TABLES["sales"].column("amount").valid.copy()
    for _ in range(2):
        assert_three_way(plan)
    assert (TABLES["sales"].column("amount").valid == before).all()


def test_fused_like_is_not_slower_than_unfused_on_a_selective_chain():
    n = 100_000
    rng = np.random.default_rng(3)
    table = Table([Column("k", DType.INT, rng.integers(0, 1000, n), None, "t"),
                   Column("s", DType.STRING, np.array([f"name{i % 997}" for i in range(n)], dtype=object),
                          None, "t")])

    class Cat:
        def schema(self, t):
            return [("k", DType.INT), ("s", DType.STRING)]
    plan = Filter(child=Filter(child=scan("t"), predicate=op("<", col("k"), lit(10))),
                  predicate=op("LIKE", col("s"), lit("name1%", DType.STRING)))
    timings = {}
    for fuse in (False, True):
        src = generate(plan, Cat(), fuse=fuse)
        compile_and_run(src, {"t": table})
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            compile_and_run(src, {"t": table})
            best = min(best, time.perf_counter() - t0)
        timings[fuse] = best
    # LIKE sees ~1% of the rows either way; before the fix fused was ~11x slower
    assert timings[True] < 2 * timings[False]


# ------------------------------------------------------------------ what does NOT fuse

def test_chains_that_do_not_start_at_a_scan_compile_operator_by_operator():
    having = Filter(child=Aggregate(child=scan("sales"), group_keys=[col("region")],
                                    aggs=[(AggCall("sum", col("qty")), "q")]),
                    predicate=op(">", col("q"), lit(3)))
    filter_over_project = Filter(child=Project(child=scan("sales"),
                                               exprs=[(op("+", col("qty"), lit(1)), "q1")]),
                                 predicate=op(">", col("q1"), lit(3)))
    project_over_join = Project(
        child=Join(left=scan("orders"), right=scan("customer"),
                   condition=op("=", col("cust_id", "orders"), col("id", "customer")),
                   kind="inner"),
        exprs=[(col("name"), "name")])
    for plan in (having, filter_over_project, project_over_join):
        assert fusable_chain(plan) is None
        assert_three_way(plan)
    # ...but a fusable chain underneath still fuses on its own
    src = src_of(filter_over_project, True)
    assert "# Filter[" in src and "Fused pipeline: 2 operators" in src


def test_scan_alone_is_not_a_chain():
    assert fusable_chain(scan("sales", pred=op(">", col("qty"), lit(2)))) is None


def test_fusion_happens_inside_bigger_plans():
    plan = Sort(child=Aggregate(
        child=Filter(child=scan("sales"), predicate=op(">", col("qty"), lit(1))),
        group_keys=[col("region")], aggs=[(AggCall("count", None), "n")]),
        keys=[(col("n"), True), (col("region"), False)])
    src = src_of(plan, True)
    assert "Fused pipeline: 2 operators" in src and "# Aggregate[" in src
    assert_three_way(plan, ordered=True)


def test_header_says_whether_fusion_is_on():
    plan = _canonical_chain()
    assert "operator fusion on" in src_of(plan, True)
    assert "operator fusion off" in src_of(plan, False)
    assert "operator fusion on" in generate(plan, CATALOG)  # the default


# ------------------------------------------------------------------ the point of it: less memory

def _wide_table(n=50_000, width=8, seed=0):
    rng = np.random.default_rng(seed)
    cols = [Column(f"c{i}", DType.INT, rng.integers(0, 1000, n), None, "wide") for i in range(width)]
    return Table(cols)


class _WideCatalog:
    def schema(self, table):
        return [(f"c{i}", DType.INT) for i in range(8)]


def _peak(src, tables):
    compile_and_run(src, tables)  # warm-up: imports, compiled code objects
    tracemalloc.start()
    try:
        compile_and_run(src, tables)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_fused_pipeline_allocates_less_than_operator_at_a_time():
    tables = {"wide": _wide_table()}
    plan = Project(child=Filter(child=Filter(child=scan("wide"), predicate=op("<", col("c0"), lit(500))),
                                predicate=op(">", col("c1"), lit(100))),
                   exprs=[(op("+", col("c2"), col("c3")), "s")])
    fused = generate(plan, _WideCatalog(), fuse=True)
    unfused = generate(plan, _WideCatalog(), fuse=False)
    ok, why = compare_tables(compile_and_run(unfused, tables), compile_and_run(fused, tables))
    assert ok, why
    assert _peak(fused, tables) < 0.6 * _peak(unfused, tables)
