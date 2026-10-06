"""Phase C3: Scan / Filter / Project compile to vectorized numpy and match the interpreter."""
import ast
import random

import pyarrow as pa
import pytest

from codegen import CodegenError, compile_and_run, generate
from runtime import Table, compare_tables, interpret
from runtime._compat import DType, Filter, Project, UnaryOp

from codegen_fixtures import (
    CATALOG, SAMPLE_PLANS, SINGLE_TABLE_PLANS, TABLES, col, lit, op, orders_join_customer, scan,
)

SINGLE_IDS = sorted(SINGLE_TABLE_PLANS)


def compiled(plan):
    return generate(plan, CATALOG, mode="compiled")


def assert_matches_interpreter(plan, src=None, ordered=False):
    src = src or compiled(plan)
    ok, why = compare_tables(interpret(plan, TABLES), compile_and_run(src, TABLES), ordered=ordered)
    assert ok, f"{why}\n\n{src}"


# ------------------------------------------------------------------ the phase requirements

@pytest.mark.parametrize("name", SINGLE_IDS)
def test_single_table_query_matches_interpreter(name):
    assert_matches_interpreter(SINGLE_TABLE_PLANS[name][0])


@pytest.mark.parametrize("name", SINGLE_IDS)
def test_single_table_source_has_no_interpreter_calls(name):
    src = compiled(SINGLE_TABLE_PLANS[name][0])
    assert "interpret" not in src
    assert "import numpy as np" in src
    calls = {n.func.id for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert calls <= {"as_table", "read_column", "build_table", "like", "int", "float"}


@pytest.mark.parametrize("name", sorted(SAMPLE_PLANS))
def test_auto_mode_matches_interpreter_on_whole_corpus(name):
    plan, ordered = SAMPLE_PLANS[name]
    assert_matches_interpreter(plan, generate(plan, CATALOG), ordered=ordered)


def test_every_corpus_plan_compiles_since_phase_c4():
    for name, (plan, _) in SAMPLE_PLANS.items():
        assert "mode: compiled" in generate(plan, CATALOG), name
    assert "mode: passthrough" in generate(orders_join_customer(), CATALOG, mode="passthrough")


def test_passthrough_mode_still_available_for_comparisons():
    plan = SINGLE_TABLE_PLANS["arith_precedence"][0]
    src = generate(plan, CATALOG, mode="passthrough")
    assert "return interpret(" in src
    assert_matches_interpreter(plan, src)


# ------------------------------------------------------------------ Scan reads only what it needs

def test_scan_reads_only_requested_columns():
    src = compiled(scan("sales", columns=["id", "qty"]))
    assert "'id'" in src and "'qty'" in src
    assert "'amount'" not in src and "'region'" not in src


def test_pushed_predicate_reads_predicate_columns_first_then_masks_the_rest():
    src = compiled(SINGLE_TABLE_PLANS["pushed_on_unselected_column"][0])
    lines = [ln.strip() for ln in src.splitlines()]
    read_region = next(i for i, ln in enumerate(lines) if "read_column(sales_1, 'region')" in ln)
    mask = next(i for i, ln in enumerate(lines) if ln.startswith("pushed_1 ="))
    read_id = next(i for i, ln in enumerate(lines) if "'id', rows=pushed_1" in ln)
    assert read_region < mask < read_id
    assert "'region'" not in src.split("return build_table")[1]  # not in the output


def test_filter_builds_one_mask_and_applies_it_to_every_column():
    plan = Filter(child=scan("sales", columns=["id", "qty", "amount"]),
                  predicate=op(">", col("qty"), lit(2)))
    src = compiled(plan)
    assert src.count("keep_1 =") == 1
    assert src.count("[keep_1]") == 6  # 3 columns x (values, ok)


def test_project_of_plain_column_copies_nothing():
    src = compiled(Project(child=scan("sales", columns=["id"]), exprs=[(col("id"), "renamed")]))
    body = src.split("def run(tables):")[1]
    assert body.count(" = ") == 2  # the as_table line and the read_column line, nothing else
    assert "('renamed', DType.INT, None, id_1, id_1_ok)" in src


def test_non_null_computed_column_carries_no_mask():
    src = compiled(Project(child=scan("sales"), exprs=[(op("+", col("id"), lit(1)), "next_id")]))
    assert "('next_id', DType.INT, None, next_id_1, id_1_ok)" in src  # reuses id's mask


# ------------------------------------------------------------------ inputs and errors

def test_accepts_pyarrow_input_tables():
    plan = Filter(child=scan("t"), predicate=op(">", col("x"), lit(1)))

    class Cat:
        def schema(self, table):
            return [("x", DType.INT)]

    out = compile_and_run(generate(plan, Cat()), {"t": pa.table({"x": [3, None, 1, 2]})})
    assert out.to_pydict() == {"x": [3, 2]}


def test_scan_schema_comes_from_table_schema_when_no_catalog():
    plan = Filter(child=scan("sales"), predicate=op(">", col("qty"), lit(4)))
    with pytest.raises(CodegenError, match="pass a catalog"):
        generate(plan, None, mode="compiled")
    from runtime._compat import Scan
    with_schema = Scan(table="sales", columns=None, pushed_predicate=None,
                       table_schema=TABLES["sales"].schema)
    out = compile_and_run(generate(Filter(child=with_schema, predicate=plan.predicate)), TABLES)
    assert out.column("id").to_pylist() == [5, 6]


def test_unknown_scan_column_fails_at_generation_time():
    with pytest.raises(CodegenError, match="unknown columns"):
        compiled(scan("sales", columns=["nope"]))


def test_empty_result_keeps_schema():
    out = compile_and_run(compiled(SINGLE_TABLE_PLANS["constant_false_filter"][0]), TABLES)
    assert out.num_rows == 0
    assert out.schema == TABLES["sales"].schema


# ------------------------------------------------------------------ randomized differential test

class _ExprFuzzer:
    """Random, well-typed expressions over `sales`, biased toward the nasty cases:
    NULL columns, NULL literals, zero divisors, negative modulo, nested Kleene logic."""

    def __init__(self, seed):
        self.r = random.Random(seed)

    def num(self, depth):
        r = self.r
        if depth == 0 or r.random() < 0.3:
            return r.choice([
                lambda: col("id"), lambda: col("qty"), lambda: col("amount"),
                lambda: lit(r.randint(-3, 4)), lambda: lit(r.choice([0.0, 2.5, -1.5]), DType.FLOAT),
                lambda: lit(None, DType.INT),
            ])()
        if r.random() < 0.1:
            return UnaryOp(op="-", operand=self.num(depth - 1))
        return op(r.choice(["+", "-", "*", "/", "%"]), self.num(depth - 1), self.num(depth - 1))

    def boolean(self, depth):
        r = self.r
        if depth == 0 or r.random() < 0.3:
            return r.choice([
                lambda: op(r.choice(["=", "<>", "<", "<=", ">", ">="]), self.num(1), self.num(1)),
                lambda: op(r.choice(["=", "<>", "<"]), col("region"),
                           lit(r.choice(["EU", "US", "F"]), DType.STRING)),
                lambda: op(r.choice([">=", "<"]), col("day"),
                           lit(r.choice(["2024-03-01", "2024-05-15"]), DType.DATE)),
                lambda: UnaryOp(op=r.choice(["IS NULL", "IS NOT NULL"]), operand=self.num(1)),
                lambda: lit(r.choice([True, False, None]), DType.BOOL),
            ])()
        if r.random() < 0.2:
            return UnaryOp(op="NOT", operand=self.boolean(depth - 1))
        return op(r.choice(["AND", "OR"]), self.boolean(depth - 1), self.boolean(depth - 1))


@pytest.mark.parametrize("seed", range(200))
def test_fuzz_project_and_filter_match_interpreter(seed):
    f = _ExprFuzzer(seed)
    num, cond = f.num(3), f.boolean(3)
    project = Project(child=scan("sales"), exprs=[(num, "n"), (cond, "b"), (col("id"), "id")])
    assert_matches_interpreter(project)
    assert_matches_interpreter(Filter(child=scan("sales"), predicate=cond))


# ------------------------------------------------------------------ constant FALSE / NULL predicates
# Asked by Person B for constant folding (B4): a Filter whose predicate folds to FALSE
# returns an empty table with the child's schema, and the child is never executed.

def _never_true_plans():
    computed = Project(child=scan("sales"), exprs=[
        (col("id"), "id"), (op("/", col("qty"), lit(2)), "half"), (lit("x", DType.STRING), "tag")])
    return {
        "filter_false_over_scan": Filter(child=scan("sales"), predicate=lit(False, DType.BOOL)),
        "filter_null_over_computed_project": Filter(child=computed, predicate=lit(None, DType.BOOL)),
        "scan_pushed_false": scan("sales", columns=["day", "id"], pred=lit(False, DType.BOOL)),
        "false_below_project_and_filter": Project(
            child=Filter(child=Filter(child=scan("orders"), predicate=lit(False, DType.BOOL)),
                         predicate=op(">", col("total"), lit(1))),
            exprs=[(op("*", col("total"), lit(2)), "dbl")]),
    }


@pytest.mark.parametrize("name", sorted(_never_true_plans()))
def test_never_true_predicate_returns_empty_table_with_childs_schema(name):
    plan = _never_true_plans()[name]
    expected = interpret(plan, TABLES)
    actual = compile_and_run(compiled(plan), TABLES)
    assert actual.num_rows == 0
    assert actual.schema == expected.schema
    assert actual.qualified_names() == expected.qualified_names()


@pytest.mark.parametrize("name", sorted(_never_true_plans()))
def test_never_true_predicate_does_not_run_the_child(name):
    src = compiled(_never_true_plans()[name])
    assert "as_table(" not in src and "read_column(" not in src
    assert "np.empty(0" in src
    # it even works when the input table is missing entirely
    compile_and_run(src, {})
