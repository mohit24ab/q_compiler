"""Phase C6: the benchmark runner and the alias bridge it binds queries through."""
import csv
import types

import pytest

from bench import runner
from bench.aliases import AliasError, alias_map, resolve_aliases
from runtime import interpret
from runtime._compat import Filter, Join, Limit, format_plan

from codegen_fixtures import CATALOG, TABLES, col, op, scan


@pytest.fixture(scope="module")
def tiny():
    return runner.load_data("tiny")


# ------------------------------------------------------------------ alias bridge

def _refs(plan):
    from bench.aliases import _refs as refs
    return list(refs(plan))


def test_binder_alias_qualifiers_become_table_names(tiny):
    catalog, tables = tiny
    from frontend.binder import parse_and_bind
    sql = runner.load_queries(["q07"])["q07"]
    bound = parse_and_bind(sql, catalog)
    assert {r.table for r in _refs(bound)} == {"o", "c"}          # what the binder emits today
    assert alias_map(bound, catalog) == {"c": "customer", "o": "orders"}
    fixed = resolve_aliases(bound, catalog)
    assert {r.table for r in _refs(fixed)} == {"orders", "customer"}
    assert interpret(fixed, tables).num_rows > 0                  # and now it runs


def test_plans_already_using_table_names_are_returned_unchanged():
    plan = Join(left=scan("orders"), right=scan("customer"), kind="inner",
                condition=op("=", col("cust_id", "orders"), col("id", "customer")))
    assert resolve_aliases(plan, CATALOG) is plan


def test_alias_that_fits_several_tables_or_none_is_refused():
    both_have_id = Filter(child=Join(left=scan("orders"), right=scan("customer"), kind="inner",
                                     condition=None),
                          predicate=op(">", col("id", "x"), col("id", "orders")))
    with pytest.raises(AliasError, match="'x'.*fit \\['customer', 'orders'\\]"):
        resolve_aliases(both_have_id, CATALOG)
    nowhere = Filter(child=scan("orders"), predicate=op(">", col("nope", "x"), col("id")))
    with pytest.raises(AliasError, match="no scanned table"):
        resolve_aliases(nowhere, CATALOG)


def test_rewrite_reaches_every_expression_slot():
    from runtime._compat import AggCall, Aggregate, Project, Sort
    plan = Sort(child=Project(child=Aggregate(
        child=Filter(child=scan("sales", pred=op(">", col("qty", "s"), col("id", "s"))),
                     predicate=op(">", col("amount", "s"), col("qty"))),
        group_keys=[col("region", "s")], aggs=[(AggCall("sum", col("qty", "s")), "q")]),
        exprs=[(col("region", "s"), "region"), (col("q"), "q")]),
        keys=[(col("q"), True)])
    fixed = resolve_aliases(plan, CATALOG)
    assert all(r.table in (None, "sales") for r in _refs(fixed))
    assert format_plan(fixed).count("sales.") == format_plan(plan).count("s.")


# ------------------------------------------------------------------ the runner

@pytest.fixture(scope="module")
def tiny_rows(tiny):
    catalog, tables = tiny
    optimize, label = runner.find_optimizer()
    rows = []
    for name, sql in runner.load_queries(["q01", "q07", "q14"]).items():
        rows += runner.measure_query(name, sql, catalog, tables, optimize, label, "tiny",
                                     runs=1, warmup=1)
    return rows


def test_runner_measures_every_configuration_and_every_answer_is_right(tiny_rows):
    rows = tiny_rows
    assert len(rows) == 3 * len(runner.CONFIGURATIONS)
    assert all(m.matches_reference for m in rows)
    assert all(m.runtime_ms > 0 and m.peak_memory_kb > 0 and m.rows_scanned > 0 for m in rows)
    compiled = [m for m in rows if m.configuration.startswith("compiled")]
    assert all(m.compile_ms is not None and m.compile_ms > 0 for m in compiled)
    assert all(m.compile_ms is None for m in rows if m not in compiled)
    q01 = {m.configuration: m for m in rows if m.query == "q01"}
    assert q01["interpreted_unoptimized"].rows_out == 10                # LIMIT 10
    assert q01["interpreted_unoptimized"].rows_scanned == 100           # tiny customer
    md = runner.summary(rows).splitlines()
    assert any(line.startswith("| geomean") for line in md)
    # the binder's plan for q01 is Limit <- Sort <- Project <- Filter <- Scan: one chain fuses
    assert q01["compiled_unoptimized"].fused_pipelines == 1
    assert q01["compiled_unoptimized_unfused"].fused_pipelines == 0
    assert q01["interpreted_unoptimized"].fused_pipelines is None


def test_interleaved_timing_runs_every_callable_each_round():
    calls = []
    fns = {k: (lambda k=k: calls.append(k) or k) for k in "abc"}
    medians, first, calls_per = runner.time_interleaved(fns, runs=3, warmup=1, min_block_s=0)
    assert calls == list("abc") + list("abc") * 3
    assert first == {"a": "a", "b": "b", "c": "c"} and set(medians) == set("abc")


def test_csv_is_readable_by_person_a_report(tiny_rows, tmp_path):
    from bench.report import CSV_COLUMNS as A_COLUMNS, BenchmarkRecord
    path = runner.write_csv(tiny_rows, tmp_path / "r.csv")
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames[:len(A_COLUMNS)] == A_COLUMNS
        records = [BenchmarkRecord(query=r["query"], configuration=r["configuration"],
                                   runtime_ms=float(r["runtime_ms"]),
                                   rows_scanned=int(r["rows_scanned"]),
                                   peak_memory_kb=float(r["peak_memory_kb"])) for r in reader]
    from bench.report import CONFIGURATIONS as A_CONFIGS
    assert set(A_CONFIGS) <= {r.configuration for r in records}


def test_wrong_answers_are_reported_not_hidden(tiny):
    catalog, tables = tiny
    def broken_optimizer(plan, catalog):          # drops every row
        return Limit(child=plan, n=0), []
    sql = runner.load_queries(["q01"])["q01"]
    rows = runner.measure_query("q01", sql, catalog, tables, broken_optimizer, "broken", "tiny",
                                runs=1, warmup=1)
    verdict = {m.configuration: m.matches_reference for m in rows}
    assert verdict == {"interpreted_unoptimized": True, "interpreted_optimized": False,
                       "compiled_unoptimized": True, "compiled_optimized": False,
                       "compiled_unoptimized_unfused": True, "compiled_optimized_unfused": False}
    assert "WRONG" in runner.summary(rows)


def test_main_exits_nonzero_on_a_wrong_answer(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "find_optimizer",
                        lambda: (lambda plan, catalog: (Limit(child=plan, n=0), []), "broken"))
    code = runner.main(["--scale", "tiny", "--queries", "q01", "--runs", "1",
                        "--configs", "interpreted_optimized", "--out", str(tmp_path / "x.csv")])
    assert code == 1


def test_identity_optimizer_when_person_b_has_none(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "optimizer", types.ModuleType("optimizer"))
    optimize, label = runner.find_optimizer()
    assert label.startswith("identity")
    plan = scan("sales")
    assert optimize(plan, CATALOG) == (plan, [])


def test_cells_scanned_counts_pruned_columns_and_pushed_predicate_columns():
    plain = scan("sales")
    pruned = scan("sales", columns=["id"], pred=op(">", col("qty"), col("amount")))
    assert runner.cells_scanned(plain, TABLES, CATALOG) == 6 * 5
    assert runner.cells_scanned(pruned, TABLES, CATALOG) == 6 * 3   # id, qty, amount
    assert runner.rows_scanned(pruned, TABLES) == 6


def test_csv_round_trip_and_markdown_report(tiny_rows, tmp_path):
    path = runner.write_csv(tiny_rows, tmp_path / "r.csv")
    assert runner.read_csv(path) == tiny_rows
    md = runner.write_markdown(runner.read_csv(path), tmp_path / "r.md").read_text(encoding="utf-8")
    assert "every answer matches the reference interpreter: yes" in md
    assert "| geomean" in md and "python:" in md
    assert runner.main(["--summarize", str(path)]) == 0
    assert (tmp_path / "r.md").exists()
