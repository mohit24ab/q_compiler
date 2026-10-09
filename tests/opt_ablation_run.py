"""Run the ablation study and write the results for the report.

    python tests/opt_ablation_run.py --executor reference
    python tests/opt_ablation_run.py --executor codegen --scale 1 --golden-scale bench --repeat 5
    python tests/opt_ablation_run.py --order

Executors:
  reference  the tests' reference evaluator, on the suite's own tables. It is a
             naive row-at-a-time interpreter with nested-loop joins, so it
             exaggerates join costs; it runs anywhere.
  codegen    Person C's generated code (codegen.generate + compile_and_run), on
             two workloads: Person A's 20 golden queries, bound from SQL by
             the frontend, on Person A's dataset (--golden-scale, default
             bench); and the optimizer's suite on the larger tables of
             opt_bench_data.py (--scale). Needs Person C's code.

Writes docs/ablation/<executor>.csv and docs/ablation/<executor>.md (the
summary), or with --order, docs/ablation/pass_order.md.

docs/ablation/codegen.csv is the file Person A's bench/report.py charts. Its
format, agreed with Person A: one row per (query, configuration), with the
columns query, config, runtime_ms, result_rows, rows_scanned, values_read,
estimated_cost_us, same_result, workload. The golden queries come first, so a
chart of the first queries in the file shows them. The file is only ever
written by this script, from measurements.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import platform
import subprocess
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent)]

import opt_ir  # noqa: E402,F401
import opt_query_suite as S  # noqa: E402

import optimizer  # noqa: E402
from optimizer.ablation import (  # noqa: E402
    cost_model_check, order_table, pass_order_study, per_query_table, run_ablation, summary_table, to_csv,
)

OUT = Path(__file__).parent.parent / "docs" / "ablation"


def reference_executor():
    from opt_reference_eval import evaluate

    def prepare(plan):
        return lambda: evaluate(plan, S.TABLES).rows

    suite = Workload("suite", [(q.name, q.plan) for q in S.QUERIES], S.CATALOG, prepare, None,
                     "the suite's own tables (3 to 80 rows)")
    return [suite]


@dataclass
class Workload:
    name: str
    queries: list
    catalog: object
    prepare: object
    same: object
    data: str


def codegen_executor(scale: int, golden_scale: str):
    from codegen import generate
    from codegen.runner import compile_module
    from runtime import Table, compare_tables

    def compiled(catalog, tables):
        def prepare(plan):
            run = compile_module(generate(plan, catalog, mode="compiled"))["run"]
            return lambda: run(tables)
        return prepare

    def same(a, b):
        return compare_tables(b, a)[0]

    # Person A's 20 golden queries, bound from SQL by the frontend, on Person A's dataset.
    from bench.data.generate import create_test_catalog, generate_dataset
    from frontend import parse_and_bind

    arrow = generate_dataset(golden_scale, seed=42)
    golden_catalog = create_test_catalog(golden_scale, seed=42)
    golden_tables = {name: Table.from_arrow(t, table=name) for name, t in arrow.items()}
    fixtures = Path(__file__).parent / "fixtures" / "queries"
    golden_queries = [(f.stem, parse_and_bind(f.read_text(), golden_catalog)) for f in sorted(fixtures.glob("q*.sql"))]
    golden_sizes = ", ".join(f"{t} {tab.num_rows:,}" for t, tab in arrow.items())
    golden = Workload("golden", golden_queries, golden_catalog, compiled(golden_catalog, golden_tables), same,
                      f"Person A's dataset at scale `{golden_scale}`: {golden_sizes}")

    import opt_bench_data as B

    tables = B.make_tables(scale)
    catalog = B.catalog(tables)
    runtime_tables = {
        name: Table.from_rows([(c, t, name) for c, t in S.SCHEMAS[name]], rows)
        for name, (_, rows) in tables.items()
    }
    sizes = ", ".join(f"{t} {len(rows):,}" for t, (_, rows) in tables.items() if not t.startswith("k"))
    suite = Workload("suite", [(q.name, q.plan) for q in S.QUERIES], catalog, compiled(catalog, runtime_tables),
                     same, f"opt_bench_data.py at scale {scale}: {sizes}, k1..k9 {len(tables['k1'][1]):,} each")
    return [golden, suite]


def _without_outlier(measurements) -> list[str]:
    """The totals again, without the query that contributes most to the unoptimized total."""
    slowest = max((m for m in measurements if m.config == "no passes"), key=lambda m: m.runtime_ms)
    total = sum(m.runtime_ms for m in measurements if m.config == "no passes")
    share = slowest.runtime_ms / total if total else 0
    if share < 0.5:
        return []
    rest = [m for m in measurements if m.query != slowest.query]
    return [
        f"One query, `{slowest.query}`, takes {share:.0%} of the unoptimized total. "
        "The same totals without it:",
        "",
        summary_table(rest),
        "",
    ]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


TITLES = {
    "golden": "Person A's 20 golden queries",
    "suite": "The optimizer's differential suite",
}


def _section(workload: Workload, measurements, passes) -> list[str]:
    lines = [
        f"## {TITLES.get(workload.name, workload.name)}",
        "",
        f"{len(workload.queries)} queries. Data: {workload.data}.",
        "",
        summary_table(measurements),
        "",
        *_without_outlier(measurements),
        cost_model_check(measurements),
        "",
    ]
    for p in passes:
        lines += [f"### Without {p}", "", per_query_table(measurements, "all passes", f"without {p}", top=5), ""]
    lines += ["### No passes at all", "", per_query_table(measurements, "all passes", "no passes", top=8), ""]
    return lines


def run(executor: str, scale: int, golden_scale: str, repeat: int, note: str) -> None:
    workloads = reference_executor() if executor == "reference" else codegen_executor(scale, golden_scale)
    measurements = []
    by_workload = {}
    for w in workloads:
        ms = run_ablation(w.queries, w.catalog, w.prepare, repeat=repeat, same=w.same, workload=w.name)
        by_workload[w.name] = ms
        measurements += ms
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{executor}.csv").write_text(to_csv(measurements))
    passes = [p.name for p in optimizer.default_passes()]
    command = (f"--executor codegen --scale {scale} --golden-scale {golden_scale}"
               if executor == "codegen" else "--executor reference")
    lines = [
        f"# Ablation: {executor} executor",
        "",
        f"Generated by `python tests/opt_ablation_run.py {command} --repeat {repeat}`.",
        f"The raw numbers are in `{executor}.csv`: one row per query and configuration, with the",
        "columns query, config, runtime_ms, result_rows, rows_scanned, values_read,",
        "estimated_cost_us, same_result and workload"
        + (" (`golden` rows come first)." if executor == "codegen" else "."),
        "",
        "* Configurations: all passes; each pass switched off in turn; no passes (the plan as written).",
        f"* Runtime: the fastest of {repeat} runs of each plan, excluding optimization"
        + (", code generation and compilation." if executor == "codegen" else "."),
        "* `wrong results` counts queries whose result differs from the plan as written. It must be 0.",
        f"* Code: {note}." if note else f"* Code: commit `{_git('rev-parse', '--short', 'HEAD')}`.",
        f"* Machine: {platform.processor() or platform.machine()}, Python {platform.python_version()}.",
        "",
    ]
    for w in workloads:
        lines += _section(w, by_workload[w.name], passes)
    (OUT / f"{executor}.md").write_text("\n".join(lines))
    for w in workloads:
        print(f"== {w.name}")
        print(summary_table(by_workload[w.name]))
    print(f"wrote {OUT / executor}.csv and .md")


def order() -> None:
    queries = [(q.name, q.plan) for q in S.QUERIES]
    results = pass_order_study(queries, S.CATALOG)
    lines = [
        "# Pass order study",
        "",
        "Generated by `python tests/opt_ablation_run.py --order`.",
        "",
        f"Every order of the four passes, each run to a fixed point on the {len(queries)} suite queries.",
        "The cost is the B5 cost model's estimate for the final plans, summed. Pass applications and",
        "iterations count the work the fixed-point loop did to get there.",
        "",
        order_table(results),
        "",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pass_order.md").write_text("\n".join(lines))
    print(order_table(results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--executor", choices=["reference", "codegen"])
    parser.add_argument("--scale", type=int, default=1, help="scale of the suite's tables (opt_bench_data.py)")
    parser.add_argument("--golden-scale", default="bench", help="Person A's dataset scale for the golden queries")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--note", default="", help="extra provenance, e.g. which codegen commit")
    parser.add_argument("--order", action="store_true", help="run the pass order study")
    args = parser.parse_args()
    if args.order:
        order()
    if args.executor:
        run(args.executor, args.scale, args.golden_scale, args.repeat, args.note)


if __name__ == "__main__":
    main()
