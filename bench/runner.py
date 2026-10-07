"""Phase C6 benchmark runner: the query suite under {interpreted, compiled} x {unoptimized, optimized}.

    python -m bench.runner --scale bench                 # all 20 golden queries
    python -m bench.runner --scale tiny --queries q01 q07 --runs 3

The 2x2 separates the optimizer's contribution (interpreted_unoptimized vs
interpreted_optimized) from compilation's (interpreted_optimized vs compiled_optimized).
Two more configurations turn operator fusion (Phase C5) off, so its own contribution can
be read off too, on plans the optimizer has and hasn't already pruned:
compiled_unoptimized_unfused and compiled_optimized_unfused.

What is measured, per query and configuration:
  runtime_ms       median, over `--runs` timed samples, of the time per call, after
                   `--warmup` discarded calls. A sample is a block of back-to-back calls
                   lasting at least 50 ms (one call for anything slower: `calls_per_run`),
                   and the configurations of a query take turns, one block each per round
                   (see time_interleaved for why). Execution only: SQL binding, optimizing
                   and code generation happen once, before timing, as for a prepared query.
  compile_ms       for compiled configurations, generate() + Python's compile(), measured once.
  peak_memory_kb   tracemalloc peak during one extra run, kept apart from the timed runs
                   because tracing slows everything down (numpy reports its allocations).
  rows_scanned     base-table rows read by the plan's Scans (Person A's definition).
  cells_scanned    rows x columns each Scan reads (its output columns plus those its pushed
                   predicate needs), so column pruning shows up.
  rows_out         result rows.
  fused_pipelines  for compiled configurations, how many Scan/Filter/Project chains were
                   fused; when the fused and unfused sources are identical (0 here), the
                   "fusion" ratio is only noise and the summary leaves it out.
  matches_reference  whether the result equals interpret(unoptimized plan) (Contract §7).
                   A configuration that gives a wrong answer is still timed, but reported,
                   and the runner exits non-zero.

Inputs: the golden queries in tests/fixtures/queries/ and the generator in
bench/data/generate.py (both Person A's), converted to runtime Tables once up front so
every configuration measures query work, not Arrow conversion. Binding goes through
frontend.binder.parse_and_bind and then bench.aliases.resolve_aliases (see that module).

The optimizer is `optimizer.optimize` when Person B's package provides it, otherwise the
identity, and the `optimizer` column says which one ran. The CSV starts with Person A's
columns (bench/report.py: CSV_COLUMNS) so `bench.report.generate_charts(path)` reads it.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import platform
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from bench.aliases import resolve_aliases
from codegen import generate
from codegen.runner import compile_module
from runtime import Table, compare_tables, interpret
from runtime.expr_eval import node_kind

ROOT = Path(__file__).resolve().parent.parent
QUERY_DIR = ROOT / "tests" / "fixtures" / "queries"

# name -> (engine, optimized, fused)
CONFIGURATIONS = {
    "interpreted_unoptimized": ("interpreted", False, None),
    "interpreted_optimized": ("interpreted", True, None),
    "compiled_unoptimized": ("compiled", False, True),
    "compiled_optimized": ("compiled", True, True),
    "compiled_unoptimized_unfused": ("compiled", False, False),
    "compiled_optimized_unfused": ("compiled", True, False),
}


@dataclass
class Measurement:
    # Person A's columns first, in A's order
    query: str
    configuration: str
    runtime_ms: float
    rows_scanned: int
    peak_memory_kb: float
    # ours
    compile_ms: float | None
    cells_scanned: int
    rows_out: int
    runs: int
    calls_per_run: int            # back-to-back calls timed as one sample
    matches_reference: bool
    fused_pipelines: int | None   # compiled only: "# Fused pipeline" sections in the source
    optimizer: str
    scale: str


CSV_COLUMNS = [f.name for f in fields(Measurement)]


# ---------------------------------------------------------------------- workload

def load_queries(names=None) -> dict[str, str]:
    queries = {p.stem: p.read_text(encoding="utf-8").strip()
               for p in sorted(QUERY_DIR.glob("q*.sql"))}
    if names:
        unknown = sorted(set(names) - set(queries))
        if unknown:
            raise SystemExit(f"unknown queries {unknown}; have {sorted(queries)}")
        queries = {n: queries[n] for n in names}
    return queries


def load_data(scale: str, seed: int = 42):
    """(catalog, tables) with tables already converted to runtime Tables."""
    from bench.data.generate import create_test_catalog, generate_dataset
    catalog = create_test_catalog(scale=scale, seed=seed)
    arrow = generate_dataset(scale=scale, seed=seed)
    return catalog, {name: Table.from_arrow(t, name) for name, t in arrow.items()}


def find_optimizer():
    """(optimize, label): Person B's optimizer if importable, else the identity."""
    try:
        import optimizer
        optimize = getattr(optimizer, "optimize", None)
    except ImportError:
        optimize = None
    if callable(optimize):
        return optimize, "optimizer.optimize"
    return (lambda plan, catalog: (plan, [])), "identity (optimizer.optimize not available)"


def bind(sql: str, catalog):
    from frontend.binder import parse_and_bind
    return resolve_aliases(parse_and_bind(sql, catalog), catalog)


# ---------------------------------------------------------------------- plan metrics

def _scans(plan):
    if node_kind(plan) == "Scan":
        yield plan
    for child in plan.children:
        yield from _scans(child)


def rows_scanned(plan, tables) -> int:
    return sum(tables[s.table].num_rows for s in _scans(plan))


def cells_scanned(plan, tables, catalog) -> int:
    from codegen.exprgen import column_refs
    total = 0
    for s in _scans(plan):
        schema = s.table_schema or catalog.schema(s.table)
        cols = set(s.columns) if s.columns is not None else {n for n, _ in schema}
        cols |= {r.name for r in column_refs(s.pushed_predicate)}
        total += tables[s.table].num_rows * len(cols)
    return total


def is_ordered(sql: str) -> bool:
    return "order by" in sql.lower()


# ---------------------------------------------------------------------- measuring

def _peak_kb(fn) -> float:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1] / 1024.0
    finally:
        tracemalloc.stop()


def time_interleaved(fns: dict, runs: int, warmup: int, min_block_s: float = 0.05):
    """({key: median ms per call}, {key: first result}, {key: calls per sample}).

    `warmup` untimed calls of each callable first (at least one: its time sizes the
    blocks, its result is the one checked). Then `runs` rounds; each round times one
    block of every callable, in turn. A block is as many back-to-back calls as fill
    `min_block_s` (one call for anything slower), and its time / calls is one sample.

    Why blocks, and why round-robin. Timing one call of each configuration in turn put
    a 0.25 ms compiled query right after a seconds-long interpreted one, whose garbage
    and cache traffic then landed inside the short call (1.6 ms measured); collecting
    garbage before each call instead left the caches cold. Blocks spread any such
    one-off cost over many calls, as `timeit` does. Rounds still alternate between
    configurations, so a slow stretch of the machine (another process, throttling)
    lands on all of them alike: timing one configuration's runs back to back measured
    the very same code at 27.0 and 17.8 ms. The garbage collector is off inside a block.
    """
    first, per_call = {}, {}
    for key, fn in fns.items():
        for _ in range(max(1, warmup)):
            t0 = time.perf_counter()
            out = fn()
            per_call[key] = time.perf_counter() - t0
            first.setdefault(key, out)
    calls = {k: max(1, math.ceil(min_block_s / max(t, 1e-9))) if min_block_s else 1
             for k, t in per_call.items()}
    timings = {key: [] for key in fns}
    for _ in range(max(1, runs)):
        for key, fn in fns.items():
            n = calls[key]
            gc.disable()
            try:
                t0 = time.perf_counter()
                for _ in range(n):
                    fn()
                elapsed = (time.perf_counter() - t0) / n
            finally:
                gc.enable()
            timings[key].append(elapsed)
    medians = {k: statistics.median(v) * 1000.0 for k, v in timings.items()}
    return medians, first, calls


def measure_query(name, sql, catalog, tables, optimize, optimizer_label, scale,
                  configs=tuple(CONFIGURATIONS), runs=5, warmup=1) -> list[Measurement]:
    plan = bind(sql, catalog)
    optimized, _traces = optimize(plan, catalog)
    reference = interpret(plan, tables)
    fns, compile_ms, plans, pipelines = {}, {}, {}, {}
    for config in configs:
        engine, use_opt, fused = CONFIGURATIONS[config]
        plans[config] = p = optimized if use_opt else plan
        if engine == "interpreted":
            fns[config] = lambda p=p: interpret(p, tables)
            compile_ms[config], pipelines[config] = None, None
        else:
            t0 = time.perf_counter()
            source = generate(p, catalog, mode="compiled", fuse=fused)
            run = compile_module(source)["run"]
            compile_ms[config] = round((time.perf_counter() - t0) * 1000.0, 3)
            pipelines[config] = source.count("# Fused pipeline")
            fns[config] = lambda run=run: run(tables)
    medians, results, calls = time_interleaved(fns, runs, warmup)
    out = []
    for config in configs:
        ok, _ = compare_tables(reference, results[config], ordered=is_ordered(sql))
        p = plans[config]
        out.append(Measurement(
            query=name, configuration=config, runtime_ms=round(medians[config], 3),
            rows_scanned=rows_scanned(p, tables), peak_memory_kb=round(_peak_kb(fns[config]), 2),
            compile_ms=compile_ms[config], cells_scanned=cells_scanned(p, tables, catalog),
            rows_out=results[config].num_rows, runs=max(1, runs), calls_per_run=calls[config],
            matches_reference=ok,
            fused_pipelines=pipelines[config],
            optimizer=optimizer_label if CONFIGURATIONS[config][1] else "none", scale=scale))
    return out


def run_suite(scale="tiny", queries=None, configs=tuple(CONFIGURATIONS), runs=5, warmup=1,
              seed=42, progress=None) -> list[Measurement]:
    catalog, tables = load_data(scale, seed)
    optimize, label = find_optimizer()
    results = []
    for name, sql in load_queries(queries).items():
        rows = measure_query(name, sql, catalog, tables, optimize, label, scale,
                             configs=configs, runs=runs, warmup=warmup)
        results.extend(rows)
        if progress:
            progress(name, rows)
    return results


# ---------------------------------------------------------------------- output

def write_csv(results: list[Measurement], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for m in results:
            w.writerow(asdict(m))
    return path


def _geomean(xs):
    xs = [x for x in xs if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def summary(results: list[Measurement]) -> str:
    """Markdown: per-query times and the three ratios the 2x2 (+ fusion) is for."""
    by = {(m.query, m.configuration): m for m in results}
    queries = sorted({m.query for m in results})
    cols = [c for c in CONFIGURATIONS if any((q, c) in by for q in queries)]

    def ratio(q, a, b):
        if (q, a) not in by or (q, b) not in by or by[(q, b)].runtime_ms <= 0:
            return None
        if "unfused" in a and not by[(q, b)].fused_pipelines:
            return None  # nothing fused: both configurations ran the same code
        return by[(q, a)].runtime_ms / by[(q, b)].runtime_ms

    ratios = {
        "optimizer, interpreted": ("interpreted_unoptimized", "interpreted_optimized"),
        "optimizer, compiled": ("compiled_unoptimized", "compiled_optimized"),
        "compilation": ("interpreted_optimized", "compiled_optimized"),
        "fusion, unoptimized": ("compiled_unoptimized_unfused", "compiled_unoptimized"),
        "fusion, optimized": ("compiled_optimized_unfused", "compiled_optimized"),
        "total": ("interpreted_unoptimized", "compiled_optimized"),
    }
    head = ["query", *[f"{c} ms" for c in cols], *[f"{r} x" for r in ratios]]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for q in queries:
        cells = [q]
        for c in cols:
            m = by.get((q, c))
            cells.append("" if m is None else f"{m.runtime_ms:.2f}" + ("" if m.matches_reference else " WRONG"))
        for a, b in ratios.values():
            r = ratio(q, a, b)
            cells.append("" if r is None else f"{r:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    geo = ["geomean", *["" for _ in cols],
           *[f"{_geomean([ratio(q, a, b) for q in queries]):.2f}" for a, b in ratios.values()]]
    lines.append("| " + " | ".join(geo) + " |")
    lines.append("")
    lines.append("A blank fusion ratio means no chain fused: both configurations ran the same code.")
    return "\n".join(lines)


def read_csv(path) -> list[Measurement]:
    """The rows write_csv wrote, with their types back."""
    def parse(name, value):
        if value == "" and name in ("compile_ms", "fused_pipelines"):
            return None
        kind = Measurement.__dataclass_fields__[name].type
        if "bool" in kind:
            return value == "True"
        if "int" in kind:
            return int(value)
        if "float" in kind:
            return float(value)
        return value
    with Path(path).open(encoding="utf-8") as f:
        return [Measurement(**{k: parse(k, v) for k, v in row.items()}) for row in csv.DictReader(f)]


def environment() -> dict[str, str]:
    import numpy
    return {"python": platform.python_version(), "numpy": numpy.__version__,
            "platform": platform.platform(), "processor": platform.processor() or "unknown"}


def write_markdown(results: list[Measurement], path, env: dict | None = None) -> Path:
    """The summary as a small report: where it was measured, then the table."""
    path = Path(path)
    first = results[0]
    env = env or environment()
    optimizers = sorted({m.optimizer for m in results} - {"none"})
    lines = [
        f"# Benchmark results: scale `{first.scale}`",
        "",
        f"Produced by `python -m bench.runner --scale {first.scale}`; raw numbers in "
        f"`{path.with_suffix('.csv').name}`. Median of {first.runs} timed runs per configuration "
        f"(after a warm-up), configurations interleaved round-robin. Times are execution "
        f"only: binding, optimizing and code generation happen once, beforehand.",
        "",
        f"* optimizer: {', '.join(optimizers) or 'none'}",
        *[f"* {k}: {v}" for k, v in env.items()],
        f"* every answer matches the reference interpreter: "
        f"{'yes' if all(m.matches_reference for m in results) else 'NO'}",
        "",
        "Ratios are speedups (higher is better). `optimizer` compares unoptimized with "
        "optimized plans on the same engine, `compilation` compares the interpreter with "
        "generated code on the optimized plan, `fusion` compares generated code with fusion "
        "off and on, `total` compares the naive baseline with the full pipeline.",
        "",
        summary(results),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scale", default="tiny", choices=["tiny", "bench"])
    ap.add_argument("--queries", nargs="*", help="e.g. q01 q07 (default: all)")
    ap.add_argument("--configs", nargs="*", choices=list(CONFIGURATIONS), default=list(CONFIGURATIONS))
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", help="CSV path (default: bench/results/runner_<scale>.csv); "
                                  "a .md summary is written next to it")
    ap.add_argument("--summarize", metavar="CSV", help="only rewrite the .md summary of a CSV")
    args = ap.parse_args(argv)
    if args.summarize:
        md = write_markdown(read_csv(args.summarize), Path(args.summarize).with_suffix(".md"))
        print(md.read_text(encoding="utf-8"))
        return 0

    def progress(name, rows):
        times = "  ".join(f"{m.configuration}={m.runtime_ms:.1f}ms" + ("" if m.matches_reference else "(WRONG)")
                          for m in rows)
        print(f"{name}: {times}", flush=True)

    _, label = find_optimizer()
    print(f"scale={args.scale} runs={args.runs} warmup={args.warmup} optimizer={label}", flush=True)
    results = run_suite(args.scale, args.queries, args.configs, args.runs, args.warmup,
                        args.seed, progress)
    out = write_csv(results, args.out or ROOT / "bench" / "results" / f"runner_{args.scale}.csv")
    md = write_markdown(results, out.with_suffix(".md"))
    print(f"\nwrote {len(results)} rows to {out}, summary in {md}\n")
    print(summary(results))
    wrong = [f"{m.query}/{m.configuration}" for m in results if not m.matches_reference]
    if wrong:
        print(f"\nWRONG ANSWERS: {wrong}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
