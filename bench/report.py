from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import statistics
import time
import tracemalloc
from typing import Any, Callable

from bench.data.generate import create_test_catalog, generate_dataset
from bench.harness import (
    compile_and_run,
    generate,
    interpret,
    optimize,
)
from catalog.catalog import Catalog
from frontend import parse_and_bind
from ir.nodes import PlanNode, Scan


CONFIGURATIONS = [
    "interpreted_unoptimized",
    "interpreted_optimized",
    "compiled_unoptimized",
    "compiled_optimized",
]

CSV_COLUMNS = [
    "query",
    "configuration",
    "runtime_ms",
    "rows_scanned",
    "peak_memory_kb",
]


@dataclass(frozen=True)
class BenchmarkRecord:
    """Individual benchmark run measurement."""
    query: str
    configuration: str
    runtime_ms: float
    rows_scanned: int
    peak_memory_kb: float


def find_scans(node: PlanNode) -> list[Scan]:
    """Recursively locates all Scan nodes in a PlanNode tree."""
    scans: list[Scan] = []
    if isinstance(node, Scan):
        scans.append(node)
    for child in node.children:
        scans.extend(find_scans(child))
    return scans


def compute_rows_scanned(plan: PlanNode, tables: dict[str, Any], catalog: Catalog | None = None) -> int:
    """Computes total input rows scanned by table scans in the plan."""
    scans = find_scans(plan)
    total_rows = 0
    for s in scans:
        if s.table in tables:
            t = tables[s.table]
            if hasattr(t, "num_rows"):
                total_rows += t.num_rows
            else:
                total_rows += len(t)
        elif catalog is not None:
            try:
                total_rows += catalog.row_count(s.table)
            except Exception:
                pass
    return total_rows


def create_execution_callable(
    config: str,
    sql: str,
    catalog: Catalog,
    tables: dict[str, Any],
) -> tuple[Callable[[], Any], PlanNode]:
    """Builds an execution callable and reference plan for a specific benchmark configuration."""
    plan_unopt = parse_and_bind(sql, catalog)

    if config == "interpreted_unoptimized":
        return (lambda: interpret(plan_unopt, tables)), plan_unopt

    elif config == "interpreted_optimized":
        plan_opt, _ = optimize(plan_unopt, catalog)
        return (lambda: interpret(plan_opt, tables)), plan_opt

    elif config == "compiled_unoptimized":
        source = generate(plan_unopt, catalog)
        try:
            from codegen.runner import compile_module
            run_fn = compile_module(source)["run"]
            return (lambda: run_fn(tables)), plan_unopt
        except Exception:
            return (lambda: compile_and_run(source, tables)), plan_unopt

    elif config == "compiled_optimized":
        plan_opt, _ = optimize(plan_unopt, catalog)
        source = generate(plan_opt, catalog)
        try:
            from codegen.runner import compile_module
            run_fn = compile_module(source)["run"]
            return (lambda: run_fn(tables)), plan_opt
        except Exception:
            return (lambda: compile_and_run(source, tables)), plan_opt

    else:
        raise ValueError(
            f"Unknown configuration '{config}'. Expected one of {CONFIGURATIONS}."
        )


def run_single_benchmark(
    query_name: str,
    sql: str,
    config: str,
    catalog: Catalog,
    tables: dict[str, Any],
    num_runs: int = 5,
    warmup_runs: int = 1,
) -> BenchmarkRecord:
    """Executes a benchmark for a single query and configuration.

    Measures median wall-clock runtime (discarding warmup runs) and peak memory tracking.
    """
    exec_fn, plan = create_execution_callable(config, sql, catalog, tables)
    rows_scanned = compute_rows_scanned(plan, tables, catalog)

    # 1. Warmup runs (discarded from timing)
    for _ in range(warmup_runs):
        exec_fn()

    # 2. Timed runs & peak memory measurement
    runtimes_sec: list[float] = []
    max_peak_bytes = 0

    for _ in range(max(1, num_runs)):
        tracemalloc.start()
        t0 = time.perf_counter()
        exec_fn()
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtimes_sec.append(t1 - t0)
        if peak > max_peak_bytes:
            max_peak_bytes = peak

    median_ms = round(statistics.median(runtimes_sec) * 1000.0, 3)
    peak_kb = round(max_peak_bytes / 1024.0, 2)

    return BenchmarkRecord(
        query=query_name,
        configuration=config,
        runtime_ms=median_ms,
        rows_scanned=rows_scanned,
        peak_memory_kb=peak_kb,
    )


def run_benchmark_matrix(
    queries: dict[str, str],
    catalog: Catalog,
    tables: dict[str, Any],
    configs: list[str] | None = None,
    num_runs: int = 5,
    warmup_runs: int = 1,
) -> list[BenchmarkRecord]:
    """Runs all benchmark configurations across all provided queries."""
    active_configs = configs if configs is not None else CONFIGURATIONS
    results: list[BenchmarkRecord] = []

    for q_name, sql in queries.items():
        for cfg in active_configs:
            rec = run_single_benchmark(
                query_name=q_name,
                sql=sql,
                config=cfg,
                catalog=catalog,
                tables=tables,
                num_runs=num_runs,
                warmup_runs=warmup_runs,
            )
            results.append(rec)

    return results


def export_results_to_csv(
    results: list[BenchmarkRecord],
    output_path: str | Path = "bench/results.csv",
) -> Path:
    """Exports benchmark records to formatted CSV matching expected columns."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with out_p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rec in results:
            writer.writerow(asdict(rec))

    return out_p


def load_ablation_data(csv_path: str | Path = "docs/ablation/codegen.csv") -> list[dict[str, Any]] | None:
    """Reads ablation metric records from docs/ablation/codegen.csv if present."""
    p = Path(csv_path)
    if not p.exists():
        return None

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def generate_charts(
    results: list[BenchmarkRecord] | Path | None = None,
    output_dir: str | Path = "bench/figures",
    ablation_csv: str | Path = "docs/ablation/codegen.csv",
) -> dict[str, Path | None]:
    """Generates publication-ready figures using matplotlib if available.

    Produces:
      - runtime_comparison.png: comparing unoptimized vs optimized/compiled modes across queries.
      - ablation_chart.png: pass contribution / ablation chart.

    If matplotlib is unavailable, returns structured summary without crashing.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_manifest: dict[str, Path | None] = {
        "runtime_comparison": None,
        "ablation_chart": None,
    }

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive headless backend
        import matplotlib.pyplot as plt
    except ImportError:
        # Graceful fallback: return without crashing
        return output_manifest

    # 1. Parse results if path or None provided
    records: list[BenchmarkRecord] = []
    if isinstance(results, list):
        records = results
    elif isinstance(results, (str, Path)) and Path(results).exists():
        with Path(results).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                records.append(
                    BenchmarkRecord(
                        query=r["query"],
                        configuration=r["configuration"],
                        runtime_ms=float(r["runtime_ms"]),
                        rows_scanned=int(r["rows_scanned"]),
                        peak_memory_kb=float(r["peak_memory_kb"]),
                    )
                )

    # 2. Generate runtime_comparison.png
    if records:
        queries = sorted(list({r.query for r in records}))
        cfg_runtimes: dict[str, list[float]] = {}
        for cfg in CONFIGURATIONS:
            cfg_runtimes[cfg] = []
            for q in queries:
                matching = [r.runtime_ms for r in records if r.query == q and r.configuration == cfg]
                cfg_runtimes[cfg].append(matching[0] if matching else 0.0)

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.2
        indices = range(len(queries))

        colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE"]
        for i, cfg in enumerate(CONFIGURATIONS):
            pos = [x + (i - 1.5) * bar_width for x in indices]
            ax.bar(pos, cfg_runtimes[cfg], width=bar_width, label=cfg, color=colors[i % len(colors)])

        ax.set_xlabel("Query", fontsize=12, fontweight="bold")
        ax.set_ylabel("Runtime (ms)", fontsize=12, fontweight="bold")
        ax.set_title("Runtime Comparison across Execution Modes", fontsize=14, fontweight="bold")
        ax.set_xticks(list(indices))
        ax.set_xticklabels(queries, rotation=45, ha="right")
        ax.legend(title="Configuration")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        runtime_path = out_dir / "runtime_comparison.png"
        fig.savefig(runtime_path, dpi=300)
        plt.close(fig)
        output_manifest["runtime_comparison"] = runtime_path

    # 3. Generate ablation_chart.png from docs/ablation/codegen.csv
    ablation_data = load_ablation_data(ablation_csv)
    if ablation_data:
        if "config" in ablation_data[0]:
            configs = sorted(list({row["config"] for row in ablation_data}))
            seen_q: list[str] = []
            for r in ablation_data:
                if r["query"] not in seen_q:
                    seen_q.append(r["query"])
            queries_ab = seen_q[:10] if len(seen_q) > 15 else seen_q

            fig, ax = plt.subplots(figsize=(12, 6))
            bar_w = 0.8 / max(len(configs), 1)
            idxs = range(len(queries_ab))

            for j, cfg in enumerate(configs):
                vals = []
                for q in queries_ab:
                    matching = [float(r["runtime_ms"]) for r in ablation_data if r["query"] == q and r["config"] == cfg]
                    vals.append(matching[0] if matching else 0.0)
                pos = [x + (j - len(configs) / 2) * bar_w for x in idxs]
                ax.bar(pos, vals, width=bar_w, label=cfg)

            ax.set_xlabel("Query", fontsize=12, fontweight="bold")
            ax.set_ylabel("Runtime (ms)", fontsize=12, fontweight="bold")
            ax.set_title("Pass Contribution & Ablation Breakdown", fontsize=14, fontweight="bold")
            ax.set_xticks(list(idxs))
            ax.set_xticklabels(queries_ab, rotation=45, ha="right")
            ax.legend(title="Configuration")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            ablation_path = out_dir / "ablation_chart.png"
            fig.savefig(ablation_path, dpi=300)
            plt.close(fig)
            output_manifest["ablation_chart"] = ablation_path
        else:
            queries_ab = [row["query"] for row in ablation_data]
            metric_cols = [c for c in ablation_data[0].keys() if c != "query"]

            fig, ax = plt.subplots(figsize=(10, 5))
            bar_w = 0.8 / len(metric_cols)
            idxs = range(len(queries_ab))

            for j, col_name in enumerate(metric_cols):
                vals = [float(row[col_name]) for row in ablation_data]
                pos = [x + (j - len(metric_cols) / 2) * bar_w for x in idxs]
                ax.bar(pos, vals, width=bar_w, label=col_name)

            ax.set_xlabel("Query", fontsize=12, fontweight="bold")
            ax.set_ylabel("Runtime (ms)", fontsize=12, fontweight="bold")
            ax.set_title("Pass Contribution & Ablation Breakdown", fontsize=14, fontweight="bold")
            ax.set_xticks(list(idxs))
            ax.set_xticklabels(queries_ab)
            ax.legend(title="Stage / Pass")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()

            ablation_path = out_dir / "ablation_chart.png"
            fig.savefig(ablation_path, dpi=300)
            plt.close(fig)
            output_manifest["ablation_chart"] = ablation_path

    return output_manifest


def main() -> None:
    """CLI runner to execute benchmark pipeline and export results/charts."""
    parser = argparse.ArgumentParser(description="Query Compiler Benchmark Runner")
    parser.add_argument("--scale", choices=["tiny", "bench"], default="tiny", help="Dataset scale")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs")
    parser.add_argument("--output-csv", default="bench/results.csv", help="CSV export destination")
    parser.add_argument("--figures-dir", default="bench/figures", help="Directory for generated figures")
    args = parser.parse_args()

    print(f"Generating {args.scale} dataset...")
    catalog = create_test_catalog(scale=args.scale)
    tables = generate_dataset(scale=args.scale)

    queries_dir = Path("tests/fixtures/queries")
    query_files = sorted(queries_dir.glob("q*.sql"))
    queries = {q.stem: q.read_text(encoding="utf-8").strip() for q in query_files}
    print(f"Loaded {len(queries)} query fixtures.")

    print("Running benchmark matrix across configurations...")
    records = run_benchmark_matrix(
        queries=queries,
        catalog=catalog,
        tables=tables,
        num_runs=args.runs,
        warmup_runs=args.warmup,
    )

    csv_path = export_results_to_csv(records, args.output_csv)
    print(f"Exported {len(records)} benchmark records to {csv_path}.")

    figures = generate_charts(records, output_dir=args.figures_dir)
    print(f"Chart generation status: {figures}")

    docs_assets = Path("docs/assets")
    docs_assets.mkdir(parents=True, exist_ok=True)
    generate_charts(records, output_dir=docs_assets)


if __name__ == "__main__":
    main()
