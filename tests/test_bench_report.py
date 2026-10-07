from __future__ import annotations

import csv
from pathlib import Path
import pytest

from bench.data.generate import create_test_catalog, generate_dataset
from bench.report import (
    CONFIGURATIONS,
    CSV_COLUMNS,
    BenchmarkRecord,
    export_results_to_csv,
    generate_charts,
    load_ablation_data,
    run_benchmark_matrix,
    run_single_benchmark,
)
from catalog.catalog import Catalog


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "queries"


@pytest.fixture(scope="module")
def tiny_catalog_and_tables() -> tuple[Catalog, dict]:
    catalog = create_test_catalog(scale="tiny")
    tables = generate_dataset(scale="tiny", seed=42)
    return catalog, tables


def test_benchmark_matrix_subset_execution(tiny_catalog_and_tables: tuple[Catalog, dict]) -> None:
    """Verifies matrix benchmark execution on a 2-query subset across all 4 configurations."""
    catalog, tables = tiny_catalog_and_tables
    queries = {
        "q01": (FIXTURES_DIR / "q01.sql").read_text(encoding="utf-8").strip(),
        "q02": (FIXTURES_DIR / "q02.sql").read_text(encoding="utf-8").strip(),
    }

    records = run_benchmark_matrix(
        queries=queries,
        catalog=catalog,
        tables=tables,
        configs=CONFIGURATIONS,
        num_runs=2,
        warmup_runs=1,
    )

    # 2 queries * 4 configurations = 8 records
    assert len(records) == 8

    recorded_queries = {r.query for r in records}
    recorded_configs = {r.configuration for r in records}

    assert recorded_queries == {"q01", "q02"}
    assert recorded_configs == set(CONFIGURATIONS)

    for r in records:
        assert isinstance(r, BenchmarkRecord)
        assert r.runtime_ms > 0.0
        assert r.rows_scanned > 0
        assert r.peak_memory_kb >= 0.0


def test_csv_export_format(tmp_path: Path, tiny_catalog_and_tables: tuple[Catalog, dict]) -> None:
    """Verifies that export_results_to_csv produces well-formed CSV output with exact expected columns."""
    catalog, tables = tiny_catalog_and_tables
    queries = {
        "q01": (FIXTURES_DIR / "q01.sql").read_text(encoding="utf-8").strip(),
    }

    records = run_benchmark_matrix(
        queries=queries,
        catalog=catalog,
        tables=tables,
        configs=["interpreted_unoptimized", "compiled_optimized"],
        num_runs=1,
        warmup_runs=0,
    )

    csv_dest = tmp_path / "results.csv"
    exported_path = export_results_to_csv(records, output_path=csv_dest)
    assert exported_path == csv_dest
    assert csv_dest.exists()

    with csv_dest.open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert len(reader) == 3  # Header + 2 data rows
    assert reader[0] == CSV_COLUMNS

    for row in reader[1:]:
        assert len(row) == 5
        query_val, config_val, runtime_str, rows_str, mem_str = row
        assert query_val == "q01"
        assert config_val in ("interpreted_unoptimized", "compiled_optimized")
        assert float(runtime_str) >= 0.0
        assert int(rows_str) > 0
        assert float(mem_str) >= 0.0


def test_metric_calculation_determinism(tiny_catalog_and_tables: tuple[Catalog, dict]) -> None:
    """Verifies deterministic row scanning and timing execution."""
    catalog, tables = tiny_catalog_and_tables
    sql = (FIXTURES_DIR / "q01.sql").read_text(encoding="utf-8").strip()

    rec1 = run_single_benchmark("q01", sql, "interpreted_unoptimized", catalog, tables, num_runs=2, warmup_runs=1)
    rec2 = run_single_benchmark("q01", sql, "interpreted_unoptimized", catalog, tables, num_runs=2, warmup_runs=1)

    # Scanned rows must be perfectly deterministic
    assert rec1.rows_scanned == rec2.rows_scanned == 100
    assert rec1.runtime_ms > 0.0
    assert rec2.runtime_ms > 0.0


def test_chart_generation_and_ablation_loader(tmp_path: Path) -> None:
    """Verifies graceful chart generator execution and ablation dataset loading."""
    records = [
        BenchmarkRecord("q01", "interpreted_unoptimized", 10.5, 100, 15.0),
        BenchmarkRecord("q01", "compiled_optimized", 2.1, 100, 10.0),
    ]

    # Chart generation must run without crashing (whether matplotlib is present or absent)
    manifest = generate_charts(records, output_dir=tmp_path / "figures")
    assert isinstance(manifest, dict)
    assert "runtime_comparison" in manifest
    assert "ablation_chart" in manifest

    # Ablation loader checks docs/ablation/codegen.csv
    ablation_data = load_ablation_data("docs/ablation/codegen.csv")
    assert ablation_data is not None
    assert len(ablation_data) > 0
    assert "query" in ablation_data[0]
