from __future__ import annotations

from pathlib import Path
import pytest

from bench.data.generate import create_test_catalog, generate_dataset
from bench.harness.differential import (
    QueryResult,
    compare_results,
    run_differential_query,
)
from catalog.catalog import Catalog


FIXTURES_DIR = Path(__file__).parent / "fixtures"
QUERIES_DIR = FIXTURES_DIR / "queries"
QUERY_IDS = [f"q{i:02d}" for i in range(1, 21)]


@pytest.fixture(scope="module")
def shared_dataset_and_catalog() -> tuple[Catalog, dict]:
    """Generates deterministic tiny star-schema tables and registered Catalog instance."""
    catalog = create_test_catalog()
    tables = generate_dataset("tiny", seed=42)
    return catalog, tables


@pytest.mark.parametrize("query_id", QUERY_IDS)
def test_differential_query_conformance(
    query_id: str,
    shared_dataset_and_catalog: tuple[Catalog, dict],
) -> None:
    """Verifies that each of the 20 golden queries executes differentially with identical results."""
    catalog, tables = shared_dataset_and_catalog
    sql_path = QUERIES_DIR / f"{query_id}.sql"
    assert sql_path.exists(), f"Query fixture missing: {sql_path}"

    sql = sql_path.read_text(encoding="utf-8").strip()
    assert run_differential_query(sql, catalog, tables) is True


def test_differential_harness_detects_mismatches() -> None:
    """Verifies that compare_results catches row count, schema, and value discrepancies."""
    res_a = QueryResult(column_names=["col1", "col2"], rows=[(1, "a"), (2, "b"), (3, "c")])
    res_b = QueryResult(column_names=["col1", "col2"], rows=[(1, "a"), (2, "WRONG"), (3, "c")])

    # 1. Detects value mismatch and formats differing row
    ok, diff = compare_results(res_a, res_b)
    assert ok is False
    assert "differing row(s) detected" in diff
    assert "WRONG" in diff

    # 2. Detects row count mismatch
    res_c = QueryResult(column_names=["col1", "col2"], rows=[(1, "a")])
    ok_count, diff_count = compare_results(res_a, res_c)
    assert ok_count is False
    assert "Row counts differ" in diff_count

    # 3. Detects schema mismatch
    res_d = QueryResult(column_names=["diff_col", "col2"], rows=[(1, "a")])
    ok_schema, diff_schema = compare_results(res_a, res_d)
    assert ok_schema is False
    assert "Schema columns differ" in diff_schema
