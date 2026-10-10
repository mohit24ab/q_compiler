from __future__ import annotations

from pathlib import Path
import duckdb
import pytest

from bench.data.generate import generate_dataset

FIXTURES_DIR = Path(__file__).parent / "fixtures"
QUERIES_DIR = FIXTURES_DIR / "queries"
QUERY_IDS = [f"q{i:02d}" for i in range(1, 21)]


@pytest.fixture(scope="module")
def duckdb_con() -> duckdb.DuckDBPyConnection:
    """Registers tiny star-schema dataset in an in-memory DuckDB connection."""
    con = duckdb.connect()
    tables = generate_dataset(scale="tiny", seed=42)
    for name, table in tables.items():
        con.register(name, table)
    return con


@pytest.mark.parametrize("query_id", QUERY_IDS)
def test_duckdb_oracle_executes_golden_queries(
    query_id: str,
    duckdb_con: duckdb.DuckDBPyConnection,
) -> None:
    """Validates that golden queries execute in DuckDB without syntax or semantic errors."""
    sql_path = QUERIES_DIR / f"{query_id}.sql"
    assert sql_path.exists(), f"Query file not found: {sql_path}"

    sql = sql_path.read_text(encoding="utf-8").strip()
    result = duckdb_con.execute(sql).fetchall()
    assert result is not None
