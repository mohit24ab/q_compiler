from __future__ import annotations

from pathlib import Path
import pytest

from bench.data.generate import create_test_catalog
from catalog.catalog import Catalog
from frontend.binder import parse_and_bind
from ir.nodes import PlanNode
from ir.printer import format_plan

FIXTURES_DIR = Path(__file__).parent / "fixtures"
QUERIES_DIR = FIXTURES_DIR / "queries"
GOLDEN_DIR = FIXTURES_DIR / "golden"

QUERY_IDS = [f"q{i:02d}" for i in range(1, 21)]


@pytest.fixture(scope="module")
def catalog() -> Catalog:
    """Provides a shared star-schema Catalog instance populated with tiny PyArrow tables."""
    return create_test_catalog()


@pytest.mark.parametrize("query_id", QUERY_IDS)
def test_golden_plan_matches(query_id: str, catalog: Catalog) -> None:
    """Verifies that the canonical unoptimized IR plan for each query matches its golden fixture."""
    sql_path = QUERIES_DIR / f"{query_id}.sql"
    golden_path = GOLDEN_DIR / f"{query_id}.txt"

    assert sql_path.exists(), f"Query file not found: {sql_path}"
    assert golden_path.exists(), f"Golden fixture not found: {golden_path}"

    sql = sql_path.read_text(encoding="utf-8").strip()
    plan: PlanNode = parse_and_bind(sql, catalog)
    actual_plan = format_plan(plan).replace("\r\n", "\n").strip()

    expected_plan = golden_path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()

    assert actual_plan == expected_plan, (
        f"Generated plan for {query_id} did not match golden fixture.\n"
        f"--- Actual ---\n{actual_plan}\n"
        f"--- Expected ---\n{expected_plan}"
    )
