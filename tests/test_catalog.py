import datetime
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from catalog.catalog import Catalog
from catalog.stats import ColumnStats
from ir.dtype import DType
from ir.nodes import Scan


@pytest.fixture
def sample_parquet_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.parquet"
    table = pa.table({
        "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
        "amount": pa.array([10.5, 20.0, None, 40.2, 50.0], type=pa.float64()),
        "region": pa.array(["US", "EU", "US", "AP", None], type=pa.string()),
        "active": pa.array([True, False, True, True, False], type=pa.bool_()),
        "created_date": pa.array([
            datetime.date(2024, 1, 1),
            datetime.date(2024, 1, 2),
            datetime.date(2024, 1, 3),
            datetime.date(2024, 1, 4),
            datetime.date(2024, 1, 5),
        ], type=pa.date32())
    })
    pq.write_table(table, file_path)
    return file_path


@pytest.fixture
def sample_csv_file(tmp_path: Path) -> Path:
    file_path = tmp_path / "sample.csv"
    content = "id,region,score\n1,East,100\n2,West,200\n3,East,300\n"
    file_path.write_text(content)
    return file_path


def test_catalog_csv_loading_and_lookups(sample_csv_file: Path):
    cat = Catalog()
    cat.register_csv("csv_table", sample_csv_file)

    assert cat.schema("csv_table") == [("id", DType.INT), ("region", DType.STRING), ("score", DType.INT)]
    assert cat.row_count("csv_table") == 3

    stats_region = cat.stats("csv_table", "region")
    assert stats_region.ndv == 2
    assert stats_region.null_count == 0


def test_catalog_parquet_loading_and_statistics(sample_parquet_file: Path):
    cat = Catalog()
    cat.register_parquet("pq_table", sample_parquet_file)

    assert cat.row_count("pq_table") == 5

    # INT stats
    s_id = cat.stats("pq_table", "id")
    assert s_id.ndv == 5
    assert s_id.min == 1
    assert s_id.max == 5
    assert s_id.null_count == 0

    # FLOAT stats with nulls
    s_amt = cat.stats("pq_table", "amount")
    assert s_amt.ndv == 4
    assert s_amt.min == 10.5
    assert s_amt.max == 50.0
    assert s_amt.null_count == 1

    # STRING stats
    s_reg = cat.stats("pq_table", "region")
    assert s_reg.ndv == 3
    assert s_reg.null_count == 1

    # BOOL stats
    s_act = cat.stats("pq_table", "active")
    assert s_act.ndv == 2
    assert s_act.null_count == 0

    # DATE stats
    s_dt = cat.stats("pq_table", "created_date")
    assert s_dt.ndv == 5
    assert s_dt.min == "2024-01-01"
    assert s_dt.max == "2024-01-05"
    assert s_dt.null_count == 0


def test_catalog_error_paths():
    cat = Catalog()
    with pytest.raises(KeyError):
        cat.schema("nonexistent")
    with pytest.raises(KeyError):
        cat.row_count("nonexistent")
    with pytest.raises(KeyError):
        cat.stats("nonexistent", "col")


def test_scan_table_schema_integration():
    schema = [("id", DType.INT), ("region", DType.STRING), ("amount", DType.FLOAT)]

    # 1. Full schema when columns is None
    scan_full = Scan(table="sales", columns=None, pushed_predicate=None, table_schema=schema)
    assert scan_full.schema() == schema

    # 2. Projected columns with preserved ordering
    scan_proj = Scan(table="sales", columns=["amount", "id"], pushed_predicate=None, table_schema=schema)
    assert scan_proj.schema() == [("amount", DType.FLOAT), ("id", DType.INT)]

    # 3. Missing requested column raises ValueError
    scan_err = Scan(table="sales", columns=["unknown"], pushed_predicate=None, table_schema=schema)
    with pytest.raises(ValueError):
        scan_err.schema()

    # 4. Unpopulated table_schema raises NotImplementedError
    scan_none = Scan(table="sales", columns=None, pushed_predicate=None)
    with pytest.raises(NotImplementedError):
        scan_none.schema()

    # 5. replace_children preserves table_schema
    new_scan = scan_full.replace_children(())
    assert new_scan.table_schema == schema
    assert isinstance(new_scan.table_schema, list)