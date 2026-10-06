from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet

from ir.dtype import DType


def pyarrow_type_to_dtype(pa_type: pa.DataType) -> DType:
    """Converts PyArrow DataType representations to the project DType enumeration."""
    if pa.types.is_integer(pa_type) or pa.types.is_unsigned_integer(pa_type):
        return DType.INT
    if pa.types.is_floating(pa_type):
        return DType.FLOAT
    if pa.types.is_boolean(pa_type):
        return DType.BOOL
    if pa.types.is_date(pa_type) or pa.types.is_timestamp(pa_type) or pa.types.is_time(pa_type):
        return DType.DATE
    if pa.types.is_string(pa_type) or pa.types.is_large_string(pa_type):
        return DType.STRING

    return DType.STRING


def load_csv(path: str | Path) -> pa.Table:
    """Reads a CSV dataset into a PyArrow Table."""
    return pa_csv.read_csv(Path(path))


def load_parquet(path: str | Path) -> pa.Table:
    """Reads a Parquet dataset into a PyArrow Table."""
    return pa_parquet.read_table(Path(path))