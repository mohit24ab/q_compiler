from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc


@dataclass(frozen=True)
class ColumnStats:
    ndv: int
    min: Any
    max: Any
    null_count: int


def compute_column_stats(column: pa.ChunkedArray | pa.Array) -> ColumnStats:
    """Computes exact NDV, min, max, and null_count statistics for a PyArrow column."""
    null_count = column.null_count
    non_null_col = pc.drop_null(column)

    if len(non_null_col) == 0:
        return ColumnStats(ndv=0, min=None, max=None, null_count=null_count)

    ndv = len(pc.unique(non_null_col))

    try:
        min_max_struct = pc.min_max(non_null_col)
        min_py = min_max_struct["min"].as_py()
        max_py = min_max_struct["max"].as_py()
    except Exception:
        min_py = None
        max_py = None

    if isinstance(min_py, (datetime.date, datetime.datetime)):
        min_py = min_py.isoformat()
    if isinstance(max_py, (datetime.date, datetime.datetime)):
        max_py = max_py.isoformat()

    return ColumnStats(
        ndv=ndv,
        min=min_py,
        max=max_py,
        null_count=null_count,
    )