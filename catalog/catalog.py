from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from catalog.loader import load_csv, load_parquet, pyarrow_type_to_dtype
from catalog.stats import ColumnStats, compute_column_stats
from ir.dtype import DType


class Catalog:
    """Manages physical table registrations, schema resolution, and column statistics."""

    def __init__(self) -> None:
        self._tables: dict[str, pa.Table] = {}
        self._schemas: dict[str, list[tuple[str, DType]]] = {}
        self._stats_cache: dict[str, dict[str, ColumnStats]] = {}

    def register_table(self, name: str, table: pa.Table) -> None:
        """Registers an in-memory PyArrow Table in the catalog."""
        self._tables[name] = table
        schema = [
            (field.name, pyarrow_type_to_dtype(field.type))
            for field in table.schema
        ]
        self._schemas[name] = schema
        self._stats_cache[name] = {}

    def register_csv(self, name: str, path: str | Path) -> None:
        """Loads and registers a CSV file in the catalog."""
        table = load_csv(path)
        self.register_table(name, table)

    def register_parquet(self, name: str, path: str | Path) -> None:
        """Loads and registers a Parquet file in the catalog."""
        table = load_parquet(path)
        self.register_table(name, table)

    def schema(self, table: str) -> list[tuple[str, DType]]:
        """Returns table schema as a list of (column_name, DType) tuples."""
        if table not in self._schemas:
            raise KeyError(f"Table '{table}' not found in catalog.")
        return list(self._schemas[table])

    def row_count(self, table: str) -> int:
        """Returns the total row count for a table."""
        if table not in self._tables:
            raise KeyError(f"Table '{table}' not found in catalog.")
        return self._tables[table].num_rows

    def stats(self, table: str, column: str) -> ColumnStats:
        """Returns column statistics (NDV, min, max, null_count) for a table column."""
        if table not in self._tables:
            raise KeyError(f"Table '{table}' not found in catalog.")

        pa_table = self._tables[table]
        if column not in pa_table.column_names:
            raise KeyError(f"Column '{column}' not found in table '{table}'.")

        if column not in self._stats_cache[table]:
            col_data = pa_table.column(column)
            self._stats_cache[table][column] = compute_column_stats(col_data)

        return self._stats_cache[table][column]