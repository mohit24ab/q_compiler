"""Minimal columnar table shared by the interpreter and generated code.

Layout: an ordered list of `Column`s. Each column is a numpy array plus an optional
validity mask (True = value present, False = SQL NULL), the Arrow model.

Why a list and not a plain dict: after `orders JOIN customer` there are legitimately two
columns called `id`. Each column therefore also carries the table it came from
(`table`), which is how `ColumnRef(table='orders', name='id')` is resolved.

Physical types per DType:
    INT -> int64    FLOAT -> float64    BOOL -> bool
    STRING -> object (Python str)        DATE -> datetime64[D]
The value stored under a NULL slot is an arbitrary filler; always consult `valid`.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np

from runtime._compat import DType

_NP_DTYPE = {
    "INT": np.int64,
    "FLOAT": np.float64,
    "BOOL": np.bool_,
    "STRING": object,
    "DATE": "datetime64[D]",
}
_FILLER = {
    "INT": 0,
    "FLOAT": 0.0,
    "BOOL": False,
    "STRING": "",
    "DATE": datetime.date(1970, 1, 1),
}


def to_python_value(value, dtype):
    """Normalise a Python value to the canonical form used for `dtype`."""
    if value is None:
        return None
    name = dtype.name
    if name == "DATE":
        if isinstance(value, datetime.datetime):
            return value.date()
        if isinstance(value, datetime.date):
            return value
        if isinstance(value, np.datetime64):
            return value.astype("datetime64[D]").tolist()
        return datetime.date.fromisoformat(str(value))
    if name == "INT":
        return int(value)
    if name == "FLOAT":
        return float(value)
    if name == "BOOL":
        return bool(value)
    return str(value)


@dataclass(frozen=True)
class Column:
    name: str
    dtype: DType
    values: np.ndarray
    valid: np.ndarray | None = None  # None means "no nulls"
    table: str | None = None         # originating table, for qualified lookup

    def __len__(self):
        return len(self.values)

    @property
    def null_count(self) -> int:
        return 0 if self.valid is None else int((~self.valid).sum())

    def to_pylist(self) -> list:
        out = self.values.tolist()
        if self.valid is not None:
            out = [v if ok else None for v, ok in zip(out, self.valid.tolist())]
        return out

    def take(self, indices: np.ndarray) -> "Column":
        valid = None if self.valid is None else self.valid[indices]
        return Column(self.name, self.dtype, self.values[indices], valid, self.table)

    def renamed(self, name: str | None = None, table: str | None = None) -> "Column":
        return Column(name if name is not None else self.name, self.dtype,
                      self.values, self.valid, table)

    @staticmethod
    def from_pylist(name, dtype, values, table=None) -> "Column":
        values = [to_python_value(v, dtype) for v in values]
        mask = np.array([v is not None for v in values], dtype=bool)
        filler = _FILLER[dtype.name]
        filled = [filler if v is None else v for v in values]
        arr = np.array(filled, dtype=_NP_DTYPE[dtype.name])
        if arr.ndim != 1:  # e.g. empty list, or numpy being clever with sequences
            arr = np.empty(len(filled), dtype=_NP_DTYPE[dtype.name])
            arr[:] = filled
        return Column(name, dtype, arr, None if mask.all() else mask, table)


def find_columns(columns, name: str, table: str | None = None) -> list[int]:
    """Name resolution shared by the interpreter and the code generator.

    `columns` is any sequence of objects with `.name` and `.table`. Returns the indices
    matching `name` (and `table`, if given). A qualified reference falls back to an
    unqualified column of that name, because a Project drops qualifiers.
    """
    hits = [i for i, c in enumerate(columns)
            if c.name == name and (table is None or c.table == table)]
    if not hits and table is not None:
        hits = [i for i, c in enumerate(columns) if c.name == name and c.table is None]
    return hits


class Table:
    def __init__(self, columns: list[Column], num_rows: int | None = None):
        """`num_rows` matters only when there are no columns: such a table still has
        rows (a Scan pruned to no columns under COUNT(*)), and nothing else can say how
        many. When given alongside columns it must agree with them."""
        lengths = {len(c) for c in columns}
        if num_rows is not None:
            lengths.add(int(num_rows))
        if len(lengths) > 1:
            raise ValueError(f"columns have differing lengths: {sorted(lengths)}"
                             + (f" (num_rows={num_rows})" if num_rows is not None else ""))
        self.columns = list(columns)
        self._num_rows = lengths.pop() if lengths else 0

    # ---------------------------------------------------------------- basics
    @property
    def num_rows(self) -> int:
        return self._num_rows

    def __len__(self):
        return self._num_rows

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def schema(self) -> list[tuple[str, DType]]:
        return [(c.name, c.dtype) for c in self.columns]

    def find(self, name: str, table: str | None = None) -> list[int]:
        """Indices of columns matching `name` (and `table`, if given)."""
        return find_columns(self.columns, name, table)

    def column(self, name: str, table: str | None = None) -> Column:
        hits = self.find(name, table)
        label = f"{table}.{name}" if table else name
        if not hits:
            raise KeyError(f"no column {label!r}; have {self.qualified_names()}")
        if len(hits) > 1:
            raise KeyError(f"ambiguous column {label!r}; have {self.qualified_names()}")
        return self.columns[hits[0]]

    def qualified_names(self) -> list[str]:
        return [f"{c.table}.{c.name}" if c.table else c.name for c in self.columns]

    # ---------------------------------------------------------------- reshaping
    def select(self, names: list[str]) -> "Table":
        return Table([self.column(n) for n in names], self._num_rows)

    def take(self, indices) -> "Table":
        idx = np.asarray(indices, dtype=np.int64)
        return Table([c.take(idx) for c in self.columns], len(idx))

    def filter(self, mask: np.ndarray) -> "Table":
        return self.take(np.flatnonzero(mask))

    def slice(self, start: int, stop: int | None = None) -> "Table":
        stop = self._num_rows if stop is None else min(stop, self._num_rows)
        return self.take(np.arange(max(start, 0), max(stop, 0)))

    @staticmethod
    def concat(tables: list["Table"]) -> "Table":
        if not tables:
            raise ValueError("concat of zero tables")
        first = tables[0]
        for t in tables[1:]:
            if t.schema != first.schema:
                raise ValueError(f"schema mismatch: {first.schema} vs {t.schema}")
        cols = []
        for i, c in enumerate(first.columns):
            parts = [t.columns[i] for t in tables]
            values = np.concatenate([p.values for p in parts])
            if all(p.valid is None for p in parts):
                valid = None
            else:
                valid = np.concatenate([
                    np.ones(len(p), bool) if p.valid is None else p.valid for p in parts
                ])
            cols.append(Column(c.name, c.dtype, values, valid, c.table))
        return Table(cols, sum(t.num_rows for t in tables))

    def with_table_name(self, table: str) -> "Table":
        return Table([c.renamed(table=table) for c in self.columns], self._num_rows)

    # ---------------------------------------------------------------- row view
    def to_rows(self) -> list[tuple]:
        """Rows as tuples of Python values (None for NULL, datetime.date for DATE)."""
        cols = [c.to_pylist() for c in self.columns]
        return list(zip(*cols)) if cols else [()] * self._num_rows

    def to_pydict(self) -> dict[str, list]:
        return {c.name: c.to_pylist() for c in self.columns}

    @staticmethod
    def from_rows(fields: list[tuple[str, DType, str | None]], rows: list) -> "Table":
        """Build from (name, dtype, table) field specs and an iterable of row tuples."""
        rows = list(rows)
        cols = []
        for i, (name, dtype, table) in enumerate(fields):
            cols.append(Column.from_pylist(name, dtype, [r[i] for r in rows], table))
        return Table(cols, len(rows))

    @staticmethod
    def from_pydict(data: dict[str, list], schema: list[tuple[str, DType]],
                    table: str | None = None) -> "Table":
        return Table([Column.from_pylist(n, t, data[n], table) for n, t in schema])

    @staticmethod
    def from_arrow(arrow_table, table: str | None = None) -> "Table":
        import pyarrow as pa
        cols = []
        for name, chunked in zip(arrow_table.column_names, arrow_table.columns):
            arr = chunked.combine_chunks() if isinstance(chunked, pa.ChunkedArray) else chunked
            dtype = _dtype_from_arrow(arr.type)
            valid = None
            if arr.null_count:
                valid = arr.is_valid().to_numpy(zero_copy_only=False)
                arr = arr.fill_null(_FILLER[dtype.name])
            values = arr.to_numpy(zero_copy_only=False).astype(_NP_DTYPE[dtype.name])
            cols.append(Column(name, dtype, values, valid, table))
        return Table(cols, arrow_table.num_rows)

    # ---------------------------------------------------------------- display
    def format(self, max_rows: int = 20) -> str:
        # The names the query gave its output; table-qualified only when they would clash.
        names = self.column_names
        if len(set(names)) < len(names):
            names = self.qualified_names()
        def cell(v):
            if v is None:
                return "NULL"
            return format(v, ".10g") if isinstance(v, float) else str(v)  # no 0.1+0.2 noise
        rows = [[cell(v) for v in r] for r in self.to_rows()[:max_rows]]
        widths = [max([len(n)] + [len(r[i]) for r in rows]) for i, n in enumerate(names)]
        line = lambda cells: " | ".join(c.ljust(w) for c, w in zip(cells, widths))
        out = [line(names), "-+-".join("-" * w for w in widths)]
        out += [line(r) for r in rows]
        if self._num_rows > max_rows:
            out.append(f"... ({self._num_rows} rows total)")
        return "\n".join(out)

    def __repr__(self):
        return f"Table({self._num_rows} rows, {self.qualified_names()})"


def _dtype_from_arrow(t) -> DType:
    import pyarrow.types as pt
    if pt.is_integer(t):
        return DType.INT
    if pt.is_floating(t) or pt.is_decimal(t):
        return DType.FLOAT
    if pt.is_boolean(t):
        return DType.BOOL
    if pt.is_string(t) or pt.is_large_string(t):
        return DType.STRING
    if pt.is_date(t):
        return DType.DATE
    raise TypeError(f"unsupported Arrow type {t}")
