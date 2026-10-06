"""Small runtime helpers called by generated code.

These move data in and out of `Table`s and do the few things numpy has no single
operator for. All query logic itself — masks, arithmetic, three-valued logic — is
emitted inline by the code generator, so it can be read in the generated source.

Convention in generated code: every column is two arrays, `values` and `ok`
(True where the value is present, False where it is SQL NULL).
"""
from __future__ import annotations

import re

import numpy as np

from runtime.table import _NP_DTYPE, Column, Table


def as_table(source) -> Table:
    """Accept a runtime Table or a pyarrow.Table."""
    return source if isinstance(source, Table) else Table.from_arrow(source)


def read_column(table: Table, name: str, rows=None) -> tuple[np.ndarray, np.ndarray]:
    """(values, ok) for one column of a base table, optionally only `rows` (a bool mask)."""
    col = table.column(name)
    values = col.values
    ok = np.ones(len(values), dtype=bool) if col.valid is None else col.valid
    if rows is not None:
        values, ok = values[rows], ok[rows]
    return values, ok


def build_table(columns) -> Table:
    """Assemble the result from (name, dtype, table, values, ok_or_None) tuples."""
    out = []
    for name, dtype, table, values, ok in columns:
        values = np.asarray(values, dtype=_NP_DTYPE[dtype.name])
        valid = None
        if ok is not None:
            ok = np.asarray(ok, dtype=bool)
            valid = None if ok.all() else ok
        out.append(Column(name, dtype, values, valid, table))
    return Table(out)


def like(values, pattern: str) -> np.ndarray:
    """SQL LIKE over an array of strings (% = any run, _ = any one character)."""
    regex = re.compile("".join(
        ".*" if ch == "%" else "." if ch == "_" else re.escape(ch) for ch in pattern),
        flags=re.DOTALL)
    return np.fromiter((regex.fullmatch(v) is not None for v in values),
                       dtype=bool, count=len(values))


def take_or_null(values: np.ndarray, ok, idx: np.ndarray):
    """values[idx] where idx == -1 means "no row" (LEFT JOIN padding) and yields NULL."""
    present = idx >= 0
    if len(values) == 0:  # nothing to take from: every slot is padding
        filler = "" if values.dtype == object else 0  # NULL slots must stay comparable
        return np.full(len(idx), filler, dtype=values.dtype), np.zeros(len(idx), dtype=bool)
    safe = np.where(present, idx, 0)
    ok_taken = present if ok is None else (ok[safe] & present)
    return values[safe], ok_taken
