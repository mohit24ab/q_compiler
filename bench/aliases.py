"""Bridge: rewrite alias-qualified column references to table-qualified ones.

Since main @ a9a3048 the binder resolves aliases itself, and this is a no-op on every
plan it produces; the runner and demo keep calling it as a guard, so that alias
qualifiers, should they come back, are resolved or rejected rather than run.

For `FROM orders o JOIN customer c ON o.cust_id = c.id`, the binder (frontend/binder.py)
used to emit `ColumnRef(table="o", name="cust_id")` under a `Scan(table="orders")`.
The IR has no alias field on Scan, so nothing downstream can tell that `o` means
`orders`: the interpreter and the generated code fail with "unknown column 'o.cust_id'",
and an optimizer can't tell which join input a predicate belongs to. 9 of the 20 golden
queries (every join: q07-q12, q18-q20) are affected.

Until the binder emits table names, `resolve_aliases(plan, catalog)` works out each alias
from the plan itself: an unknown qualifier `q` means the one scanned table whose schema
has every column referenced as `q.<column>`. If no table, or more than one, fits, it
raises instead of guessing. It returns the plan unchanged when every qualifier is
already a scanned table's name, so it becomes a no-op once the binder is fixed.

Self-joins (`orders a JOIN orders b`) can't be expressed without an alias on Scan at
all; there are none in the suite.
"""
from __future__ import annotations

import dataclasses

from runtime._compat import ColumnRef
from runtime.expr_eval import node_kind


class AliasError(ValueError):
    pass


def resolve_aliases(plan, catalog=None):
    mapping = alias_map(plan, catalog)
    return _rebuild(plan, mapping) if mapping else plan


def alias_map(plan, catalog=None) -> dict[str, str]:
    """{alias: table} for every qualifier that isn't the name of a scanned table."""
    tables: dict[str, set[str]] = {}
    for scan in _scans(plan):
        schema = scan.table_schema
        if schema is None:
            if catalog is None:
                raise AliasError(f"no schema for {scan.table!r}: pass a catalog")
            schema = catalog.schema(scan.table)
        tables[scan.table] = {name for name, _ in schema}

    used: dict[str, set[str]] = {}
    for ref in _refs(plan):
        if ref.table is not None and ref.table not in tables:
            used.setdefault(ref.table, set()).add(ref.name)

    mapping = {}
    for alias, names in sorted(used.items()):
        fits = sorted(t for t, cols in tables.items() if names <= cols)
        if len(fits) != 1:
            raise AliasError(f"can't tell which table {alias!r} means: it is used with "
                             f"columns {sorted(names)}, which fit {fits or 'no scanned table'}")
        mapping[alias] = fits[0]
    return mapping


def _scans(plan):
    if node_kind(plan) == "Scan":
        yield plan
    for child in plan.children:
        yield from _scans(child)


def _refs(value):
    if isinstance(value, ColumnRef):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from _refs(getattr(value, f.name))
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _refs(v)


def _rebuild(value, mapping):
    """A copy of `value` (a plan, an expression, or a list/tuple of them) with every
    ColumnRef's qualifier passed through `mapping`. Plan nodes are frozen dataclasses."""
    if isinstance(value, ColumnRef):
        return ColumnRef(table=mapping.get(value.table, value.table), name=value.name)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{f.name: _rebuild(getattr(value, f.name), mapping)
                                             for f in dataclasses.fields(value) if f.init})
    if isinstance(value, list):
        return [_rebuild(v, mapping) for v in value]
    if isinstance(value, tuple):
        return tuple(_rebuild(v, mapping) for v in value)
    return value
