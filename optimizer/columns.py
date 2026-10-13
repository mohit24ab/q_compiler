"""Column analysis shared by the rewrite passes.

The functions here answer two questions: which columns an expression reads,
and which columns a plan node produces.

A column is identified by a ``Ref``: ``(qualifier, name)``. The qualifier
is the table a column comes from, or ``None`` if it is unknown (as for
the outputs of a Project or an aggregate).
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
from typing import Any

from ir.expr import AggCall, ColumnRef
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort

Ref = tuple[str | None, str]

_LEAVES = (
    str, int, float, bool, bytes, type(None), enum.Enum,
    datetime.date, datetime.datetime, decimal.Decimal,
)


def column_refs(expr: Any) -> set[Ref]:
    """Return every column an expression (or a list/tuple of expressions) reads.

    The walk is generic over dataclass fields, so expression types added to
    the IR later are covered automatically. If the walk meets an object it
    cannot see inside, it raises ``TypeError`` rather than silently missing a
    reference. A missed reference is how pruning would drop a column the
    query needs.
    """
    found: set[Ref] = set()
    _collect(expr, found)
    return found


def _collect(x: Any, found: set[Ref]) -> None:
    if isinstance(x, ColumnRef):
        found.add((x.table, x.name))
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            _collect(getattr(x, f.name), found)
    elif isinstance(x, (list, tuple)):
        for item in x:
            _collect(item, found)
    elif not isinstance(x, _LEAVES):
        raise TypeError(f"cannot find column references inside {type(x).__name__}: {x!r}")


def resolves(ref: Ref, column: Ref) -> bool:
    """Strict name resolution, the reference evaluator's rule: the names match, and so do the qualifiers if ``ref`` has one."""
    (rq, rn), (cq, cn) = ref, column
    return rn == cn and (rq is None or rq == cq)

# The qualifier of a Ref that names an aggregate call rather than a column.
AGGREGATE = "<aggregate>"


def agg_calls(expr: Any) -> list[Any]:
    """Return every aggregate call (``AggCall``) inside an expression or list of expressions.

    Above an Aggregate, a HAVING predicate or ORDER BY key can name one of the
    Aggregate's results by repeating its call: ``HAVING COUNT(*) > 3`` holds
    ``AggCall("count")``, not a reference to the result's alias.
    """
    found: list[Any] = []
    _collect_calls(expr, found)
    return found


def _collect_calls(x: Any, found: list[Any]) -> None:
    if isinstance(x, AggCall):
        found.append(x)
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        for f in dataclasses.fields(x):
            _collect_calls(getattr(x, f.name), found)
    elif isinstance(x, (list, tuple)):
        for item in x:
            _collect_calls(item, found)


def aggregate_ref(call: Any) -> Ref:
    """The Ref under which an expression above an Aggregate names that Aggregate's result for ``call``."""
    return (AGGREGATE, repr(call))


def requirements(expr: Any) -> set[Ref]:
    """What an expression needs from below: the columns it reads, and the aggregate results it names."""
    return column_refs(expr) | {aggregate_ref(c) for c in agg_calls(expr)}


def scanned_tables(plan: Any) -> set[str]:
    """Return the names of every table the plan scans."""
    if isinstance(plan, Scan):
        return {plan.table}
    return set().union(*(scanned_tables(c) for c in plan.children))


def table_schema(scan: Scan, catalog: Any) -> list[tuple[str, Any]] | None:
    """Return the full schema of the table a Scan reads, or ``None`` if it is unknown.

    ``scan.table_schema`` is used when the binder filled it in; otherwise the
    catalog is asked.
    """
    if getattr(scan, "table_schema", None) is not None:
        return list(scan.table_schema)
    if catalog is None:
        return None
    try:
        return list(catalog.schema(scan.table))
    except KeyError:
        return None


def output_columns(node: Any, catalog: Any) -> list[Ref] | None:
    """Return the columns a node produces, in order, or ``None`` if that can't be determined."""
    if isinstance(node, Scan):
        if node.columns is not None:
            return [(node.table, c) for c in node.columns]
        schema = table_schema(node, catalog)
        return None if schema is None else [(node.table, name) for name, _ in schema]
    if isinstance(node, (Filter, Sort, Limit)):
        return output_columns(node.child, catalog)
    if isinstance(node, Project):
        return [(None, alias) for _, alias in node.exprs]
    if isinstance(node, Join):
        left = output_columns(node.left, catalog)
        right = output_columns(node.right, catalog)
        return None if left is None or right is None else left + right
    if isinstance(node, Aggregate):
        if not all(isinstance(k, ColumnRef) for k in node.group_keys):
            return None  # a computed group key's output name is not defined by the contract
        keys = [(k.table, k.name) for k in node.group_keys]
        return keys + [(None, alias) for _, alias in node.aggs]
    return None
