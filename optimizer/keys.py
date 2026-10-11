"""Which sets of columns uniquely identify a row.

A *key* of a relation is a set of its columns that no two of its rows agree
on. Keys are what make an ``ORDER BY`` **total**: if the sort keys contain a
key of the sorted relation, no two rows tie, so the sorted order is fully
determined and so is the top-n a ``LIMIT`` above it keeps. That is the
property join reordering needs before it may reorder under a Sort -- see
``optimizer/join_reordering.py``.

The analysis is deliberately one-sided: ``unique_keys`` returns keys it can
*prove*, and ``None`` when it cannot see far enough. A missed key costs a
rewrite; a wrongly claimed key costs a wrong answer, so every rule here only
fires on a case it can justify.

The rules:

  Scan      a column is a key when the catalog says it has as many distinct
            values as the table has rows, and no nulls. Two nulls are not
            distinguishable by ORDER BY, so a nullable column is never a key
            here even if its non-null values happen to be distinct.
  Filter    a subset of distinct rows is still distinct, so keys survive.
  Sort      reordering rows does not make two of them equal.
  Limit     likewise: keeping some of the rows keeps them distinct.
  Project   a key survives if every one of its columns is passed through
            under some output name. A computed expression is not tracked.
  Aggregate the group keys are a key of the output: one row per group. With
            no group keys the output is a single row, whose key is the empty
            set -- which every ordering trivially contains.
  Join      ``kl | kr`` is always a key of an inner join. More usefully, if
            the join condition equates a whole key of the right side, each
            left row matches at most one right row, so the left side's keys
            survive on their own; and symmetrically. This is the FK -> PK
            case, and it is what carries a fact table's key up through its
            dimension joins.

Outer joins are not handled: they introduce null-extended rows, so they get
``None``. Anything else unrecognised gets ``None`` too.
"""

from __future__ import annotations

from typing import Any

from ir.expr import BinaryOp, ColumnRef
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort

from optimizer.columns import Ref, output_columns, resolves, table_schema
from optimizer.expressions import split_conjuncts

#: Returned instead of a key list when the shape of a node defeats the analysis.
UNKNOWN = None


def unique_keys(node: Any, catalog: Any) -> list[frozenset[Ref]] | None:
    """Return column sets that are known to be unique in ``node``'s output.

    ``None`` means "cannot tell", which callers must treat as "assume there
    are duplicates". An empty list means the analysis ran and found no key.
    """
    if isinstance(node, Scan):
        return _scan_keys(node, catalog)
    if isinstance(node, (Filter, Sort, Limit)):
        return unique_keys(node.child, catalog)
    if isinstance(node, Project):
        return _project_keys(node, catalog)
    if isinstance(node, Aggregate):
        return _aggregate_keys(node)
    if isinstance(node, Join):
        return _join_keys(node, catalog)
    return UNKNOWN


def ordering_is_total(sort: Sort, catalog: Any) -> bool:
    """True if ``sort``'s keys contain a key of the relation it sorts.

    Then no two rows tie on the ordering, so the sorted result is one
    specific sequence rather than one of several valid ones.
    """
    keys = unique_keys(sort.child, catalog)
    if keys is None:
        return False
    sort_refs = [(k.table, k.name) for k, _desc in sort.keys if isinstance(k, ColumnRef)]
    return any(_covered(key, sort_refs) for key in keys)


def _covered(key: frozenset[Ref], sort_refs: list[Ref]) -> bool:
    """Every column of ``key`` is named among the sort keys."""
    return all(any(resolves(ref, column) for ref in sort_refs) for column in key)


# -------------------------------------------------------------------- scan


def _scan_keys(scan: Scan, catalog: Any) -> list[frozenset[Ref]] | None:
    schema = table_schema(scan, catalog)
    if schema is None or catalog is None:
        return UNKNOWN
    try:
        rows = catalog.row_count(scan.table)
    except KeyError:
        return UNKNOWN
    if rows <= 0:
        return UNKNOWN  # an empty table says nothing about the live one
    visible = scan.columns if scan.columns is not None else [name for name, _ in schema]
    keys = []
    for name in visible:
        try:
            stats = catalog.stats(scan.table, name)
        except KeyError:
            continue
        if stats.ndv == rows and stats.null_count == 0:
            keys.append(frozenset({(scan.table, name)}))
    return keys


# ----------------------------------------------------------------- project


def _project_keys(project: Project, catalog: Any) -> list[frozenset[Ref]] | None:
    child_keys = unique_keys(project.child, catalog)
    if child_keys is None:
        return UNKNOWN
    # child column -> the name it is published under, and whether the alias
    # left the column's own name intact. Only plain pass-through columns are
    # tracked: a computed expression may collapse distinct inputs.
    passed: list[tuple[Ref, Ref, bool]] = [
        ((expr.table, expr.name), (None, alias), alias == expr.name)
        for expr, alias in project.exprs
        if isinstance(expr, ColumnRef)
    ]
    out = []
    for key in child_keys:
        mapped, keeps_own_name = set(), True
        for column in key:
            match = next((p for p in passed if resolves(p[0], column)), None)
            if match is None:
                break
            _, published, same_name = match
            mapped.add(published)
            keeps_own_name = keeps_own_name and same_name
        else:
            out.append(frozenset(mapped))
            if keeps_own_name:
                # ``SELECT lineitem.id AS id ... ORDER BY lineitem.id`` is legal:
                # a pass-through column keeps its qualified name above the
                # Project, so the sort keys may still spell it that way.
                out.append(key)
    return out


# --------------------------------------------------------------- aggregate


def _aggregate_keys(agg: Aggregate) -> list[frozenset[Ref]] | None:
    if not all(isinstance(k, ColumnRef) for k in agg.group_keys):
        return UNKNOWN  # a computed group key has no defined output name
    # One row per distinct combination of group keys -- including the empty
    # combination, which is the single row a global aggregate returns.
    return [frozenset((k.table, k.name) for k in agg.group_keys)]


# -------------------------------------------------------------------- join


def _join_keys(join: Join, catalog: Any) -> list[frozenset[Ref]] | None:
    if join.kind != "inner":
        return UNKNOWN  # null-extended rows are outside this analysis
    left = unique_keys(join.left, catalog)
    right = unique_keys(join.right, catalog)
    if left is None or right is None:
        return UNKNOWN
    left_out = output_columns(join.left, catalog)
    right_out = output_columns(join.right, catalog)
    if left_out is None or right_out is None:
        return UNKNOWN

    equated_left, equated_right = _equated(join.condition, left_out, right_out)

    keys = {kl | kr for kl in left for kr in right}
    # Each left row matches at most one right row, so a left key stays unique.
    if any(_covered(k, equated_right) for k in right):
        keys.update(left)
    if any(_covered(k, equated_left) for k in left):
        keys.update(right)
    return _minimal(keys)


def _equated(condition: Any, left_out: list[Ref], right_out: list[Ref]) -> tuple[list[Ref], list[Ref]]:
    """Columns joined by a plain ``a = b`` conjunct, split by the side they belong to."""
    left_side, right_side = [], []
    for conjunct in split_conjuncts(condition):
        if not (isinstance(conjunct, BinaryOp) and conjunct.op == "="):
            continue
        if not (isinstance(conjunct.left, ColumnRef) and isinstance(conjunct.right, ColumnRef)):
            continue
        for ref in (conjunct.left, conjunct.right):
            column = (ref.table, ref.name)
            in_left = any(resolves(column, c) for c in left_out)
            in_right = any(resolves(column, c) for c in right_out)
            if in_left and not in_right:  # ambiguous names are claimed by neither side
                left_side.append(column)
            elif in_right and not in_left:
                right_side.append(column)
    return left_side, right_side


def _minimal(keys: set[frozenset[Ref]]) -> list[frozenset[Ref]]:
    """Drop any key that is a superset of another: it adds nothing."""
    return [k for k in keys if not any(other < k for other in keys)]
