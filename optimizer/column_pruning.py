"""Column pruning: read and carry only the columns the query uses.

The pass walks the plan top-down. Each node receives a *requirement*: the
set of column references its parent needs from it. From that, each node
works out what it needs from its own children:

    Filter     requirement + columns in the predicate
    Sort       requirement + columns in the sort keys
    Limit      requirement
    Join       requirement + columns in the condition, sent to both sides
    Project    columns read by the expressions the parent uses.
               Expressions nobody uses are dropped.
    Aggregate  columns in the group keys + columns read by the aggregates
               the parent uses. Aggregates nobody uses are dropped.
               Group keys are never dropped, because they define the groups.
    Scan       ``columns`` narrowed to those some reference can resolve to.
               A column only ``pushed_predicate`` reads is dropped too:
               codegen filters the full table before narrowing to
               ``columns`` (confirmed by Person C; pinned by their C1 test
               test_scan_pushed_predicate_may_use_unselected_column).

The root's requirement is ALL, so the query's output never changes.

A column used only in a join condition or a filter predicate is still
required, even if it never reaches the output. The Filter, Sort and Join
rules above make sure such columns are added to the requirement.

Name matching is deliberately conservative, because keeping a column is
always safe and dropping one never is:

* A Scan keeps column ``c`` of table ``T`` for a reference ``(q, c)`` when
  ``q`` is ``None``, or ``q == T``, or ``q`` is not the name of any table in
  the plan (it may be an alias the optimizer can't see through).
* A Project or Aggregate output is matched on its alias alone. The
  qualifier is ignored.

**Project insertion.** When a Join's output carries columns its parent does
not need (typically join keys that are consumed by the join itself), a
narrowing Project is inserted directly above the Join. A Project's outputs
are bare aliases with no table qualifier. So insertion only happens when it
is safe under strict name resolution: the kept column names must be unique,
and no qualified reference above may resolve to them. Otherwise the Join is
left alone. See the B2 report for the request to Person A to pin these
naming rules.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ir.dtype import DType
from ir.expr import ColumnRef
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort

from optimizer.columns import Ref, column_refs, output_columns, scanned_tables, table_schema

ALL = None  # requirement sentinel: the parent needs every output column

# When nothing at all is needed from a Scan (e.g. COUNT(*)), keep one column
# so the row count survives. Prefer the cheapest type to read.
_WIDTH_RANK = {DType.BOOL: 0, DType.INT: 1, DType.DATE: 1, DType.FLOAT: 2, DType.STRING: 3}


class ColumnPruning:
    name = "column_pruning"

    def apply(self, plan: Any, catalog: Any) -> Any:
        return _Pruner(catalog, scanned_tables(plan)).prune(plan, ALL)


class _Pruner:
    def __init__(self, catalog: Any, tables: set[str]):
        self.catalog = catalog
        self.tables = tables

    def prune(self, node: Any, req: set[Ref] | None) -> Any:
        if isinstance(node, Scan):
            return self._scan(node, req)
        if isinstance(node, Project):
            return self._project(node, req)
        if isinstance(node, Aggregate):
            return self._aggregate(node, req)
        if isinstance(node, Join):
            return self._join(node, req)
        if isinstance(node, Filter):
            return self._unary(node, req, node.predicate)
        if isinstance(node, Sort):
            return self._unary(node, req, [expr for expr, _ in node.keys])
        if isinstance(node, Limit):
            return self._unary(node, req, None)
        # A node kind this pass doesn't know: its children must keep everything.
        return _with_children(node, [self.prune(c, ALL) for c in node.children])

    # -- pass-through nodes ------------------------------------------------

    def _unary(self, node: Any, req: set[Ref] | None, uses: Any) -> Any:
        child_req = ALL if req is ALL else req | column_refs(uses)
        child = self._narrow_join(self.prune(node.child, child_req), child_req)
        return _with_children(node, [child])

    def _join(self, node: Join, req: set[Ref] | None) -> Any:
        child_req = ALL if req is ALL else req | column_refs(node.condition)
        left = self._narrow_join(self.prune(node.left, child_req), child_req)
        right = self._narrow_join(self.prune(node.right, child_req), child_req)
        return _with_children(node, [left, right])

    # -- nodes that define new names ---------------------------------------

    def _project(self, node: Project, req: set[Ref] | None) -> Any:
        exprs = node.exprs
        if req is not ALL:
            wanted = {name for _, name in req}
            exprs = [(e, a) for e, a in node.exprs if a in wanted] or [_cheapest_expr(node.exprs)]
        child = self.prune(node.child, column_refs([e for e, _ in exprs]))
        if len(exprs) == len(node.exprs):
            return _with_children(node, [child])
        return dataclasses.replace(node, child=child, exprs=exprs)

    def _aggregate(self, node: Aggregate, req: set[Ref] | None) -> Any:
        aggs = node.aggs
        if req is not ALL:
            wanted = {name for _, name in req}
            aggs = [(c, a) for c, a in node.aggs if a in wanted]
            if not aggs and not node.group_keys and node.aggs:
                # A global aggregate returns exactly one row however many
                # aggregates it computes. Keep one, so it is not left with
                # zero columns.
                aggs = [_cheapest_agg(node.aggs)]
        child_req = column_refs(node.group_keys) | column_refs([call for call, _ in aggs])
        child = self.prune(node.child, child_req)
        if len(aggs) == len(node.aggs):
            return _with_children(node, [child])
        return dataclasses.replace(node, child=child, aggs=aggs)

    # -- leaves --------------------------------------------------------------

    def _scan(self, node: Scan, req: set[Ref] | None) -> Any:
        if req is ALL:
            return node
        schema = table_schema(node, self.catalog)
        if node.columns is not None:
            base = list(node.columns)
        elif schema is not None:
            base = [name for name, _ in schema]
        else:
            return node  # the table's columns can't be listed, so leave it be
        needed = {n for q, n in req if self._may_name_table(q, node.table)}
        keep = [c for c in base if c in needed] or [_narrowest(base, schema)]
        if keep == base:
            return node
        return dataclasses.replace(node, columns=keep)

    # -- Project insertion -------------------------------------------------

    def _narrow_join(self, node: Any, req: set[Ref] | None) -> Any:
        """If ``node`` is a Join producing columns ``req`` doesn't need, put a narrowing Project above it."""
        if req is ALL or not isinstance(node, Join):
            return node
        outputs = output_columns(node, self.catalog)
        if outputs is None:
            return node
        kept = [col for col in outputs if any(self._hits(r, col) for r in req)]
        if len(kept) == len(outputs):
            return node
        kept = kept or outputs[:1]
        names = [name for _, name in kept]
        if len(set(names)) < len(names):
            return node  # would need qualifiers to tell the columns apart
        if any(q is not None and any(self._hits((q, n), col) for col in kept) for q, n in req):
            return node  # a qualified reference above could stop resolving
        exprs = [(ColumnRef(table=q, name=n), n) for q, n in kept]
        return Project(child=node, exprs=exprs)

    # -- name matching -----------------------------------------------------

    def _may_name_table(self, qualifier: str | None, table: str) -> bool:
        return qualifier is None or qualifier == table or qualifier not in self.tables

    def _hits(self, ref: Ref, column: Ref) -> bool:
        """Return True if reference ``ref`` might resolve to output column ``column``."""
        (rq, rn), (cq, cn) = ref, column
        return rn == cn and (rq is None or cq is None or rq == cq or rq not in self.tables)


def _with_children(node: Any, children: list[Any]) -> Any:
    if all(new is old for new, old in zip(children, node.children)):
        return node
    return node.replace_children(tuple(children))


def _narrowest(base: list[str], schema: list[tuple[str, Any]] | None) -> str:
    types = dict(schema) if schema is not None else {}
    return min(base, key=lambda c: _WIDTH_RANK.get(types.get(c), len(_WIDTH_RANK)))


def _cheapest_expr(exprs: list[tuple[Any, str]]) -> tuple[Any, str]:
    return next((pair for pair in exprs if isinstance(pair[0], ColumnRef)), exprs[0])


def _cheapest_agg(aggs: list[tuple[Any, str]]) -> tuple[Any, str]:
    return next((pair for pair in aggs if pair[0].arg is None), aggs[0])
