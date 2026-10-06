"""Predicate pushdown: filter rows as early as possible.

Each Filter predicate is split on AND into conjuncts. Each conjunct is then
pushed down as far as its column references allow. Conjuncts that reach a
Scan become part of ``Scan.pushed_predicate``, so codegen can filter during
the read. A conjunct that cannot go further stays in a Filter at the
highest node it cannot cross.

Rules, per node the conjunct is pushed into:

  Filter     Its own conjuncts join the pushed set (inner ones first).
  Sort       Always crossed. Filtering does not change the order of the
             rows that remain.
  Project    Crossed if every column the conjunct reads is an unqualified
             reference to a uniquely named output. The references are
             rewritten as the expressions behind them: ``value > 10`` over
             ``qty * amount AS value`` becomes ``qty * amount > 10``.
  Aggregate  Crossed if the conjunct reads only group keys, and there is at
             least one group key. A predicate on an aggregate result (HAVING)
             stays. A global aggregate returns one row even for empty input,
             so not even a constant predicate may cross it.
  Join       INNER: the ON conjuncts and the incoming conjuncts are pooled.
             Those reading only the left side go left, only the right side
             go right, and the rest form the join condition.
             LEFT: incoming conjuncts reading only the left (preserved)
             side go left. ON conjuncts reading only the right
             (null-producing) side go right. Everything else stays where it
             is, because pushing a WHERE conjunct into the null-producing
             side changes the answer: it filters rows *before* they are
             NULL-extended, rather than removing the extended rows after.
             The exception: if an incoming conjunct is null-rejecting, it
             can never be TRUE for a NULL-extended row, and the LEFT join is
             equivalent to an INNER join. The pass converts it to INNER, and
             the INNER rules apply.
  Limit      Never crossed. Filtering before the limit selects different
             rows.
  Unknown    Never crossed. The node's children are still optimized.

A conjunct is routed to a join side by strict name resolution against
each side's output columns. If a conjunct cannot be routed (a column
appears on both sides, or on neither, or a side's columns are unknown), it
is not pushed into either side.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ir.expr import ColumnRef
from ir.nodes import Aggregate, Filter, Join, Project, Scan, Sort

from optimizer.columns import Ref, column_refs, output_columns, table_schema
from optimizer.expressions import TRUE, can_be_true, conjoin, rebuild, split_conjuncts, substitute

LEFT, RIGHT = "L", "R"


class PredicatePushdown:
    name = "predicate_pushdown"

    def apply(self, plan: Any, catalog: Any) -> Any:
        return _Pusher(catalog).push(plan, [])


class _Pusher:
    def __init__(self, catalog: Any):
        self.catalog = catalog

    def push(self, node: Any, preds: list[Any]) -> Any:
        """Return a plan equivalent to ``Filter(node, AND(preds))``, with the predicates pushed as deep as allowed."""
        if isinstance(node, Filter):
            return self.push(node.child, split_conjuncts(node.predicate) + preds)
        if isinstance(node, Scan):
            return self._scan(node, preds)
        if isinstance(node, Sort):
            return _with_children(node, [self.push(node.child, preds)])
        if isinstance(node, Project):
            return self._project(node, preds)
        if isinstance(node, Aggregate):
            return self._aggregate(node, preds)
        if isinstance(node, Join):
            return self._join(node, preds)
        # Limit, and any node kind this pass doesn't know: a barrier.
        return _filter(_with_children(node, [self.push(c, []) for c in node.children]), preds)

    # -- leaves --------------------------------------------------------------

    def _scan(self, node: Scan, preds: list[Any]) -> Any:
        if not preds:
            return node
        available = self._scan_columns(node)
        land, keep = _partition(
            preds, lambda p: available is None or {n for _, n in column_refs(p)} <= available
        )
        if land:
            pushed = conjoin(split_conjuncts(node.pushed_predicate) + land)
            node = dataclasses.replace(node, pushed_predicate=pushed)
        return _filter(node, keep)

    def _scan_columns(self, node: Scan) -> set[str] | None:
        if node.columns is not None:
            return set(node.columns)
        schema = table_schema(node, self.catalog)
        return None if schema is None else {name for name, _ in schema}

    # -- nodes that define names ---------------------------------------------

    def _project(self, node: Project, preds: list[Any]) -> Any:
        aliases = [alias for _, alias in node.exprs]
        mapping = {(None, a): e for e, a in node.exprs if aliases.count(a) == 1}
        down, stay = _partition(preds, lambda p: column_refs(p) <= mapping.keys())
        child = self.push(node.child, [substitute(p, mapping) for p in down])
        return _filter(_with_children(node, [child]), stay)

    def _aggregate(self, node: Aggregate, preds: list[Any]) -> Any:
        outputs = [k.name for k in node.group_keys if isinstance(k, ColumnRef)] + [a for _, a in node.aggs]
        keys: dict[Ref, Any] = {}
        for k in node.group_keys:
            if isinstance(k, ColumnRef) and outputs.count(k.name) == 1:
                keys[(None, k.name)] = k
                if k.table is not None:
                    keys[(k.table, k.name)] = k
        down, stay = _partition(
            preds, lambda p: bool(node.group_keys) and column_refs(p) <= keys.keys()
        )
        child = self.push(node.child, [substitute(p, keys) for p in down])
        return _filter(_with_children(node, [child]), stay)

    # -- joins ---------------------------------------------------------------

    def _join(self, node: Join, preds: list[Any]) -> Any:
        lcols = output_columns(node.left, self.catalog)
        rcols = output_columns(node.right, self.catalog)

        def sides(p):
            return _sides(p, lcols, rcols)

        if node.kind == "left" and rcols is not None and lcols is not None:
            def right_only(ref: ColumnRef) -> bool:
                r = (ref.table, ref.name)
                return any(_resolves(r, c) for c in rcols) and not any(_resolves(r, c) for c in lcols)

            if any(not can_be_true(p, right_only) for p in preds):
                node = dataclasses.replace(node, kind="inner")

        on = split_conjuncts(node.condition)
        if node.kind == "inner":
            to_left, to_right, cond, above = [], [], [], []
            for p in on + preds:
                s = sides(p)
                if s is None or s == {LEFT, RIGHT}:
                    cond.append(p)
                elif s == {RIGHT}:
                    to_right.append(p)
                else:  # left-only, or reads no columns at all
                    to_left.append(p)
        elif node.kind == "left":
            to_left, above = _partition(preds, lambda p: sides(p) in ({LEFT}, set()))
            to_right, cond = _partition(on, lambda p: sides(p) in ({RIGHT}, set()))
        else:  # a join kind this pass doesn't know: leave its predicates alone
            to_left, to_right, cond, above = [], [], on, preds

        left = self.push(node.left, to_left)
        right = self.push(node.right, to_right)
        # With every ON conjunct pushed down, the join keeps a TRUE condition.
        condition = rebuild(cond, node.condition) or (node.condition if node.condition == TRUE else TRUE)
        if left is not node.left or right is not node.right or condition is not node.condition:
            node = dataclasses.replace(node, left=left, right=right, condition=condition)
        return _filter(node, above)


def _sides(pred: Any, lcols: list[Ref] | None, rcols: list[Ref] | None) -> set[str] | None:
    """Return the join sides ``pred`` reads (empty for constants), or None if it can't be routed."""
    if lcols is None or rcols is None:
        return None
    sides = set()
    for ref in column_refs(pred):
        in_left = any(_resolves(ref, c) for c in lcols)
        in_right = any(_resolves(ref, c) for c in rcols)
        if in_left == in_right:  # ambiguous, or not found on either side
            return None
        sides.add(LEFT if in_left else RIGHT)
    return sides


def _resolves(ref: Ref, column: Ref) -> bool:
    """Apply strict name resolution, the same rule as the reference evaluator."""
    (rq, rn), (cq, cn) = ref, column
    return rn == cn and (rq is None or rq == cq)


def _partition(items: list[Any], keep_left) -> tuple[list[Any], list[Any]]:
    yes, no = [], []
    for item in items:
        (yes if keep_left(item) else no).append(item)
    return yes, no


def _filter(node: Any, preds: list[Any]) -> Any:
    return Filter(child=node, predicate=conjoin(preds)) if preds else node


def _with_children(node: Any, children: list[Any]) -> Any:
    if all(new is old for new, old in zip(children, node.children)):
        return node
    return node.replace_children(tuple(children))
