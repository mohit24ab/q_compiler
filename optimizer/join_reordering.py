"""Join reordering: pick the cheapest order for every tree of inner joins.

A *join region* is a maximal tree of INNER joins. Whatever hangs below it
(a Scan, Filter, Aggregate, Project, LEFT join, ...) is a *leaf*, and is
treated as one relation; the region's *conjuncts* are the conjuncts of all
its join conditions. Every join tree over the same leaves returns the same
rows, provided each conjunct is evaluated once all the leaves it reads have
been joined: inner join is commutative and associative, and a conjunct of an
inner join's condition is just a filter over the join's output.

The join graph has one vertex per leaf, and one edge per conjunct that reads
two or more leaves. The plan for a set of leaves is built as follows.

  Up to 8 leaves: dynamic programming over connected subsets (Selinger's
  method, extended to bushy trees). For every connected set of leaves, in
  order of size, keep the cheapest plan that joins exactly that set: the
  cheapest of best(S1) JOIN best(S2) over every split of the set into two
  connected halves that a conjunct links. The cost is optimizer/cost.py's,
  so it accounts for intermediate result sizes, hash build and probe, and
  output.

  More than 8 leaves: greedy. Repeatedly join the two plans whose join has
  the smallest estimated output, among the pairs a conjunct links.

  Cross products are only considered when the graph forces them: when it
  has more than one connected component, each component is planned on its
  own, and the components are then joined to each other without a condition.
  A component that only a conjunct over three or more leaves holds together
  (``c.id + o.id = l.order_id``) has no split into linked halves, so the
  dynamic program plans it again with cross products allowed.

Each conjunct goes into the condition of the lowest join that sees every
leaf it reads. A conjunct that reads one leaf becomes a Filter on that
leaf (pushdown normally leaves none), and a constant one stays on the top
join. The input with more estimated rows goes on the left: codegen builds
its hash table on the smaller side whichever it is, so this only makes the
printed plans read consistently.

The pass leaves a region alone when:

  * it has fewer than 3 leaves: two relations have only one join order;
  * a column reference does not resolve to exactly one leaf, or a leaf's
    output columns are unknown: a conjunct could not be placed safely;
  * the region's output column order is visible in the query result
    (nothing above it but Filter, Sort, Limit and joins up to the root).
    Reordering changes the order of a join's output columns, which only a
    Project or Aggregate above it makes irrelevant. The binder always puts
    a Project on top;
  * a Sort or Limit is above it (``keep_row_order``, on by default).
    Reordering changes the order of the join's output rows. Under a Sort
    that only reorders rows whose sort keys tie, and under a Limit it
    changes which rows are kept. Both are legal SQL, but Contract §7 asks
    for identical rows in order under ORDER BY, and the team's differential
    harness checks exactly that;
  * the best plan is not at least 1% cheaper than the current one. Equal
    cost alternatives are not worth a change, and this keeps the pass
    stable when it runs again on its own output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ir.nodes import Aggregate, Filter, Join, Limit, Project, Sort

from optimizer.columns import column_refs, output_columns, resolves
from optimizer.cost import CostModel
from optimizer.expressions import TRUE, conjoin, split_conjuncts

DP_LIMIT = 8
MIN_GAIN = 0.01


class JoinReordering:
    name = "join_reordering"

    def __init__(self, dp_limit: int = DP_LIMIT, min_gain: float = MIN_GAIN, keep_row_order: bool = True):
        self.dp_limit = dp_limit
        self.min_gain = min_gain
        self.keep_row_order = keep_row_order

    def apply(self, plan: Any, catalog: Any) -> Any:
        reorderer = _Reorderer(catalog, self.dp_limit, self.min_gain, self.keep_row_order)
        return reorderer.visit(plan, order_visible=True, rows_ordered=False)


@dataclass(frozen=True)
class JoinGraph:
    """The leaves of a join region and its conjuncts, each with the bitmask of the leaves it reads."""

    leaves: tuple[Any, ...]
    conjuncts: tuple[tuple[int, Any], ...]

    def linked(self, a: int, b: int) -> bool:
        """Some conjunct reads leaves on both sides and nothing outside them."""
        both = a | b
        return any(m & a and m & b and not m & ~both for m, _ in self.conjuncts)

    def components(self) -> list[int]:
        """The connected components, as leaf bitmasks, in leaf order."""
        parent = list(range(len(self.leaves)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for m, _ in self.conjuncts:
            members = [i for i in range(len(self.leaves)) if m >> i & 1]
            for i in members[1:]:
                parent[find(i)] = find(members[0])
        masks: dict[int, int] = {}
        for i in range(len(self.leaves)):
            masks[find(i)] = masks.get(find(i), 0) | 1 << i
        return sorted(masks.values(), key=lambda m: m & -m)


def build_graph(leaves: list[Any], conjuncts: list[Any], catalog: Any) -> JoinGraph | None:
    """Return the join graph, or None if some column reference doesn't name exactly one leaf."""
    columns = [output_columns(leaf, catalog) for leaf in leaves]
    if any(c is None for c in columns):
        return None
    masked = []
    for p in conjuncts:
        mask = 0
        for ref in column_refs(p):
            hits = [i for i, cols in enumerate(columns) if any(resolves(ref, c) for c in cols)]
            if len(hits) != 1:
                return None
            mask |= 1 << hits[0]
        masked.append((mask, p))
    return JoinGraph(tuple(leaves), tuple(masked))


@dataclass(frozen=True)
class JoinPlan:
    plan: Any
    cost: float
    algorithm: str  # "dp" or "greedy"


def plan_joins(graph: JoinGraph, model: CostModel, dp_limit: int = DP_LIMIT) -> JoinPlan:
    """Return the cheapest join tree the algorithm finds for the graph."""
    return _Planner(graph, model).plan(dp_limit)


@dataclass(frozen=True)
class _Entry:
    mask: int
    plan: Any
    cost: float


class _Planner:
    def __init__(self, graph: JoinGraph, model: CostModel):
        self.graph = graph
        self.model = model
        n = len(graph.leaves)
        self.multi = [(m, p) for m, p in graph.conjuncts if m & (m - 1)]
        self.constants = [p for m, p in graph.conjuncts if m == 0]
        self.base = []
        for i, leaf in enumerate(graph.leaves):
            own = [p for m, p in graph.conjuncts if m == 1 << i]
            plan = Filter(child=leaf, predicate=conjoin(own)) if own else leaf
            self.base.append(_Entry(1 << i, plan, model.cost(plan).total))
        self.full = (1 << n) - 1

    def plan(self, dp_limit: int) -> JoinPlan:
        n = len(self.graph.leaves)
        if n <= dp_limit:
            algorithm = "dp"
            parts = [self._dp(component) for component in self.graph.components()]
        else:
            algorithm = "greedy"
            parts = [self._greedy(component) for component in self.graph.components()]
        # Disconnected components: join them with cross products, smallest results first.
        best = self._greedy_units(parts, linked=lambda a, b: True)
        plan = best.plan
        if self.constants:
            kept = [p for p in split_conjuncts(plan.condition) if p != TRUE]
            plan = Join(left=plan.left, right=plan.right, condition=conjoin(kept + self.constants), kind="inner")
        return JoinPlan(plan, self.model.cost(plan).total, algorithm)

    def _join(self, a: _Entry, b: _Entry) -> _Entry:
        mask = a.mask | b.mask
        conds = [p for m, p in self.multi if not m & ~mask and m & ~a.mask and m & ~b.mask]
        rows = self.model.estimator.rows
        left, right = (a, b) if (rows(a.plan), -a.mask) >= (rows(b.plan), -b.mask) else (b, a)
        node = Join(left=left.plan, right=right.plan, condition=conjoin(conds) or TRUE, kind="inner")
        return _Entry(mask, node, a.cost + b.cost + self.model.node_cost(node).total)

    def _dp(self, component: int, linked=None) -> _Entry:
        linked = linked or self.graph.linked
        best: dict[int, _Entry] = {e.mask: e for e in self.base if e.mask & component}
        subsets = [s for s in _submasks(component) if s & (s - 1)]
        subsets.sort(key=lambda s: (bin(s).count("1"), s))
        for s in subsets:
            low = s & -s
            for left in _submasks(s):
                right = s ^ left
                if not right or not left & low:  # each split once
                    continue
                if left in best and right in best and linked(left, right):
                    candidate = self._join(best[left], best[right])
                    if s not in best or candidate.cost < best[s].cost:
                        best[s] = candidate
        if component not in best:
            # Only a conjunct over three or more leaves connects the component, and no
            # split of it into two linked halves exists: some cross product is forced.
            return self._dp(component, linked=lambda a, b: True)
        return best[component]

    def _greedy(self, component: int) -> _Entry:
        units = [e for e in self.base if e.mask & component]
        return self._greedy_units(units, self.graph.linked)

    def _greedy_units(self, units: list[_Entry], linked) -> _Entry:
        units = list(units)
        while len(units) > 1:
            choice = None
            for i in range(len(units)):
                for j in range(i + 1, len(units)):
                    if not linked(units[i].mask, units[j].mask):
                        continue
                    joined = self._join(units[i], units[j])
                    key = (self.model.estimator.rows(joined.plan), joined.cost)
                    if choice is None or key < choice[0]:
                        choice = (key, i, j, joined)
            if choice is None:  # nothing linked: forced to cross
                return self._greedy_units(units, linked=lambda a, b: True)
            _, i, j, joined = choice
            units = [u for k, u in enumerate(units) if k not in (i, j)] + [joined]
        return units[0]


def _submasks(mask: int):
    sub = mask
    while sub:
        yield sub
        sub = (sub - 1) & mask


class _Reorderer:
    def __init__(self, catalog: Any, dp_limit: int, min_gain: float, keep_row_order: bool):
        self.catalog = catalog
        self.model = CostModel(catalog)
        self.dp_limit = dp_limit
        self.min_gain = min_gain
        self.keep_row_order = keep_row_order

    def visit(self, node: Any, order_visible: bool, rows_ordered: bool) -> Any:
        """``order_visible``: the query result shows this node's column order.
        ``rows_ordered``: a Sort or Limit above makes the order of its rows matter."""
        if isinstance(node, Join) and node.kind == "inner":
            return self._region(node, order_visible, rows_ordered)
        if isinstance(node, (Project, Aggregate)):
            inherited = False  # these name their outputs: input column order is invisible above
        elif isinstance(node, (Filter, Sort, Limit, Join)):
            inherited = order_visible
        else:
            inherited = True  # a node kind this pass doesn't know: assume order matters
        known = isinstance(node, (Project, Aggregate, Filter, Join))
        ordered = rows_ordered or not known  # Sort, Limit, and node kinds this pass doesn't know
        children = [self.visit(c, inherited, ordered) for c in node.children]
        if all(new is old for new, old in zip(children, node.children)):
            return node
        return node.replace_children(tuple(children))

    def _region(self, root: Join, order_visible: bool, rows_ordered: bool) -> Any:
        leaves, conjuncts = [], []
        _collect(root, leaves, conjuncts)
        new_leaves = [self.visit(leaf, order_visible, rows_ordered) for leaf in leaves]
        current = _rebuild(root, iter(new_leaves))
        if order_visible or (rows_ordered and self.keep_row_order) or len(leaves) < 3:
            return current
        graph = build_graph(new_leaves, [p for p in conjuncts if p != TRUE], self.catalog)
        if graph is None:
            return current
        best = plan_joins(graph, self.model, self.dp_limit)
        if best.cost < self.model.cost(current).total * (1 - self.min_gain):
            return best.plan
        return current


def _collect(node: Any, leaves: list[Any], conjuncts: list[Any]) -> None:
    if isinstance(node, Join) and node.kind == "inner":
        _collect(node.left, leaves, conjuncts)
        _collect(node.right, leaves, conjuncts)
        conjuncts.extend(split_conjuncts(node.condition))
    else:
        leaves.append(node)


def _rebuild(node: Any, leaves) -> Any:
    """The region with its leaves replaced, in the order ``_collect`` found them."""
    if isinstance(node, Join) and node.kind == "inner":
        left, right = _rebuild(node.left, leaves), _rebuild(node.right, leaves)
        if left is node.left and right is node.right:
            return node
        return node.replace_children((left, right))
    return next(leaves)
