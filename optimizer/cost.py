"""Cost model: estimated work to run a plan with Person C's generated code.

The cost of a plan is the sum of its nodes' costs. A node's cost is the
estimated rows it touches (from ``optimizer.stats``) times a per-row weight
for each kind of work it does. Every weight is a time in nanoseconds,
measured by ``optimizer.calibration`` on the code that codegen emits for
that work, so a plan's cost reads as a predicted runtime in nanoseconds.
Only the ratios between weights matter when comparing plans.

Work is booked to separate components, so a cost can be broken down:

  scan_io      Scan: converting each column read from Arrow to numpy, per
               table row. Strings cost ~18x a fixed-width value. A column
               counts if it is output or read by the pushed predicate.
  predicate    One vectorized operator (comparison, arithmetic, AND, ...) per
               row: pushed predicates, Filters, computed Project expressions,
               residual join conditions.
  gather       Copying columns through a row mask (Filter, pushed predicate:
               per input row) or an index array (join and sort output: per
               output row).
  join_build   Hash join: one dict insert per row of the smaller input.
               Codegen builds on the smaller input at run time, whichever side
               it is, so join orientation does not change the cost.
  join_probe   Hash join: one dict lookup per row of the larger input.
  join_output  Hash join: per candidate pair (rows matching the equality
               keys): append, convert, and sort into the reference order.
  nested_loop  A join with no equality key: every (left, right) pair.
  aggregation  Grouped: one dict lookup per input row to find its group,
               plus one accumulator update per row per aggregate. Global: one
               vectorized reduction per row per aggregate.
  sort         n * log2(n) per sort key, for the ranking and the lexsort.

Codegen never runs a subtree whose result is known to be empty: a Filter,
a pushed predicate or an INNER join condition that is the literal FALSE or
NULL costs nothing, and neither does anything below it.

The weights (``DEFAULT_WEIGHTS``) were measured on the development container
(Intel Xeon @ 2.80GHz, Python 3.11, numpy 2.4, pyarrow 25), taking the
median of three ``python -m optimizer.calibration`` runs at 200,000 rows,
rounded to two significant figures. The Python-level hash table loops
dominate: one insert costs about as much as 2,600 vectorized comparisons.
That is why the join order (B6) matters far more here than in a vectorized
engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ir.dtype import DType
from ir.expr import BinaryOp, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort

from optimizer.columns import column_refs, table_schema
from optimizer.constant_folding import fold
from optimizer.expressions import conjoin, op_name, split_conjuncts
from optimizer.stats import CardinalityEstimator, Estimate, resolve


@dataclass(frozen=True)
class CostWeights:
    """Nanoseconds per unit of work. See ``optimizer.calibration`` for how each is measured."""

    read_fixed: float    # Arrow to numpy, one INT/FLOAT/DATE/BOOL value
    read_string: float   # Arrow to numpy, one STRING value
    predicate: float     # one vectorized operator over one row
    gather_mask: float   # one column through a boolean mask, per input row
    gather_index: float  # one column through an index array, per output row
    hash_build: float    # one row inserted into the join's hash table
    hash_probe: float    # one row looked up in the join's hash table
    join_emit: float     # one matching pair emitted, converted and sorted
    pair: float          # one pair of a nested-loop join
    group: float         # one row assigned its group id
    aggregate: float     # one grouped accumulator update
    reduce: float        # one row of a global aggregate
    sort: float          # per n * log2(n) per sort key


#          measured (ns): run 1   run 2   run 3
DEFAULT_WEIGHTS = CostWeights(
    read_fixed=4.0,      # 5.56    3.54    3.91
    read_string=70.0,    # 70.6    66.5    87.0
    predicate=0.25,      # 0.251   0.249   0.240
    gather_mask=12.0,    # 12.4    12.3    12.7
    gather_index=4.7,    # 4.68    4.29    5.32
    hash_build=660.0,    # 654     656     813
    hash_probe=120.0,    # 123     117     118
    join_emit=670.0,     # 799     627     670
    pair=0.86,           # 0.924   0.862   0.790
    group=310.0,         # 307     308     343
    aggregate=4.2,       # 6.26    4.12    4.18
    reduce=1.5,          # 1.51    1.67    1.49
    sort=9.2,            # 9.15    8.80    9.27
)

COMPONENTS = (
    "scan_io", "predicate", "gather", "join_build", "join_probe",
    "join_output", "nested_loop", "aggregation", "sort",
)


@dataclass(frozen=True)
class Cost:
    """A cost in nanoseconds, broken down by component."""

    components: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.components.values())

    def __add__(self, other: "Cost") -> "Cost":
        merged = dict(self.components)
        for name, value in other.components.items():
            merged[name] = merged.get(name, 0.0) + value
        return Cost(merged)


def plan_cost(plan: Any, catalog: Any, weights: CostWeights = DEFAULT_WEIGHTS) -> float:
    """The estimated cost of running ``plan``, in nanoseconds."""
    return CostModel(catalog, weights).cost(plan).total


class CostModel:
    """Costs plans. One model can cost many plans; it shares one cardinality estimator across them."""

    def __init__(self, catalog: Any, weights: CostWeights = DEFAULT_WEIGHTS,
                 estimator: CardinalityEstimator | None = None):
        self.catalog = catalog
        self.weights = weights
        self.estimator = estimator if estimator is not None else CardinalityEstimator(catalog)

    def cost(self, plan: Any) -> Cost:
        """The cost of the whole subtree rooted at ``plan``."""
        total = self.node_cost(plan)
        if not _short_circuits(plan):
            for child in plan.children:
                total = total + self.cost(child)
        return total

    def node_cost(self, node: Any) -> Cost:
        """The cost of ``node``'s own work, excluding its children."""
        if _short_circuits(node):
            return Cost({})
        if isinstance(node, Scan):
            return self._scan(node)
        if isinstance(node, Filter):
            child = self.estimator.estimate(node.child)
            return Cost({
                "predicate": child.rows * _ops(node.predicate) * self.weights.predicate,
                "gather": child.rows * _width(child) * self.weights.gather_mask,
            })
        if isinstance(node, Project):
            rows = self.estimator.rows(node.child)
            ops = sum(_ops(e) for e, _ in node.exprs)
            return Cost({"predicate": rows * ops * self.weights.predicate})
        if isinstance(node, Join):
            return self._join(node)
        if isinstance(node, Aggregate):
            return self._aggregate(node)
        if isinstance(node, Sort):
            est = self.estimator.estimate(node)
            n = est.rows
            return Cost({
                "sort": n * math.log2(max(n, 2.0)) * len(node.keys) * self.weights.sort,
                "gather": n * _width(est) * self.weights.gather_index,
            })
        return Cost({})  # Limit is a slice; unknown nodes are not costed

    def _scan(self, node: Scan) -> Cost:
        w = self.weights
        rows = self.estimator.row_count(node.table)
        schema = dict(table_schema(node, self.catalog) or [])
        out = list(node.columns) if node.columns is not None else list(schema)
        pushed = None if node.pushed_predicate is None else fold(node.pushed_predicate)
        read = set(out) | {name for _, name in column_refs(pushed)}
        io = sum(w.read_string if schema.get(c) == DType.STRING else w.read_fixed for c in read)
        cost = {"scan_io": rows * io}
        if pushed is not None:
            cost["predicate"] = rows * _ops(pushed) * w.predicate
            cost["gather"] = rows * len(out) * w.gather_mask
        return Cost(cost)

    def _join(self, node: Join) -> Cost:
        w = self.weights
        left, right = self.estimator.estimate(node.left), self.estimator.estimate(node.right)
        out = self.estimator.estimate(node)
        conjuncts = [] if node.condition is None else split_conjuncts(fold(node.condition))
        if any(isinstance(c, Literal) and (c.value is None or c.value is False) for c in conjuncts):
            # A LEFT join that can never match: both sides still run, every left row is unmatched.
            return Cost({"gather": out.rows * _width(out) * w.gather_index})
        keys = [c for c in conjuncts if _is_equi_key(c, left, right)]
        residual = [c for c in conjuncts if not any(c is k for k in keys)
                    and not (isinstance(c, Literal) and c.value is True)]
        cost = {"gather": out.rows * _width(out) * w.gather_index}
        if keys:
            scope = Estimate(left.rows * right.rows, left.columns + right.columns)
            candidates = scope.rows * self.estimator.selectivity(conjoin(keys), scope)
            small, large = sorted((left.rows, right.rows))
            cost["join_build"] = small * w.hash_build
            cost["join_probe"] = large * w.hash_probe
            cost["join_output"] = candidates * w.join_emit
            cost["predicate"] = candidates * sum(_ops(c) for c in residual) * w.predicate
        else:
            pairs = left.rows * right.rows
            cost["nested_loop"] = pairs * (w.pair + sum(_ops(c) for c in residual) * w.predicate)
        return Cost(cost)

    def _aggregate(self, node: Aggregate) -> Cost:
        w = self.weights
        rows_in = self.estimator.rows(node.child)
        out = self.estimator.estimate(node)
        n_aggs = max(len(node.aggs), 0)
        computed = sum(_ops(k) for k in node.group_keys) + sum(_ops(c.arg) for c, _ in node.aggs if c.arg is not None)
        cost = {"predicate": rows_in * computed * w.predicate}
        if node.group_keys:
            cost["aggregation"] = rows_in * (w.group + n_aggs * w.aggregate)
            cost["gather"] = out.rows * len(node.group_keys) * w.gather_index
        else:
            cost["aggregation"] = rows_in * n_aggs * w.reduce
        return Cost(cost)


def explain(plan: Any, catalog: Any, weights: CostWeights = DEFAULT_WEIGHTS, formatter=None) -> str:
    """Render ``plan`` with each node's estimated rows and own cost, plus the total and its breakdown."""
    from optimizer.trace import default_formatter

    fmt = formatter or default_formatter()
    model = CostModel(catalog, weights)
    labels = fmt(plan).splitlines()
    nodes = []

    def walk(node):
        nodes.append(node)
        for child in node.children:
            walk(child)

    walk(plan)
    if len(labels) != len(nodes):  # a formatter that doesn't print one line per node
        labels = [type(n).__name__ for n in nodes]
    width = max(len(label) for label in labels)
    lines = []
    for label, node in zip(labels, nodes):
        rows = model.estimator.rows(node)
        own = model.node_cost(node).total
        lines.append(f"{label:<{width}}  rows≈{rows:>10.1f}  cost {_us(own):>10}")
    total = model.cost(plan)
    parts = ", ".join(f"{name} {_us(v)}" for name, v in sorted(total.components.items(), key=lambda kv: -kv[1]) if v)
    lines.append(f"total {_us(total.total)}" + (f"  ({parts})" if parts else ""))
    return "\n".join(lines)


def _us(ns: float) -> str:
    return f"{ns / 1000:.1f}µs"


def _short_circuits(node: Any) -> bool:
    """Codegen returns an empty result for this node without running anything below it."""
    if isinstance(node, Filter):
        return _never_true(node.predicate)
    if isinstance(node, Scan):
        return node.pushed_predicate is not None and _never_true(node.pushed_predicate)
    if isinstance(node, Join):
        return node.kind != "left" and node.condition is not None and _never_true(node.condition)
    return False


def _never_true(predicate: Any) -> bool:
    return isinstance(predicate, Literal) and (predicate.value is None or predicate.value is False)


def _ops(expr: Any) -> int:
    """Vectorized operators an expression costs: one per operator node."""
    if isinstance(expr, BinaryOp):
        return 1 + _ops(expr.left) + _ops(expr.right)
    if isinstance(expr, UnaryOp):
        return 1 + _ops(expr.operand)
    return 0


def _width(est: Estimate) -> int:
    return max(len(est.columns), 1)


def _is_equi_key(p: Any, left: Estimate, right: Estimate) -> bool:
    """``a = b`` with ``a`` from one input and ``b`` from the other: what codegen hashes on."""
    if not (isinstance(p, BinaryOp) and op_name(p.op) == "="):
        return False

    def side(expr):
        refs = column_refs(expr)
        in_left = bool(refs) and all(resolve(left.columns, r, loose=False) is not None for r in refs)
        in_right = bool(refs) and all(resolve(right.columns, r, loose=False) is not None for r in refs)
        return "L" if in_left and not in_right else "R" if in_right and not in_left else None

    return {side(p.left), side(p.right)} == {"L", "R"}
