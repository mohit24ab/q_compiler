"""Cardinality estimation from catalog statistics (Contract §6).

For every plan node the estimator computes how many rows the node is expected
to produce, together with statistics for each of its output columns, so the
node above can estimate its own predicates. It reads ``catalog.row_count``,
``catalog.schema`` and ``catalog.stats`` (ndv, min, max, null_count) and
nothing else: it never looks at the data.

Selectivity, the fraction of input rows a predicate keeps:

  col = v              (1 - nulls) / ndv, or 0 if v lies outside [min, max]
  col <> v             (1 - nulls) * (1 - 1/ndv)
  col < v  (<=, >, >=) (1 - nulls) * linear interpolation of v over [min, max]
  ranges on one column combined into one interval: x > 5 AND x < 10 is
                       P(x < 10) - P(x <= 5), not P(x > 5) * P(x < 10)
  col IS NULL          nulls          col IS NOT NULL   1 - nulls
  a = b  (two columns) (1 - nulls_a) * (1 - nulls_b) / max(ndv_a, ndv_b)
  p AND q              sel(p) * sel(q)                   (independence)
  p OR q               sel(p) + sel(q) - sel(p) * sel(q) (inclusion-exclusion)
  NOT p                NOT is pushed into p through exact three-valued-logic
                       equivalences (NOT x < v is x >= v, De Morgan, ...);
                       if that is impossible, 1 - sel(p)
  anything else        System R's defaults: 1/10 for =, 1/3 for a range,
                       1/2 for a predicate the estimator cannot read at all

Interpolation. The ndv distinct values are spread uniformly over
[min, max], and each carries 1/ndv of the rows. So
P(x < v) = (1 - 1/ndv) * (v - min) / (max - min) and
P(x <= v) = P(x < v) + 1/ndv, which makes the endpoints exact:
x < min keeps nothing, x <= max keeps every row, x >= max keeps 1/ndv.
Dates interpolate as day numbers. Strings cannot be interpolated: a string
bound outside [min, max] is still exact, and one inside falls back to 1/3.

Row counts per node:

  Scan        row_count * sel(pushed_predicate)
  Filter      rows(child) * sel(predicate)
  Project     rows(child)          Sort   rows(child)
  Limit       min(n, rows(child))
  Join        INNER: |L| * |R| * sel(condition). For an equi-join a = b this
              is the textbook |L| * |R| / max(ndv_a, ndv_b), with NULL keys
              (which never match) taken out.
              LEFT: inner + |L| * (1 - m), where m is the fraction of left rows
              with a match. For a key a = b, containment gives
              m = (1 - nulls_a) * min(1, ndv_b / ndv_a); without one, m is
              min(1, inner / |L|). Every left row appears at least once.
  Aggregate   grouped: min(rows(child), product of the group keys' ndv),
              where a key with NULLs has one extra group. Global: exactly 1.

Column statistics flow upward with the rows: a Filter on x = 5 leaves x
with ndv 1 and min = max = 5; a range narrows min and max and scales ndv;
any comparison on x removes x's NULLs; the right side of a LEFT join gains
NULLs for unmatched rows; join keys get min(ndv_a, ndv_b) (containment).
Below a join, no column has more distinct values than its node has rows.
A join's output does not cap ndv that way: a later join's selectivity
depends on the values a column draws from, not on how many happen to be
present, and without the cap the estimate for a set of joined relations is
the same in every join order, which the join reordering's dynamic program
relies on. A column a Filter does not constrain keeps each of its values with the chance
that at least one of the value's rows survives (Cardenas' urn model):
ndv' = ndv * (1 - (1 - sel) ** (rows / ndv)).

Known gaps, measured in docs/cardinality_estimates.md: skewed columns
(there is no most-common-values list or histogram in Contract §6),
correlated columns (independence is assumed), ranges inside a string
column's [min, max], and predicates on aggregate results.
"""

from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass
from typing import Any

from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort

from optimizer.columns import Ref, column_refs, table_schema
from optimizer.constant_folding import fold
from optimizer.expressions import op_name, split_conjuncts

# System R's defaults (Selinger et al., 1979), used when statistics can't answer.
DEFAULT_EQ_SEL = 1 / 10
DEFAULT_RANGE_SEL = 1 / 3
DEFAULT_SEL = 1 / 2
# Rows assumed for a table the catalog does not know.
DEFAULT_ROW_COUNT = 1000.0

_COMPARISONS = {"=", "!=", "<>", "<", "<=", ">", ">="}
_RANGES = {"<", "<=", ">", ">="}
_FLIP = {"=": "=", "!=": "!=", "<>": "<>", "<": ">", "<=": ">=", ">": "<", ">=": "<="}
_INVERSE = {"=": "<>", "!=": "=", "<>": "=", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}


@dataclass(frozen=True)
class ColumnEstimate:
    """Estimated statistics for one column of a node's output.

    ``ndv`` counts distinct non-NULL values (None if unknown). ``min`` and
    ``max`` bound the non-NULL values, in the form the catalog reports them
    (dates may be ISO strings). ``dtype`` says how to interpolate them.
    """

    ndv: float | None
    null_frac: float = 0.0
    min: Any = None
    max: Any = None
    dtype: Any = None


UNKNOWN = ColumnEstimate(ndv=None)


@dataclass(frozen=True)
class Estimate:
    """A node's estimated output: row count and per-column statistics, in output order."""

    rows: float
    columns: tuple[tuple[Ref, ColumnEstimate], ...] = ()

    def column(self, ref: Ref) -> ColumnEstimate | None:
        i = resolve(self.columns, ref)
        return None if i is None else self.columns[i][1]


def estimate(plan: Any, catalog: Any) -> Estimate:
    """Estimate one plan's output."""
    return CardinalityEstimator(catalog).estimate(plan)


def estimate_rows(plan: Any, catalog: Any) -> float:
    """Estimate how many rows one plan returns."""
    return estimate(plan, catalog).rows


def annotate(plan: Any, catalog: Any) -> list[tuple[tuple[int, ...], Any, Estimate]]:
    """Return ``(path, node, estimate)`` for every node, in pre-order.

    ``path`` lists child indexes from the root, so ``()`` is the root and
    ``(0, 1)`` is its first child's second child.
    """
    estimator = CardinalityEstimator(catalog)
    out = []

    def walk(node, path):
        out.append((path, node, estimator.estimate(node)))
        for i, child in enumerate(node.children):
            walk(child, path + (i,))

    walk(plan, ())
    return out


class CardinalityEstimator:
    """Estimates nodes bottom-up, remembering each node it has seen.

    Reuse one estimator across many plans that share subtrees (join
    reordering does) and each subtree is estimated once.
    """

    def __init__(self, catalog: Any):
        self.catalog = catalog
        self._memo: dict[int, tuple[Any, Estimate]] = {}

    def estimate(self, node: Any) -> Estimate:
        hit = self._memo.get(id(node))
        if hit is not None and hit[0] is node:
            return hit[1]
        result = self._estimate(node)
        self._memo[id(node)] = (node, result)  # keeps node alive, so its id can't be reused
        return result

    def rows(self, node: Any) -> float:
        return self.estimate(node).rows

    def selectivity(self, predicate: Any, scope: Estimate) -> float:
        """Return the fraction of ``scope``'s rows that ``predicate`` keeps."""
        if predicate is None:
            return 1.0
        return _clamp(self._and(split_conjuncts(fold(predicate)), scope.columns))

    # -- nodes -----------------------------------------------------------------

    def _estimate(self, node: Any) -> Estimate:
        if isinstance(node, Scan):
            return self._scan(node)
        if isinstance(node, Filter):
            return self._filter(self.estimate(node.child), node.predicate)
        if isinstance(node, Project):
            child = self.estimate(node.child)
            cols = [((None, alias), self._expr_column(fold(e), child)) for e, alias in node.exprs]
            return Estimate(child.rows, _cap(cols, child.rows))
        if isinstance(node, Sort):
            return self.estimate(node.child)
        if isinstance(node, Limit):
            child = self.estimate(node.child)
            rows = min(float(max(node.n, 0)), child.rows)
            return Estimate(rows, _cap(child.columns, rows))
        if isinstance(node, Join):
            return self._join(node)
        if isinstance(node, Aggregate):
            return self._aggregate(node)
        # A node kind this estimator doesn't know: assume it passes its first input through.
        if node.children:
            return Estimate(self.estimate(node.children[0]).rows)
        return Estimate(DEFAULT_ROW_COUNT)

    def _scan(self, node: Scan) -> Estimate:
        table_rows = self.row_count(node.table)
        schema = table_schema(node, self.catalog) or []
        full = Estimate(
            table_rows,
            tuple(((node.table, name), self._stats(node.table, name, dtype, table_rows))
                  for name, dtype in schema),
        )
        # The pushed predicate reads the full table, before narrowing to Scan.columns.
        out = self._filter(full, node.pushed_predicate)
        if node.columns is None:
            return out
        by_name = {name: c for (_, name), c in out.columns}
        return Estimate(out.rows, tuple(((node.table, n), by_name.get(n, UNKNOWN)) for n in node.columns))

    def _filter(self, child: Estimate, predicate: Any) -> Estimate:
        if predicate is None:
            return child
        conjuncts = split_conjuncts(fold(predicate))
        sel = _clamp(self._and(conjuncts, child.columns))
        rows = child.rows * sel
        narrowed = self._narrow(child.columns, conjuncts)
        # A column the predicate doesn't constrain loses the values whose every row was removed.
        survivors = [
            (ref, new if new is not old else _thinned(old, sel, child.rows))
            for (ref, new), (_, old) in zip(narrowed, child.columns)
        ]
        return Estimate(rows, _cap(survivors, rows))

    def _join(self, node: Join) -> Estimate:
        left, right = self.estimate(node.left), self.estimate(node.right)
        scope = left.columns + right.columns
        conjuncts = [] if node.condition is None else split_conjuncts(fold(node.condition))
        inner = left.rows * right.rows * _clamp(self._and(conjuncts, scope))
        narrowed = self._narrow(scope, conjuncts)
        # Join outputs keep each column's ndv uncapped by the row count (the
        # values the column draws from, not the values present). That keeps a
        # later join's selectivity right, and makes the estimate for a set of
        # joined relations the same whatever order they were joined in.
        # Aggregate caps its group count by its input rows itself.
        if node.kind != "left":
            return Estimate(inner, tuple(narrowed))
        # LEFT: the condition never removes a left row. A left row with no match
        # appears once, NULL-extended.
        unmatched = left.rows * (1.0 - self._matched_fraction(left, right, conjuncts, inner))
        rows = inner + unmatched
        padded = []
        for ref, c in narrowed[len(left.columns):]:
            nulls = (inner * c.null_frac + unmatched) / rows if rows else c.null_frac
            padded.append((ref, dataclasses.replace(c, null_frac=nulls)))
        return Estimate(rows, tuple(list(left.columns) + padded))

    def _matched_fraction(self, left: Estimate, right: Estimate, conjuncts, inner: float) -> float:
        """The fraction of left rows that find at least one match.

        For an equi-join key a = b, containment says the smaller set of key
        values is contained in the larger: a non-NULL left key finds a match
        with probability min(1, ndv_b / ndv_a). Without an equi-key, the
        matches are assumed spread over as many left rows as possible.
        """
        if not left.rows:
            return 0.0
        spread = min(1.0, inner / left.rows)
        fractions = []
        for p in conjuncts:
            if not (isinstance(p, BinaryOp) and op_name(p.op) == "="):
                continue
            for a_expr, b_expr in ((p.left, p.right), (p.right, p.left)):
                i, j = _index(left.columns, a_expr, loose=False), _index(right.columns, b_expr, loose=False)
                if i is None or j is None:
                    continue
                a, b = left.columns[i][1], right.columns[j][1]
                if a.ndv is not None and b.ndv is not None:
                    contained = min(1.0, b.ndv / a.ndv) if a.ndv else 0.0
                    fractions.append((1.0 - a.null_frac) * contained)
        return min([spread] + fractions)

    def _aggregate(self, node: Aggregate) -> Estimate:
        child = self.estimate(node.child)
        keys = [(k, self._expr_column(fold(k), child)) for k in node.group_keys]
        if not keys:
            rows = 1.0  # a global aggregate returns one row, even for empty input
        else:
            groups = 1.0
            for _, c in keys:
                distinct = child.rows if c.ndv is None else c.ndv
                groups *= distinct + min(1.0, c.null_frac * child.rows)  # NULL is a group too
            rows = min(child.rows, groups)
        cols: list[tuple[Ref, ColumnEstimate]] = []
        for i, (k, c) in enumerate(keys):
            ref = (k.table, k.name) if isinstance(k, ColumnRef) else (None, f"<group key {i}>")
            cols.append((ref, c))
        for call, alias in node.aggs:
            cols.append(((None, alias), self._agg_column(call, child, rows)))
        return Estimate(rows, _cap(cols, rows))

    def _agg_column(self, call: AggCall, child: Estimate, rows: float) -> ColumnEstimate:
        func = call.func.lower()
        if func == "count":
            return ColumnEstimate(ndv=rows, dtype=DType.INT)
        arg = self._expr_column(fold(call.arg), child) if call.arg is not None else UNKNOWN
        nulls = 1.0 if arg.null_frac >= 1.0 else 0.0
        if func in ("min", "max", "avg"):  # these stay within the argument's range
            return ColumnEstimate(ndv=rows, null_frac=nulls, min=arg.min, max=arg.max,
                                  dtype=DType.FLOAT if func == "avg" else arg.dtype)
        return ColumnEstimate(ndv=rows, null_frac=nulls, dtype=arg.dtype)

    def _expr_column(self, expr: Any, child: Estimate) -> ColumnEstimate:
        """Statistics of an expression evaluated over the child's rows."""
        if isinstance(expr, ColumnRef):
            c = child.column((expr.table, expr.name))
            return UNKNOWN if c is None else c
        if isinstance(expr, Literal):
            if expr.value is None:
                return ColumnEstimate(ndv=0.0, null_frac=1.0, dtype=expr.dtype)
            return ColumnEstimate(ndv=1.0, min=expr.value, max=expr.value, dtype=expr.dtype)
        inputs = [child.column(r) for r in column_refs(expr)]
        ndv = None
        if all(c is not None and c.ndv is not None for c in inputs):
            ndv = 1.0
            for c in inputs:
                ndv *= c.ndv
        not_null = 1.0
        for c in inputs:
            not_null *= 1.0 - (c.null_frac if c is not None else 0.0)
        return ColumnEstimate(ndv=ndv, null_frac=1.0 - not_null)

    # -- catalog -----------------------------------------------------------------

    def row_count(self, table: str) -> float:
        """The table's row count from the catalog, or DEFAULT_ROW_COUNT if the catalog doesn't know it."""
        try:
            return float(self.catalog.row_count(table))
        except (AttributeError, LookupError):
            return DEFAULT_ROW_COUNT

    def _stats(self, table: str, name: str, dtype: Any, rows: float) -> ColumnEstimate:
        try:
            s = self.catalog.stats(table, name)
        except (AttributeError, LookupError):
            return ColumnEstimate(ndv=None, dtype=dtype)
        return ColumnEstimate(
            ndv=float(s.ndv),
            null_frac=s.null_count / rows if rows else 0.0,
            min=s.min, max=s.max, dtype=dtype,
        )

    # -- selectivity --------------------------------------------------------------

    def _and(self, conjuncts: list[Any], scope) -> float:
        """Selectivity of a conjunction. Constraints on one column are combined before multiplying."""
        sel = 1.0
        groups: dict[int, list[tuple[str, Any]]] = {}
        for p in conjuncts:
            atom = _atom(p, scope)
            if atom is None:
                sel *= self._sel(p, scope)
            else:
                groups.setdefault(atom[0], []).append(atom[1:])
        for i, atoms in groups.items():
            sel *= _column_selectivity(scope[i][1], atoms)
        return sel

    def _sel(self, p: Any, scope) -> float:
        if isinstance(p, Literal):
            if p.value is True:
                return 1.0
            return 0.0 if p.value is None or p.value is False else DEFAULT_SEL
        atom = _atom(p, scope)
        if atom is not None:
            return _column_selectivity(scope[atom[0]][1], [atom[1:]])
        if isinstance(p, BinaryOp):
            o = op_name(p.op)
            if o == "AND":
                return self._and(split_conjuncts(p), scope)
            if o == "OR":
                a, b = self._and([p.left], scope), self._and([p.right], scope)
                return a + b - a * b
            if o in _COMPARISONS:
                return _comparison(o, p.left, p.right, scope)
            return DEFAULT_SEL
        if isinstance(p, UnaryOp):
            o = op_name(p.op)
            if o == "NOT":
                negated = _negate(p.operand)
                return self._sel(negated, scope) if negated is not None else 1.0 - self._sel(p.operand, scope)
            if o in ("IS NULL", "IS NOT NULL"):
                nulls = 1.0
                for ref in column_refs(p.operand):
                    c = _lookup(scope, ref)
                    nulls *= 1.0 - (c.null_frac if c is not None else 0.0)
                nulls = 1.0 - nulls  # an expression is NULL if any input is
                return nulls if o == "IS NULL" else 1.0 - nulls
        return DEFAULT_SEL

    def _narrow(self, columns, conjuncts: list[Any]) -> list[tuple[Ref, ColumnEstimate]]:
        """Column statistics after the conjuncts have removed their rows."""
        out = list(columns)
        groups: dict[int, list[tuple[str, Any]]] = {}
        for p in conjuncts:
            atom = _atom(p, columns)
            if atom is not None:
                groups.setdefault(atom[0], []).append(atom[1:])
            elif isinstance(p, BinaryOp) and op_name(p.op) in _COMPARISONS:
                i, j = _index(columns, p.left), _index(columns, p.right)
                for k in (i, j):  # a comparison never keeps a row where its column is NULL
                    if k is not None:
                        out[k] = (out[k][0], dataclasses.replace(out[k][1], null_frac=0.0))
                if op_name(p.op) == "=" and i is not None and j is not None:
                    ndvs = [out[k][1].ndv for k in (i, j) if out[k][1].ndv is not None]
                    if ndvs:  # containment: the matching keys are the smaller side's values
                        for k in (i, j):
                            out[k] = (out[k][0], dataclasses.replace(out[k][1], ndv=min(ndvs)))
        for i, atoms in groups.items():
            out[i] = (out[i][0], _restrict(out[i][1], atoms))
        return out


# --------------------------------------------------------------------------
# Atoms: a comparison of one column with a literal
# --------------------------------------------------------------------------


def _atom(p: Any, scope) -> tuple[int, str, Any] | None:
    """Recognise ``col op literal`` (either way round), ``col IS [NOT] NULL`` and bare boolean columns.

    Returns ``(column index in scope, op, value)``, or None.
    """
    if isinstance(p, ColumnRef):
        i = _index(scope, p)
        return None if i is None else (i, "=", True)
    if isinstance(p, UnaryOp):
        o = op_name(p.op)
        i = _index(scope, p.operand)
        if i is None:
            return None
        if o in ("IS NULL", "IS NOT NULL"):
            return (i, o, None)
        if o == "NOT" and isinstance(p.operand, ColumnRef):
            return (i, "=", False)
        return None
    if isinstance(p, BinaryOp) and op_name(p.op) in _COMPARISONS:
        o = op_name(p.op)
        for column, value, symbol in ((p.left, p.right, o), (p.right, p.left, _FLIP[o])):
            i = _index(scope, column)
            if i is not None and isinstance(value, Literal) and value.value is not None:
                return (i, symbol, value.value)
    return None


def _column_selectivity(c: ColumnEstimate, atoms: list[tuple[str, Any]]) -> float:
    """The fraction of rows satisfying every atom on one column."""
    ops = {o for o, _ in atoms}
    if "IS NULL" in ops:
        return 0.0 if len(ops) > 1 else c.null_frac  # IS NULL and any comparison: nothing
    values = [(o, v) for o, v in atoms if o != "IS NOT NULL"]
    fraction = _values_fraction(c, values)
    if fraction is None:  # the bounds can't be compared: treat each atom as independent
        fraction = 1.0
        for atom in values:
            single = _values_fraction(c, [atom])
            fraction *= DEFAULT_RANGE_SEL if single is None else single
    return (1.0 - c.null_frac) * fraction


def _values_fraction(c: ColumnEstimate, atoms: list[tuple[str, Any]]) -> float | None:
    """Among the column's non-NULL values, the fraction satisfying every atom; None if unknowable."""
    try:
        lower, upper = _bounds(c, atoms)
        eqs = [v for o, v in atoms if o == "="]
        nes = [v for o, v in atoms if o in ("!=", "<>")]
        if eqs:
            v = eqs[0]
            if any(_key(e, c.dtype) != _key(v, c.dtype) for e in eqs):
                return 0.0  # x = 1 AND x = 2
            if not _within(v, lower, upper, c.dtype) or any(_key(n, c.dtype) == _key(v, c.dtype) for n in nes):
                return 0.0
            return _eq_fraction(c, v)
        hi = 1.0 if upper is None else _below(c, upper[0], inclusive=upper[1])
        lo = 0.0 if lower is None else _below(c, lower[0], inclusive=not lower[1])
        if hi is None or lo is None:
            return None
        fraction = max(0.0, hi - lo)
        for n in {_key(v, c.dtype): v for v in nes}.values():
            if _within(n, lower, upper, c.dtype):
                fraction = max(0.0, fraction - _eq_fraction(c, n))
        return fraction
    except TypeError:  # bounds of mixed types, which can't be ordered
        return None


def _bounds(c: ColumnEstimate, atoms):
    """The tightest lower and upper bound among the atoms, each as ``(value, inclusive)``."""
    lower = upper = None
    for o, v in atoms:
        if o in (">", ">="):
            bound = (v, o == ">=")
            if lower is None or _tighter(bound, lower, c.dtype, above=True):
                lower = bound
        elif o in ("<", "<="):
            bound = (v, o == "<=")
            if upper is None or _tighter(bound, upper, c.dtype, above=False):
                upper = bound
    return lower, upper


def _tighter(a, b, dtype, above: bool) -> bool:
    ka, kb = _key(a[0], dtype), _key(b[0], dtype)
    if ka == kb:
        return not a[1]  # x > 5 is tighter than x >= 5
    return ka > kb if above else ka < kb


def _within(v, lower, upper, dtype) -> bool:
    k = _key(v, dtype)
    if lower is not None:
        lk = _key(lower[0], dtype)
        if k < lk or (k == lk and not lower[1]):
            return False
    if upper is not None:
        uk = _key(upper[0], dtype)
        if k > uk or (k == uk and not upper[1]):
            return False
    return True


def _eq_fraction(c: ColumnEstimate, v: Any) -> float:
    """Among non-NULL values, the fraction equal to ``v``."""
    if _outside(c, v):
        return 0.0
    if c.ndv is None:
        return DEFAULT_EQ_SEL
    return 1.0 / max(c.ndv, 1.0)


def _outside(c: ColumnEstimate, v: Any) -> bool:
    x, lo, hi = _key(v, c.dtype), _key(c.min, c.dtype), _key(c.max, c.dtype)
    if x is None or lo is None or hi is None or not _same_kind(x, lo, hi):
        return False
    return x < lo or x > hi


def _below(c: ColumnEstimate, v: Any, inclusive: bool) -> float | None:
    """Among non-NULL values, P(x < v), or P(x <= v) if ``inclusive``; None if unknowable."""
    x, lo, hi = _key(v, c.dtype), _key(c.min, c.dtype), _key(c.max, c.dtype)
    if x is None or lo is None or hi is None or not _same_kind(x, lo, hi):
        return None
    point = 1.0 / max(c.ndv, 1.0) if c.ndv else 0.0
    if x < lo:
        return 0.0
    if x > hi:
        return 1.0
    if x == lo and not inclusive:
        return 0.0
    if x == hi and inclusive:
        return 1.0
    if isinstance(x, str):
        return None  # inside [min, max]: strings can't be interpolated
    fraction = 0.0 if hi == lo else (x - lo) / (hi - lo) * (1.0 - point)
    return _clamp(fraction + (point if inclusive else 0.0))


def _restrict(c: ColumnEstimate, atoms: list[tuple[str, Any]]) -> ColumnEstimate:
    """The column's statistics over just the rows satisfying ``atoms``."""
    ops = {o for o, _ in atoms}
    if "IS NULL" in ops:
        return ColumnEstimate(ndv=0.0, null_frac=1.0, dtype=c.dtype)
    values = [(o, v) for o, v in atoms if o != "IS NOT NULL"]
    eqs = [v for o, v in values if o == "="]
    if eqs:
        return ColumnEstimate(ndv=1.0, null_frac=0.0, min=eqs[0], max=eqs[0], dtype=c.dtype)
    fraction = _values_fraction(c, values)
    ndv = c.ndv
    if ndv is not None and fraction is not None:
        ndv = max(1.0, ndv * fraction) if fraction > 0 else 0.0
    low, high = c.min, c.max
    try:
        lower, upper = _bounds(c, values)
        if lower is not None and (low is None or _key(lower[0], c.dtype) > _key(low, c.dtype)):
            low = lower[0]
        if upper is not None and (high is None or _key(upper[0], c.dtype) < _key(high, c.dtype)):
            high = upper[0]
    except TypeError:
        pass
    return ColumnEstimate(ndv=ndv, null_frac=0.0, min=low, max=high, dtype=c.dtype)


def _comparison(o: str, left: Any, right: Any, scope) -> float:
    """A comparison that isn't ``col op literal``: two columns, or expressions."""
    if any(isinstance(x, Literal) and x.value is None for x in (left, right)):
        return 0.0  # comparing with NULL is never TRUE
    i, j = _index(scope, left), _index(scope, right)
    if i is not None and j is not None:
        a, b = scope[i][1], scope[j][1]
        if i == j:  # x op x: TRUE for every non-NULL x, or for none
            return (1.0 - a.null_frac) if o in ("=", "<=", ">=") else 0.0
        not_null = (1.0 - a.null_frac) * (1.0 - b.null_frac)
        ndvs = [c.ndv for c in (a, b) if c.ndv is not None]
        eq = 1.0 / max(max(ndvs), 1.0) if ndvs else DEFAULT_EQ_SEL
        if o == "=":
            return not_null * eq
        if o in ("!=", "<>"):
            return not_null * (1.0 - eq)
        return not_null * DEFAULT_RANGE_SEL
    if o == "=":
        return DEFAULT_EQ_SEL
    if o in ("!=", "<>"):
        return 1.0 - DEFAULT_EQ_SEL
    return DEFAULT_RANGE_SEL


def _negate(p: Any) -> Any | None:
    """Return an expression equal to ``NOT p`` under three-valued logic, with the NOT pushed inside; None if impossible."""
    if isinstance(p, Literal) and (p.value is None or isinstance(p.value, bool)):
        return p if p.value is None else Literal(value=not p.value, dtype=p.dtype)
    if isinstance(p, BinaryOp):
        o = op_name(p.op)
        if o in _INVERSE:
            return BinaryOp(op=_INVERSE[o], left=p.left, right=p.right)
        if o in ("AND", "OR"):  # De Morgan holds in three-valued logic
            return BinaryOp(op="OR" if o == "AND" else "AND",
                            left=_not(p.left), right=_not(p.right))
    if isinstance(p, UnaryOp):
        o = op_name(p.op)
        if o == "NOT":
            return p.operand
        if o == "IS NULL":
            return UnaryOp(op="IS NOT NULL", operand=p.operand)
        if o == "IS NOT NULL":
            return UnaryOp(op="IS NULL", operand=p.operand)
    return None


def _not(p: Any) -> Any:
    negated = _negate(p)
    return UnaryOp(op="NOT", operand=p) if negated is None else negated


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def resolve(columns, ref: Ref, loose: bool = True) -> int | None:
    """Index of the column ``ref`` names, or None if it names none or several.

    Strict resolution comes first (the reference evaluator's rule). If a
    qualified ref matches nothing, and ``loose`` is set, it may match by
    name alone: a Project's outputs lose their table qualifiers, and an
    estimate is better with the column's statistics than without. Deciding
    which side of a join a column comes from must use ``loose=False``.
    """
    q, n = ref
    strict = [i for i, ((cq, cn), _) in enumerate(columns) if cn == n and (q is None or q == cq)]
    if len(strict) == 1:
        return strict[0]
    if not strict and q is not None and loose:
        loose = [i for i, ((_, cn), _) in enumerate(columns) if cn == n]
        if len(loose) == 1:
            return loose[0]
    return None


def _index(columns, expr: Any, loose: bool = True) -> int | None:
    return resolve(columns, (expr.table, expr.name), loose) if isinstance(expr, ColumnRef) else None


def _lookup(columns, ref: Ref) -> ColumnEstimate | None:
    i = resolve(columns, ref)
    return None if i is None else columns[i][1]


def _key(value: Any, dtype: Any) -> Any:
    """An orderable form of a value: numbers as float, dates as day numbers, other strings unchanged."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime.datetime):
        return float(value.date().toordinal())
    if isinstance(value, datetime.date):
        return float(value.toordinal())
    if isinstance(value, str):
        if dtype == DType.DATE:
            try:
                return float(datetime.date.fromisoformat(value[:10]).toordinal())
            except ValueError:
                return None
        return value
    return None


def _same_kind(*keys: Any) -> bool:
    return all(isinstance(k, str) for k in keys) or all(isinstance(k, float) for k in keys)


def _thinned(c: ColumnEstimate, sel: float, rows: float) -> ColumnEstimate:
    """A column's ndv after keeping a random fraction ``sel`` of ``rows`` rows (Cardenas' urn model).

    Each of the ndv values occupies rows / ndv rows; a value survives if any
    of its rows does: ndv' = ndv * (1 - (1 - sel) ** (rows / ndv)).
    """
    if c.ndv is None or c.ndv <= 0 or sel >= 1.0:
        return c
    survivors = c.ndv * (1.0 - (1.0 - sel) ** (rows / c.ndv))
    return dataclasses.replace(c, ndv=survivors)


def _cap(columns, rows: float) -> tuple[tuple[Ref, ColumnEstimate], ...]:
    """No column has more distinct values than there are rows."""
    return tuple(
        (ref, c if c.ndv is None or c.ndv <= rows else dataclasses.replace(c, ndv=rows))
        for ref, c in columns
    )


def _clamp(x: float) -> float:
    return min(1.0, max(0.0, x))
