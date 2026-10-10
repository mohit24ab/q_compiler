"""Constant folding and expression simplification.

Expression rules, applied bottom-up. These are exact under SQL's
three-valued logic, so they are valid anywhere: SELECT lists, group keys,
aggregate arguments, sort keys, and predicates.

    literal op literal     ->  the computed literal (2 * 3 -> 6;
                               '2024-01-01' < '2024-06-01' -> TRUE)
    NULL op x              ->  NULL, for operators that propagate NULL
    x AND TRUE -> x        x AND FALSE -> FALSE     x AND x -> x
    x OR FALSE -> x        x OR TRUE   -> TRUE      x OR x  -> x
    NOT NOT x  -> x        NOT TRUE    -> FALSE     NOT NULL -> NULL
    literal IS [NOT] NULL  ->  TRUE / FALSE

A few literal pairs are left unfolded on purpose, because the result would
depend on runtime semantics the contract doesn't pin down: integer
division, any division by zero, ``%``, results outside int64,
comparisons between different types, and comparisons with a string that
reads as a date but isn't written YYYY-MM-DD (``'19950101'``): as strings
it orders differently from the date the runtime reads it as.

Predicate rules apply only where a predicate decides whether a row is kept:
Filter predicates, join conditions, and Scan.pushed_predicate. There, NULL
and FALSE both reject the row, so these are valid too:

    a NULL or FALSE conjunct           ->  FALSE
    duplicate conjuncts                ->  kept once
    contradictory comparisons of one   ->  FALSE  (x = 1 AND x = 2,
    column with literals                           x > 5 AND x < 3,
                                                   x = 3 AND x <> 3)

A contradiction is NULL, not FALSE, when the column is NULL. So these
rules must never touch a SELECT expression.

Plan rules:

    Filter[TRUE]          removed; a Scan's TRUE pushed_predicate is dropped
    Limit 0               empty
    no-op Project         removed: it passes its input through unchanged,
                          same columns, same names, same order

An empty result is written ``Filter[FALSE]`` over the subtree whose
columns it keeps. That marker is lifted as high as emptiness provably
propagates: through Project, Sort, Limit, Filter, a grouped Aggregate, an
INNER join on either side, and the left side of a LEFT join. It stops at a
global Aggregate, which returns one row even for empty input, and at the
right side of a LEFT join, whose left rows survive NULL-extended. Codegen
can then return an empty table of the child's columns without computing
the child at all.
"""

from __future__ import annotations

import dataclasses
import datetime
import operator
from typing import Any

from ir.dtype import DType
from ir.expr import BinaryOp, ColumnRef, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort

from optimizer.columns import Ref, table_schema
from optimizer.expressions import TRUE, op_name, rebuild, split_conjuncts

FALSE = Literal(value=False, dtype=DType.BOOL)

_INT64 = (-(2**63), 2**63 - 1)
_NUMERIC = (DType.INT, DType.FLOAT)
_ARITHMETIC = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}
_COMPARISON = {
    "=": operator.eq, "!=": operator.ne, "<>": operator.ne,
    "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
}
_NULL_PROPAGATING = set(_ARITHMETIC) | set(_COMPARISON) | {"%"}


class ConstantFolding:
    name = "constant_folding"

    def apply(self, plan: Any, catalog: Any) -> Any:
        return _Folder(catalog).plan(plan)


# --------------------------------------------------------------------------
# Expressions
# --------------------------------------------------------------------------


def fold(expr: Any) -> Any:
    """Simplify an expression using only rules that are exact under three-valued logic."""
    if isinstance(expr, BinaryOp):
        left, right = fold(expr.left), fold(expr.right)
        op = op_name(expr.op)
        if op == "AND":
            return _and(left, right, expr)
        if op == "OR":
            return _or(left, right, expr)
        if isinstance(left, Literal) and isinstance(right, Literal):
            folded = _fold_literals(op, left, right)
            if folded is not None:
                return folded
        if op in _NULL_PROPAGATING and (_is_null_literal(left) or _is_null_literal(right)):
            return Literal(value=None, dtype=DType.BOOL if op in _COMPARISON else _null_type(left, right))
        return _rebuilt(expr, left=left, right=right)
    if isinstance(expr, UnaryOp):
        operand = fold(expr.operand)
        op = op_name(expr.op)
        if op == "NOT":
            if isinstance(operand, Literal) and operand.dtype == DType.BOOL:
                return operand if operand.value is None else Literal(value=not operand.value, dtype=DType.BOOL)
            if isinstance(operand, UnaryOp) and op_name(operand.op) == "NOT":
                return operand.operand
        if op in ("IS NULL", "IS NOT NULL") and isinstance(operand, Literal):
            return Literal(value=(operand.value is None) == (op == "IS NULL"), dtype=DType.BOOL)
        if op == "-" and isinstance(operand, Literal) and operand.dtype in _NUMERIC:
            return operand if operand.value is None else Literal(value=-operand.value, dtype=operand.dtype)
        return _rebuilt(expr, operand=operand)
    if dataclasses.is_dataclass(expr) and not isinstance(expr, (ColumnRef, Literal, type)):
        changes = {}
        for f in dataclasses.fields(expr):
            old = getattr(expr, f.name)
            new = fold(old) if dataclasses.is_dataclass(old) else old
            if new is not old:
                changes[f.name] = new
        return dataclasses.replace(expr, **changes) if changes else expr
    return expr


def simplify_predicate(expr: Any) -> Any:
    """Simplify an expression whose only job is to keep or reject rows. NULL and FALSE both reject."""
    folded = fold(expr)
    conjuncts: list[Any] = []
    for c in split_conjuncts(folded):
        if _is_false_or_null(c):
            return FALSE
        if c != TRUE and c not in conjuncts:
            conjuncts.append(c)
    if _contradictory(conjuncts):
        return FALSE
    return rebuild(conjuncts, folded) or TRUE


def _and(left, right, original):
    if left == FALSE or right == FALSE:
        return FALSE
    if left == TRUE:
        return right
    if right == TRUE or left == right:
        return left
    return _rebuilt(original, left=left, right=right)


def _or(left, right, original):
    if left == TRUE or right == TRUE:
        return TRUE
    if left == FALSE:
        return right
    if right == FALSE or left == right:
        return left
    return _rebuilt(original, left=left, right=right)


def _fold_literals(op: str, a: Literal, b: Literal) -> Literal | None:
    if a.value is None or b.value is None:
        return None  # handled by NULL propagation, for operators known to propagate
    if op in _COMPARISON:
        if not _comparable(a, b):
            return None
        return Literal(value=_COMPARISON[op](a.value, b.value), dtype=DType.BOOL)
    if op in _ARITHMETIC and a.dtype in _NUMERIC and b.dtype in _NUMERIC and _is_number(a) and _is_number(b):
        if op == "/" and (b.value == 0 or (a.dtype == DType.INT and b.dtype == DType.INT)):
            return None
        dtype = DType.FLOAT if DType.FLOAT in (a.dtype, b.dtype) else DType.INT
        value = _ARITHMETIC[op](a.value, b.value)
        if dtype == DType.INT and not _INT64[0] <= value <= _INT64[1]:
            return None
        return Literal(value=value, dtype=dtype)
    return None


def _comparable(a: Literal, b: Literal) -> bool:
    if a.dtype in _NUMERIC and b.dtype in _NUMERIC:
        return _is_number(a) and _is_number(b)
    if _odd_date(a) or _odd_date(b):
        return False
    return a.dtype == b.dtype and type(a.value) is type(b.value)


def _odd_date(lit: Literal) -> bool:
    """A string that reads as a date but isn't written YYYY-MM-DD, like '19950101'."""
    if not isinstance(lit.value, str):
        return False
    try:
        return datetime.date.fromisoformat(lit.value).isoformat() != lit.value
    except ValueError:
        return False


def _is_number(lit: Literal) -> bool:
    return isinstance(lit.value, (int, float)) and not isinstance(lit.value, bool)


def _is_null_literal(expr: Any) -> bool:
    return isinstance(expr, Literal) and expr.value is None


def _is_false_or_null(expr: Any) -> bool:
    return isinstance(expr, Literal) and (expr.value is None or expr.value is False)


def _null_type(left: Any, right: Any) -> DType:
    types = [e.dtype for e in (left, right) if isinstance(e, Literal)]
    return DType.FLOAT if DType.FLOAT in types else types[0]


def _rebuilt(original: Any, **fields: Any) -> Any:
    if all(getattr(original, k) is v for k, v in fields.items()):
        return original
    return dataclasses.replace(original, **fields)


# Contradiction detection: per column, the equalities, exclusions and bounds
# that its conjuncts impose against non-NULL literals.

_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "=", "!=": "!=", "<>": "<>"}


def _contradictory(conjuncts: list[Any]) -> bool:
    constraints: dict[Ref, list[tuple[str, Any]]] = {}
    for c in conjuncts:
        if not (isinstance(c, BinaryOp) and op_name(c.op) in _FLIP):
            continue
        op, left, right = op_name(c.op), c.left, c.right
        if isinstance(left, Literal) and isinstance(right, ColumnRef):
            op, left, right = _FLIP[op], right, left
        if isinstance(left, ColumnRef) and isinstance(right, Literal) and right.value is not None:
            constraints.setdefault((left.table, left.name), []).append((op, right))
    return any(_unsatisfiable(cs) for cs in constraints.values())


def _unsatisfiable(constraints: list[tuple[str, Literal]]) -> bool:
    literals = [lit for _, lit in constraints]
    if not all(_comparable(literals[0], other) for other in literals[1:]):
        return False  # mixed types: don't guess
    equal = {lit.value for op, lit in constraints if op == "="}
    if len(equal) > 1:
        return True
    lower = upper = None  # (value, inclusive)
    for op, lit in constraints:
        v = lit.value
        if op in (">", ">=") and (lower is None or v > lower[0] or (v == lower[0] and op == ">")):
            lower = (v, op == ">=")
        if op in ("<", "<=") and (upper is None or v < upper[0] or (v == upper[0] and op == "<")):
            upper = (v, op == "<=")
    if equal:
        (v,) = equal
        if any(op in ("!=", "<>") and lit.value == v for op, lit in constraints):
            return True
        if lower and (v < lower[0] or (v == lower[0] and not lower[1])):
            return True
        if upper and (v > upper[0] or (v == upper[0] and not upper[1])):
            return True
    if lower and upper:
        return lower[0] > upper[0] or (lower[0] == upper[0] and not (lower[1] and upper[1]))
    return False


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------


def is_empty(node: Any) -> bool:
    """Return True if ``node`` is the empty-result marker: a Filter whose predicate is the literal FALSE."""
    return isinstance(node, Filter) and node.predicate == FALSE


def _empty(node: Any) -> Any:
    return Filter(child=node, predicate=FALSE)


def _unwrap(node: Any) -> Any:
    return node.child if is_empty(node) else node


class _Folder:
    def __init__(self, catalog: Any):
        self.catalog = catalog

    def plan(self, node: Any) -> Any:
        children = [self.plan(c) for c in node.children]
        if any(new is not old for new, old in zip(children, node.children)):
            node = node.replace_children(tuple(children))

        if isinstance(node, Filter):
            return self._filter(node)
        if isinstance(node, Scan):
            return self._scan(node)
        if isinstance(node, Project):
            node = _rebuilt(node, exprs=_fold_pairs(node.exprs))
            if is_empty(node.child):
                return _empty(dataclasses.replace(node, child=node.child.child))
            return node.child if self._is_noop(node) else node
        if isinstance(node, Sort):
            node = _rebuilt(node, keys=_fold_pairs(node.keys))
            return self._lift(node)
        if isinstance(node, Limit):
            if node.n == 0 and not is_empty(node.child):
                return _empty(node.child)
            return self._lift(node)
        if isinstance(node, Aggregate):
            node = _rebuilt(node, group_keys=_fold_list(node.group_keys), aggs=_fold_pairs(node.aggs))
            # A global aggregate over no rows still returns one row, so it stops the lift.
            return self._lift(node) if node.group_keys else node
        if isinstance(node, Join):
            return self._join(node)
        return node

    def _filter(self, node: Filter) -> Any:
        predicate = simplify_predicate(node.predicate)
        if is_empty(node.child):
            return node.child
        if predicate == TRUE:
            return node.child
        return _rebuilt(node, predicate=predicate)

    def _scan(self, node: Scan) -> Any:
        if node.pushed_predicate is None:
            return node
        predicate = simplify_predicate(node.pushed_predicate)
        if predicate == TRUE:
            return dataclasses.replace(node, pushed_predicate=None)
        if predicate == FALSE:
            return _empty(dataclasses.replace(node, pushed_predicate=None))
        return _rebuilt(node, pushed_predicate=predicate)

    def _join(self, node: Join) -> Any:
        node = _rebuilt(node, condition=simplify_predicate(node.condition))
        if node.kind == "inner" and (node.condition == FALSE or is_empty(node.left) or is_empty(node.right)):
            return _empty(dataclasses.replace(node, left=_unwrap(node.left), right=_unwrap(node.right)))
        if node.kind == "left" and is_empty(node.left):
            return _empty(dataclasses.replace(node, left=node.left.child))
        return node

    def _lift(self, node: Any) -> Any:
        """Lift an empty marker from a single-child node's input to above the node itself."""
        if is_empty(node.child):
            return _empty(node.replace_children((node.child.child,)))
        return node

    def _is_noop(self, node: Project) -> bool:
        """Return True if the Project passes its input through unchanged.

        It must emit every input column, once, in input order, under the
        column's own name. The input must not come straight from a Join:
        how codegen names a join's output columns isn't pinned down, so the
        Project may be what gives them their plain names.
        """
        outputs = self._plain_outputs(node.child)
        if outputs is None or len(outputs) != len(node.exprs):
            return False
        names = [n for _, n in outputs]
        if len(set(names)) != len(names):
            return False
        return all(
            isinstance(e, ColumnRef) and e.name == n == alias and e.table in (None, q)
            for (e, alias), (q, n) in zip(node.exprs, outputs)
        )

    def _plain_outputs(self, node: Any) -> list[Ref] | None:
        if isinstance(node, Scan):
            if node.columns is not None:
                return [(node.table, c) for c in node.columns]
            schema = table_schema(node, self.catalog)
            return None if schema is None else [(node.table, n) for n, _ in schema]
        if isinstance(node, (Filter, Sort, Limit)):
            return self._plain_outputs(node.child)
        if isinstance(node, Project):
            return [(None, a) for _, a in node.exprs]
        if isinstance(node, Aggregate) and all(isinstance(k, ColumnRef) for k in node.group_keys):
            return [(k.table, k.name) for k in node.group_keys] + [(None, a) for _, a in node.aggs]
        return None


def _fold_list(exprs: list[Any]) -> list[Any]:
    folded = [fold(e) for e in exprs]
    return exprs if all(n is o for n, o in zip(folded, exprs)) else folded


def _fold_pairs(pairs: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    folded = [(fold(e), extra) for e, extra in pairs]
    return pairs if all(n[0] is o[0] for n, o in zip(folded, pairs)) else folded
