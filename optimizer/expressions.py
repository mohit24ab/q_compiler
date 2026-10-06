"""Expression helpers for rewrite passes.

Every helper here returns a new expression and never mutates its input.
Where nothing changes, the original object is returned unchanged.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from ir.dtype import DType
from ir.expr import BinaryOp, ColumnRef, Literal, UnaryOp

from optimizer.columns import Ref

TRUE = Literal(value=True, dtype=DType.BOOL)


def op_name(op: str) -> str:
    """Return the canonical spelling of an operator: ``"is_not_null"`` becomes ``"IS NOT NULL"``.

    Person A's canonical spellings are uppercase with single spaces
    ("IS NULL", "IS NOT NULL"). Underscore variants ("IS_NULL",
    "IS_NOT_NULL") must also be accepted. Every operator comparison in the
    optimizer goes through this function.
    """
    return op.upper().replace("_", " ")


def split_conjuncts(expr: Any) -> list[Any]:
    """Flatten nested ANDs: ``a AND (b AND c)`` becomes ``[a, b, c]``. ``None`` gives ``[]``."""
    if expr is None:
        return []
    if isinstance(expr, BinaryOp) and op_name(expr.op) == "AND":
        return split_conjuncts(expr.left) + split_conjuncts(expr.right)
    return [expr]


def conjoin(conjuncts: list[Any]) -> Any | None:
    """Rebuild a left-deep AND from conjuncts, the inverse of ``split_conjuncts``. ``[]`` gives ``None``."""
    if not conjuncts:
        return None
    result = conjuncts[0]
    for conjunct in conjuncts[1:]:
        result = BinaryOp(op="AND", left=result, right=conjunct)
    return result


def rebuild(conjuncts: list[Any], original: Any) -> Any | None:
    """Return ``conjoin(conjuncts)``, but reuse ``original`` if it already holds exactly these conjuncts.

    Reusing the original keeps a pass from reporting a change when all it
    did was split a predicate and put it back together.
    """
    if conjuncts == split_conjuncts(original):
        return original
    return conjoin(conjuncts)


def substitute(expr: Any, mapping: dict[Ref, Any]) -> Any:
    """Replace each ``ColumnRef`` whose ``(table, name)`` appears in ``mapping`` with the mapped expression."""
    if isinstance(expr, ColumnRef):
        return mapping.get((expr.table, expr.name), expr)
    if dataclasses.is_dataclass(expr) and not isinstance(expr, type):
        changes = {}
        for f in dataclasses.fields(expr):
            old = getattr(expr, f.name)
            new = substitute(old, mapping)
            if new is not old:
                changes[f.name] = new
        return dataclasses.replace(expr, **changes) if changes else expr
    if isinstance(expr, (list, tuple)):
        items = [substitute(item, mapping) for item in expr]
        if all(new is old for new, old in zip(items, expr)):
            return expr
        return type(expr)(items)
    return expr


# --------------------------------------------------------------------------
# Null rejection
# --------------------------------------------------------------------------

# A boolean expression's possible outcomes under SQL's three-valued logic.
T, F, N = "T", "F", "N"
_ANY = frozenset({T, F, N})

# Operators known to return NULL whenever an operand is NULL. An operator
# not listed here (IS DISTINCT FROM, a differently spelled IS NULL, ...)
# is never assumed to return NULL: assuming it would make an IS-NULL-style
# test look null-rejecting, and an anti-join would wrongly become INNER.
_NULL_PROPAGATING_BINARY = {"=", "!=", "<>", "<", "<=", ">", ">=", "+", "-", "*", "/", "%"}
_NULL_PROPAGATING_UNARY = {"-", "+"}

_AND = {(a, b): (F if F in (a, b) else N if N in (a, b) else T) for a in _ANY for b in _ANY}
_OR = {(a, b): (T if T in (a, b) else N if N in (a, b) else F) for a in _ANY for b in _ANY}
_NOT = {T: F, F: T, N: N}


def can_be_true(expr: Any, is_null: Callable[[ColumnRef], bool]) -> bool:
    """Return False only when ``expr`` cannot be TRUE given that the refs selected by ``is_null`` are NULL.

    If this returns False for a WHERE conjunct, with ``is_null`` selecting the
    columns of a LEFT join's right side, the conjunct is null-rejecting. It
    removes every NULL-extended row, so the LEFT join can safely become an
    INNER join. The analysis is conservative: anything it doesn't understand
    might be TRUE.
    """
    return T in _outcomes(expr, is_null)


def _outcomes(expr: Any, is_null) -> frozenset[str]:
    if isinstance(expr, BinaryOp):
        op = op_name(expr.op)
        if op in ("AND", "OR"):
            table = _AND if op == "AND" else _OR
            left, right = _outcomes(expr.left, is_null), _outcomes(expr.right, is_null)
            return frozenset(table[a, b] for a in left for b in right)
        return frozenset({N}) if _is_null(expr, is_null) else _ANY
    if isinstance(expr, UnaryOp):
        op = op_name(expr.op)
        if op == "NOT":
            return frozenset(_NOT[x] for x in _outcomes(expr.operand, is_null))
        if op == "IS NULL":
            return frozenset({T}) if _is_null(expr.operand, is_null) else frozenset({T, F})
        if op == "IS NOT NULL":
            return frozenset({F}) if _is_null(expr.operand, is_null) else frozenset({T, F})
        return frozenset({N}) if _is_null(expr, is_null) else _ANY
    if isinstance(expr, Literal):
        if expr.value is None:
            return frozenset({N})
        if expr.value is True:
            return frozenset({T})
        if expr.value is False:
            return frozenset({F})
    if isinstance(expr, ColumnRef) and is_null(expr):
        return frozenset({N})
    return _ANY


def _is_null(expr: Any, is_null) -> bool:
    """Return True if ``expr`` is certainly NULL: NULL propagates through comparisons and arithmetic."""
    if isinstance(expr, ColumnRef):
        return is_null(expr)
    if isinstance(expr, Literal):
        return expr.value is None
    if isinstance(expr, BinaryOp):
        op = op_name(expr.op)
        if op in ("AND", "OR"):
            return _outcomes(expr, is_null) == {N}
        if op in _NULL_PROPAGATING_BINARY:
            return _is_null(expr.left, is_null) or _is_null(expr.right, is_null)
        return False
    if isinstance(expr, UnaryOp):
        op = op_name(expr.op)
        if op == "NOT":
            return _outcomes(expr, is_null) == {N}
        if op in _NULL_PROPAGATING_UNARY:
            return _is_null(expr.operand, is_null)
        return False
    return False
