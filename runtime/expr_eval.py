"""Row-at-a-time expression evaluation with SQL NULL semantics.

This is the *reference* semantics for the whole project. Generated code (Phases C3+)
must agree with what is written here. Decisions, stated once:

  * Any arithmetic or comparison with a NULL operand yields NULL.
  * AND / OR use Kleene three-valued logic (FALSE AND NULL = FALSE, TRUE OR NULL = TRUE).
  * `/` always yields FLOAT (DuckDB semantics). Division or modulo by zero yields NULL.
  * `%` truncates toward zero like SQL (-7 % 3 = -1), not like Python (-7 % 3 = 2).
  * Comparing a DATE with a string literal parses the string as an ISO date.

Expressions are evaluated against a `Scope`: the input table's column layout plus a
map from already-computed expressions (aggregate calls, computed group keys) to the
column holding their value. That map is how `HAVING sum(x) > 10` above an Aggregate
finds the `sum(x)` the Aggregate produced.
"""
from __future__ import annotations

import datetime
import math
import re

from runtime._compat import DType


class InterpreterError(Exception):
    pass


# ------------------------------------------------------------------ helpers

def node_kind(node) -> str:
    """Class name used for dispatch; walks the MRO so test subclasses (MockScan) work."""
    for cls in type(node).__mro__:
        if cls.__name__ in _KNOWN_KINDS:
            return cls.__name__
    raise InterpreterError(f"unknown IR node type {type(node).__name__}")


_KNOWN_KINDS = {
    "ColumnRef", "Literal", "BinaryOp", "UnaryOp", "AggCall",
    "Scan", "Filter", "Project", "Join", "Aggregate", "Sort", "Limit",
}


def expr_key(expr) -> str:
    """Stable identity for an expression (dataclass repr is structural)."""
    return repr(expr)


def project_expr_columns(exprs, index_of, expr_columns: dict[str, int]) -> dict[str, int]:
    """The already-computed expressions a Project's output still holds, by output position.

    Output column i holds expression e, written over the Project's input. A computed e
    is held as itself. A plain column also carries whatever expression the input had
    computed into it, so an Aggregate's sum(x) renamed `total` is still sum(x) to the
    operators above: `SELECT sum(x) AS total ... ORDER BY sum(x)` sorts on `total`.
    """
    out: dict[str, int] = {}
    for i, (expr, _alias) in enumerate(exprs):
        if node_kind(expr) == "ColumnRef":
            src = index_of(expr)
            for key, j in expr_columns.items():
                if j == src:
                    out.setdefault(key, i)
        else:
            out.setdefault(expr_key(expr), i)
    return out


def render_expr(expr) -> str:
    """SQL-ish text for an expression; used for auto-generated column names."""
    kind = node_kind(expr)
    if kind == "ColumnRef":
        return f"{expr.table}.{expr.name}" if expr.table else expr.name
    if kind == "Literal":
        if expr.value is None:
            return "NULL"
        if isinstance(expr.value, str) or expr.dtype.name == "DATE":
            return f"'{expr.value}'"
        return str(expr.value)
    if kind == "BinaryOp":
        return f"({render_expr(expr.left)} {expr.op} {render_expr(expr.right)})"
    if kind == "UnaryOp":
        op = _norm(expr.op)
        if op in _POSTFIX:
            return f"({render_expr(expr.operand)} {_POSTFIX[op]})"
        return f"({op} {render_expr(expr.operand)})"
    if kind == "AggCall":
        return f"{expr.func}({'*' if expr.arg is None else render_expr(expr.arg)})"
    raise InterpreterError(f"cannot render {expr!r}")


def _norm(op: str) -> str:
    return " ".join(op.upper().split())


_POSTFIX = {"IS NULL": "IS NULL", "ISNULL": "IS NULL",
            "IS NOT NULL": "IS NOT NULL", "NOTNULL": "IS NOT NULL"}
_ARITH = {"+", "-", "*", "/", "%"}
_COMPARE = {"=", "==", "!=", "<>", "<", "<=", ">", ">="}
_BOOL = {"AND", "OR"}


# ------------------------------------------------------------------ scope

class Scope:
    """Column layout of the rows an expression is evaluated against."""

    def __init__(self, table, expr_columns: dict[str, int] | None = None):
        self.columns = table.columns
        self.table = table
        self.expr_columns = expr_columns or {}
        self._cache: dict[tuple, int] = {}

    def index_of(self, ref) -> int:
        key = (ref.table, ref.name)
        if key not in self._cache:
            hits = self.table.find(ref.name, ref.table)
            label = f"{ref.table}.{ref.name}" if ref.table else ref.name
            if not hits:
                raise InterpreterError(
                    f"unknown column {label!r}; available: {self.table.qualified_names()}")
            if len(hits) > 1:
                raise InterpreterError(
                    f"ambiguous column {label!r}; available: {self.table.qualified_names()}")
            self._cache[key] = hits[0]
        return self._cache[key]


# ------------------------------------------------------------------ evaluation

def evaluate(expr, row: tuple, scope: Scope):
    """Value of `expr` on one row. Returns a Python value or None for NULL."""
    kind = node_kind(expr)

    if kind == "ColumnRef":
        return row[scope.index_of(expr)]

    if kind == "Literal":
        return _literal_value(expr)

    precomputed = scope.expr_columns.get(expr_key(expr))
    if precomputed is not None:
        return row[precomputed]

    if kind == "AggCall":
        raise InterpreterError(
            f"aggregate {render_expr(expr)} used outside an Aggregate "
            f"(or not produced by the Aggregate below)")

    if kind == "UnaryOp":
        op = _norm(expr.op)
        v = evaluate(expr.operand, row, scope)
        if op == "NOT":
            return None if v is None else (not v)
        if op == "-":
            return None if v is None else -v
        if op == "+":
            return v
        if op in ("IS NULL", "ISNULL"):
            return v is None
        if op in ("IS NOT NULL", "NOTNULL"):
            return v is not None
        raise InterpreterError(f"unsupported unary operator {expr.op!r}")

    if kind == "BinaryOp":
        op = _norm(expr.op)
        if op in _BOOL:
            return _kleene(op, expr, row, scope)
        a = evaluate(expr.left, row, scope)
        b = evaluate(expr.right, row, scope)
        if a is None or b is None:
            return None
        if op in _ARITH:
            return _arith(op, a, b)
        if op in _COMPARE:
            a, b = _coerce_for_compare(a, b)
            return _compare(op, a, b)
        if op == "||":
            return f"{a}{b}"
        if op == "LIKE":
            return _like(a, b)
        if op == "NOT LIKE":
            return not _like(a, b)
        raise InterpreterError(f"unsupported binary operator {expr.op!r}")

    raise InterpreterError(f"cannot evaluate {kind} as an expression")


def is_true(value) -> bool:
    """WHERE / ON / HAVING keep a row only when the predicate is TRUE (not NULL)."""
    return value is True


def _literal_value(lit):
    if lit.value is None:
        return None
    if lit.dtype.name == "DATE":
        if isinstance(lit.value, datetime.date):
            return lit.value
        return datetime.date.fromisoformat(str(lit.value))
    return lit.value


def _kleene(op, expr, row, scope):
    a = evaluate(expr.left, row, scope)
    if op == "AND":
        if a is False:
            return False
        b = evaluate(expr.right, row, scope)
        if b is False:
            return False
        return None if (a is None or b is None) else True
    # OR
    if a is True:
        return True
    b = evaluate(expr.right, row, scope)
    if b is True:
        return True
    return None if (a is None or b is None) else False


def _arith(op, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return None if b == 0 else a / b
    # % : SQL truncating modulo
    if b == 0:
        return None
    if isinstance(a, float) or isinstance(b, float):
        return math.fmod(a, b)
    r = abs(a) % abs(b)
    return -r if a < 0 else r


def _coerce_for_compare(a, b):
    if isinstance(a, datetime.date) and isinstance(b, str):
        b = datetime.date.fromisoformat(b)
    elif isinstance(b, datetime.date) and isinstance(a, str):
        a = datetime.date.fromisoformat(a)
    return a, b


def _compare(op, a, b):
    if op in ("=", "=="):
        return a == b
    if op in ("!=", "<>"):
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


def _like(value: str, pattern: str) -> bool:
    regex = "".join(".*" if ch == "%" else "." if ch == "_" else re.escape(ch)
                    for ch in pattern)
    return re.fullmatch(regex, value, flags=re.DOTALL) is not None


# ------------------------------------------------------------------ typing

def infer_dtype(expr, scope: Scope) -> DType:
    """Output DType of `expr` evaluated in `scope`."""
    kind = node_kind(expr)
    if kind == "ColumnRef":
        return scope.columns[scope.index_of(expr)].dtype
    if kind == "Literal":
        return expr.dtype
    precomputed = scope.expr_columns.get(expr_key(expr))
    if precomputed is not None:
        return scope.columns[precomputed].dtype
    if kind == "UnaryOp":
        op = _norm(expr.op)
        if op in ("-", "+"):
            return infer_dtype(expr.operand, scope)
        return DType.BOOL
    if kind == "BinaryOp":
        op = _norm(expr.op)
        if op in _ARITH:
            if op == "/":
                return DType.FLOAT
            lt = infer_dtype(expr.left, scope)
            rt = infer_dtype(expr.right, scope)
            if lt.name == "FLOAT" or rt.name == "FLOAT":
                return DType.FLOAT
            if lt.name == "INT" and rt.name == "INT":
                return DType.INT
            raise InterpreterError(
                f"arithmetic {op!r} on {lt.name} and {rt.name}: {render_expr(expr)}")
        if op == "||":
            return DType.STRING
        return DType.BOOL
    if kind == "AggCall":
        raise InterpreterError(f"aggregate {render_expr(expr)} used outside an Aggregate")
    raise InterpreterError(f"cannot type {kind}")


def agg_result_dtype(agg, scope: Scope) -> DType:
    func = agg.func.lower()
    if func == "count":
        return DType.INT
    if func == "avg":
        return DType.FLOAT
    arg_type = infer_dtype(agg.arg, scope)
    if func == "sum" and arg_type.name not in ("INT", "FLOAT"):
        raise InterpreterError(f"sum over {arg_type.name}: {render_expr(agg)}")
    return arg_type
