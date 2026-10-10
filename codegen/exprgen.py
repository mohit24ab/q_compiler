"""Translate an IR expression into Python source over numpy column arrays.

Every expression compiles to TWO pieces of source text:

    v   the values      e.g.  ((qty_1 + 1) * 2)
    ok  where non-NULL  e.g.  (qty_1_ok & amount_1_ok)

`ok` may also be the constants "True" (never NULL) or "False" (always NULL); those are
folded away at generation time, so a column known to be non-NULL costs nothing.

Rules (they mirror runtime/expr_eval.py exactly — the interpreter is the spec):
  * Every binary operation is parenthesised, unconditionally. Ugly but never wrong.
  * Arithmetic / comparison:  ok = a.ok & b.ok.
  * `/` and `%` by zero give NULL: the divisor is swapped for 1 and ok gets `& (b != 0)`.
  * `%` is np.fmod (truncates toward zero, like SQL), never Python's `%`.
  * AND / OR are Kleene logic, written as mask algebra:
        AND: v = a & b      ok = (a.ok & b.ok) | (a.ok & ~a) | (b.ok & ~b)
        OR:  v = a | b      ok = (a.ok & b.ok) | (a.ok & a)  | (b.ok & b)
    i.e. a known FALSE decides AND, a known TRUE decides OR, else NULL wins.
  * Values sitting under a NULL are garbage and are never read without their ok mask.
"""
from __future__ import annotations

import ast
import datetime
import math
from dataclasses import dataclass

import numpy as np

from runtime.expr_eval import Scope, _norm, expr_key, infer_dtype, node_kind, render_expr
from runtime.table import find_columns


class CodegenError(Exception):
    pass


# ------------------------------------------------------------------ compile-time layout

@dataclass
class CVar:
    """One column as the generated code sees it: two variable names (or constants)."""
    name: str
    dtype: object
    table: str | None
    v: str | None = None    # None = not loaded yet (Scan loads lazily)
    ok: str | None = None   # a variable name, or the constant "True" / "False"


class Rel:
    """A relation at generation time: its columns and the expression for its row count.

    `expr_columns` maps already-computed expressions (an Aggregate's `sum(x)`, a computed
    group key) to the column holding them — the same map the interpreter keeps, so
    `HAVING sum(x) > 10` above an Aggregate reads the Aggregate's output column.
    """

    def __init__(self, columns: list[CVar], n: str, expr_columns: dict[str, int] | None = None):
        self.columns = columns
        self.n = n
        self.expr_columns = expr_columns or {}

    def find(self, name, table=None):
        return find_columns(self.columns, name, table)

    def qualified_names(self):
        return [f"{c.table}.{c.name}" if c.table else c.name for c in self.columns]


@dataclass
class Code:
    v: str
    ok: str
    scalar: bool  # True when `v` involves no column array (a constant)


# ------------------------------------------------------------------ mask folding

_CONST = {"np.True_": "True", "np.False_": "False"}


def and_ok(*parts: str) -> str:
    parts = [_CONST.get(p, p) for p in parts]
    parts = list(dict.fromkeys(p for p in parts if p != "True"))  # drop TRUEs and repeats
    if "False" in parts:
        return "False"
    if not parts:
        return "True"
    return parts[0] if len(parts) == 1 else "(" + " & ".join(parts) + ")"


def or_ok(*parts: str) -> str:
    parts = [_CONST.get(p, p) for p in parts]
    parts = list(dict.fromkeys(p for p in parts if p != "False"))
    if "True" in parts:
        return "True"
    if not parts:
        return "False"
    return parts[0] if len(parts) == 1 else "(" + " | ".join(parts) + ")"


def logical_not(v: str, scalar: bool) -> str:
    # `~` on a Python bool is bitwise (~True == -2), so constants use np.logical_not.
    return f"np.logical_not({v})" if scalar else f"(~{v})"


_NP_FULL_DTYPE = {"INT": "np.int64", "FLOAT": "np.float64", "BOOL": "bool",
                  "STRING": "object", "DATE": "'datetime64[D]'"}
_FILLER_SRC = {"INT": "0", "FLOAT": "0.0", "BOOL": "np.False_", "STRING": "''",
               "DATE": "np.datetime64('1970-01-01', 'D')"}


def literal_src(value, dtype) -> str:
    name = dtype.name
    if value is None:
        return _FILLER_SRC[name]
    if name == "DATE":
        if isinstance(value, datetime.datetime):
            value = value.date()
        if not isinstance(value, datetime.date):
            # the interpreter's parser, so both engines read '20240105' as 5 Jan 2024
            # (np.datetime64 would read it as the year 20240105); fails at generation time
            value = datetime.date.fromisoformat(str(value))
        return f"np.datetime64({value.isoformat()!r}, 'D')"
    if name == "BOOL":
        return "np.True_" if value else "np.False_"
    if name == "FLOAT":
        value = float(value)
        return f"float({str(value)!r})" if (math.isnan(value) or math.isinf(value)) else repr(value)
    if name == "INT":
        return repr(int(value))
    return repr(str(value))


def full_src(v: str, n: str, dtype) -> str:
    """Broadcast a constant to a column of length n."""
    return f"np.full({n}, {v}, dtype={_NP_FULL_DTYPE[dtype.name]})"


# ------------------------------------------------------------------ generator

_CMP = {"=": "==", "==": "==", "!=": "!=", "<>": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">="}


class ExprGen:
    """Compiles expressions against one Rel, emitting temporaries into `em` as needed."""

    def __init__(self, em, rel: Rel, load=None):
        self.em = em
        self.rel = rel
        self.scope = Scope(rel, rel.expr_columns)
        self.load = load  # callback(CVar) that emits code loading a lazy column
        self.scalar_temps: set[str] = set()

    def is_scalar(self, src: str) -> bool:
        """True when `src` reads no column array: only literals and constant temporaries.

        Derived from the source text rather than tracked as a flag, because mask folding
        can drop the array part of an expression (TRUE OR <x> is simply TRUE).
        """
        for node in ast.walk(ast.parse(src, mode="eval")):
            if isinstance(node, ast.Name) and node.id not in ("np", "float", "parse_dates") \
                    and node.id not in self.scalar_temps:
                return False
        return True

    def code(self, v: str, ok: str) -> Code:
        return Code(v, ok, self.is_scalar(v))

    def dtype(self, expr):
        return infer_dtype(expr, self.scope)

    def gen(self, expr) -> Code:
        kind = node_kind(expr)
        if kind == "ColumnRef":
            col = self.rel.columns[self.scope.index_of(expr)]
            if col.v is None:
                self.load(col)
            return self.code(col.v, col.ok)
        if kind == "Literal":
            return self.code(literal_src(expr.value, expr.dtype),
                             "False" if expr.value is None else "True")
        computed = self.rel.expr_columns.get(expr_key(expr))
        if computed is not None:  # e.g. sum(x) produced by the Aggregate below
            col = self.rel.columns[computed]
            return self.code(col.v, col.ok)
        if kind == "UnaryOp":
            return self._unary(expr)
        if kind == "BinaryOp":
            return self._binary(expr)
        if kind == "AggCall":
            raise CodegenError(f"aggregate {render_expr(expr)} is not produced by an Aggregate below")
        raise CodegenError(f"cannot compile {kind}")

    # ---------------------------------------------------------------- unary
    def _unary(self, expr) -> Code:
        op = _norm(expr.op)
        a = self.gen(expr.operand)
        if op == "NOT":
            return self.code(logical_not(a.v, a.scalar), a.ok)
        if op == "-":
            return self.code(f"(-{a.v})", a.ok)
        if op == "+":
            return a
        if op in ("IS NULL", "ISNULL", "IS NOT NULL", "NOTNULL"):
            present = {"True": "np.True_", "False": "np.False_"}.get(a.ok, a.ok)
            if op in ("IS NOT NULL", "NOTNULL"):
                return self.code(present, "True")
            return self.code(logical_not(present, self.is_scalar(present)), "True")
        raise CodegenError(f"unsupported unary operator {expr.op!r}")

    # ---------------------------------------------------------------- binary
    def _binary(self, expr) -> Code:
        op = _norm(expr.op)
        if op in ("AND", "OR"):
            return self._kleene(op, expr)

        a, b = self.gen(expr.left), self.gen(expr.right)
        if op in _CMP:
            a, b = self._coerce_dates(expr, a, b)
        ok = and_ok(a.ok, b.ok)

        if op in ("+", "-", "*"):
            return self.code(f"({a.v} {op} {b.v})", ok)
        if op in ("/", "%"):
            return self._divide(op, a, b, ok)
        if op in _CMP:
            return self.code(f"({a.v} {_CMP[op]} {b.v})", ok)
        if op == "||":
            return self.code(f"({a.v} + {b.v})", ok)
        if op in ("LIKE", "NOT LIKE"):
            if not b.scalar or a.scalar:
                raise CodegenError("LIKE needs a column on the left and a literal pattern")
            v = f"like({a.v}, {b.v})"
            return self.code(v if op == "LIKE" else f"(~{v})", ok)
        raise CodegenError(f"unsupported binary operator {expr.op!r}")

    def _divide(self, op, a: Code, b: Code, ok: str) -> Code:
        fn = "np.true_divide" if op == "/" else "np.fmod"
        if b.ok == "True" and _is_constant(b.v):
            if float(eval(b.v, {"np": np, "float": float})) == 0:
                return self.code("0", "False")             # x / 0 is always NULL
            return self.code(f"{fn}({a.v}, {b.v})", ok)
        divisor = self.name(b.v, "divisor")
        return self.code(f"{fn}({a.v}, np.where({divisor} == 0, 1, {divisor}))",
                         and_ok(ok, f"({divisor} != 0)"))

    def _kleene(self, op, expr) -> Code:
        a, b = self.gen(expr.left), self.gen(expr.right)
        av, bv = self.name(a.v, "lhs"), self.name(b.v, "rhs")
        aok, bok = self.name(a.ok, "lhs_ok"), self.name(b.ok, "rhs_ok")
        if op == "AND":
            v = f"({av} & {bv})"
            ok = or_ok(and_ok(aok, bok),
                       and_ok(aok, logical_not(av, a.scalar)),
                       and_ok(bok, logical_not(bv, b.scalar)))
        else:
            v = f"({av} | {bv})"
            ok = or_ok(and_ok(aok, bok), and_ok(aok, av), and_ok(bok, bv))
        return self.code(v, ok)

    def _coerce_dates(self, expr, a: Code, b: Code):
        """A DATE compared with a STRING compares as dates, the string parsed as an ISO
        date like the interpreter's _coerce_for_compare: a literal (`day >= '2024-01-01'`)
        at generation time, anything else (a STRING column) row by row at run time."""
        lt, rt = self.dtype(expr.left), self.dtype(expr.right)
        if lt.name == "DATE" and rt.name == "STRING":
            b = self._as_date(expr.right, b, lt, and_ok(a.ok, b.ok))
        elif rt.name == "DATE" and lt.name == "STRING":
            a = self._as_date(expr.left, a, rt, and_ok(a.ok, b.ok))
        return a, b

    def _as_date(self, expr, code: Code, date_dtype, both_ok: str) -> Code:
        if node_kind(expr) == "Literal":
            return self.code(literal_src(expr.value, date_dtype), code.ok)
        if both_ok == "False":  # the comparison is NULL on every row: nothing to parse
            return self.code(_FILLER_SRC["DATE"], code.ok)
        # only rows where both sides are present are parsed, as the interpreter only
        # parses a string it actually compares (a NULL on either side short-circuits)
        mask = "None" if both_ok == "True" else both_ok
        return self.code(f"parse_dates({code.v}, {mask})", code.ok)

    # ---------------------------------------------------------------- temporaries
    def name(self, src: str, hint: str) -> str:
        """Bind `src` to a temporary unless it is already a name or a constant."""
        if src.isidentifier() or src in ("True", "False") or _is_constant(src):
            return src
        var = self.em.fresh(hint)
        if self.is_scalar(src):
            self.scalar_temps.add(var)
        self.em.line(f"{var} = {src}")
        return var


def _is_constant(src: str) -> bool:
    try:
        compile(src, "<c>", "eval")
    except SyntaxError:
        return False
    return src.startswith(("np.True_", "np.False_", "np.datetime64(", "float(", "'", '"')) \
        or src.lstrip("-").replace(".", "", 1).isdigit()


def column_refs(expr) -> list:
    """Every ColumnRef inside `expr` (empty for None)."""
    if expr is None:
        return []
    kind = node_kind(expr)
    if kind == "ColumnRef":
        return [expr]
    if kind == "Literal":
        return []
    if kind == "UnaryOp":
        return column_refs(expr.operand)
    if kind == "BinaryOp":
        return column_refs(expr.left) + column_refs(expr.right)
    if kind == "AggCall":
        return column_refs(expr.arg)
    raise CodegenError(f"cannot inspect {kind}")


def conjuncts(expr) -> list:
    """Split `a AND b AND c` into [a, b, c]."""
    if expr is not None and node_kind(expr) == "BinaryOp" and _norm(expr.op) == "AND":
        return conjuncts(expr.left) + conjuncts(expr.right)
    return [] if expr is None else [expr]
