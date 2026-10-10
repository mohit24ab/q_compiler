"""Naive reference interpreter: `interpret(plan, tables) -> Table`.

Deliberately the slow path. Every operator converts its input to Python row tuples,
does the obvious thing, and builds a new Table. Nothing here is clever, because this is
the oracle every other component (optimizer, generated code) is checked against
(Contract §7).

The one concession: a join evaluates its condition on every *candidate* pair, and when
the condition has equality conjuncts (`l.a = r.b AND ...`) the candidates come from a
hash index on them instead of from all pairs, which is what makes a 100k x 25k join
finish. The index only skips pairs whose keys differ, which can never satisfy the
condition; the full condition is still evaluated on every pair that remains, so it alone
decides. Output order is unchanged: left rows in order, each with its matches in right
order.

Semantics not covered by expr_eval.py:
  * Aggregates ignore NULL inputs. count(*) counts rows; count(x) counts non-NULL x.
    sum/avg/min/max of zero non-NULL values is NULL. avg = sum / count, computed at the end.
  * A global aggregate (no GROUP BY) over zero rows still returns one row.
  * NULL group keys form a single group. Groups are output in first-seen order.
  * Join rows match only when the condition is TRUE, so NULL keys never match.
    A LEFT join pads unmatched left rows with NULLs.
  * Sort is stable and puts NULLs last for both ASC and DESC (DuckDB's default).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from runtime.expr_eval import (
    InterpreterError, Scope, _norm, agg_result_dtype, evaluate, expr_key, infer_dtype,
    is_true, node_kind, project_expr_columns, render_expr,
)
from runtime.table import Table, find_columns


@dataclass
class _Rel:
    """An operator's output: the table plus expressions it has already computed."""
    table: Table
    expr_columns: dict[str, int] = field(default_factory=dict)

    def scope(self) -> Scope:
        return Scope(self.table, self.expr_columns)


def interpret(plan, tables: dict) -> Table:
    """Execute `plan` against `tables` (name -> runtime Table or pyarrow.Table)."""
    return _Interpreter(tables).run(plan).table


class _Interpreter:
    def __init__(self, tables: dict):
        self.tables = tables

    def run(self, node) -> _Rel:
        kind = node_kind(node)
        method = getattr(self, f"_{kind.lower()}", None)
        if method is None:
            raise InterpreterError(f"{kind} is not a plan node")
        return method(node)

    # ------------------------------------------------------------------ Scan
    def _scan(self, node) -> _Rel:
        if node.table not in self.tables:
            raise InterpreterError(
                f"table {node.table!r} not provided; have {sorted(self.tables)}")
        source = self.tables[node.table]
        table = source if isinstance(source, Table) else Table.from_arrow(source)
        table = table.with_table_name(node.table)

        # The pushed predicate may reference columns that are not in `columns`,
        # so filter on the full table first, then narrow.
        if node.pushed_predicate is not None:
            table = self._apply_predicate(_Rel(table), node.pushed_predicate).table
        if node.columns is not None:
            table = Table([table.column(c) for c in node.columns], table.num_rows)
        return _Rel(table)

    # ------------------------------------------------------------------ Filter
    def _filter(self, node) -> _Rel:
        return self._apply_predicate(self.run(node.child), node.predicate)

    def _apply_predicate(self, rel: _Rel, predicate) -> _Rel:
        scope = rel.scope()
        keep = [i for i, row in enumerate(rel.table.to_rows())
                if is_true(evaluate(predicate, row, scope))]
        return _Rel(rel.table.take(keep), rel.expr_columns)

    # ------------------------------------------------------------------ Project
    def _project(self, node) -> _Rel:
        rel = self.run(node.child)
        scope = rel.scope()
        fields = []
        for expr, alias in node.exprs:
            qualifier = None
            # `SELECT orders.id AS id` keeps its table so a parent can still say orders.id
            if node_kind(expr) == "ColumnRef" and expr.name == alias:
                qualifier = scope.columns[scope.index_of(expr)].table
            fields.append((alias, infer_dtype(expr, scope), qualifier))
        rows = [tuple(evaluate(e, row, scope) for e, _ in node.exprs)
                for row in rel.table.to_rows()]
        return _Rel(Table.from_rows(fields, rows),
                    project_expr_columns(node.exprs, scope.index_of, rel.expr_columns))

    # ------------------------------------------------------------------ Join
    def _join(self, node) -> _Rel:
        kind = node.kind.lower()
        if kind not in ("inner", "left"):
            raise InterpreterError(f"unsupported join kind {node.kind!r}")
        left = self.run(node.left).table
        right = self.run(node.right).table
        # zero-row copy of both sides: just the column layout, for name resolution
        combined = Table(left.take([]).columns + right.take([]).columns)
        scope = Scope(combined)
        right_rows = right.to_rows()
        null_right = (None,) * len(right.columns)
        candidates = _candidate_finder(node.condition, left, right, right_rows)

        out = []
        for lrow in left.to_rows():
            matched = False
            for j in candidates(lrow):
                row = lrow + right_rows[j]
                if node.condition is None or is_true(evaluate(node.condition, row, scope)):
                    out.append(row)
                    matched = True
            if kind == "left" and not matched:
                out.append(lrow + null_right)

        fields = [(c.name, c.dtype, c.table) for c in combined.columns]
        return _Rel(Table.from_rows(fields, out))

    # ------------------------------------------------------------------ Aggregate
    def _aggregate(self, node) -> _Rel:
        rel = self.run(node.child)
        scope = rel.scope()
        aggs = [(a, alias, a.func.lower()) for a, alias in node.aggs]
        for _, _, func in aggs:
            if func not in _AGG_FUNCS:
                raise InterpreterError(f"unsupported aggregate {func!r}")

        groups: dict[tuple, list[_Acc]] = {}
        for row in rel.table.to_rows():
            key = tuple(evaluate(k, row, scope) for k in node.group_keys)
            accs = groups.get(key)
            if accs is None:
                accs = groups[key] = [_Acc(func) for _, _, func in aggs]
            for acc, (agg, _, _) in zip(accs, aggs):
                acc.add(None if agg.arg is None else evaluate(agg.arg, row, scope),
                        star=agg.arg is None)

        if not node.group_keys and not groups:
            groups[()] = [_Acc(func) for _, _, func in aggs]

        fields, expr_columns = [], {}
        for i, k in enumerate(node.group_keys):
            if node_kind(k) == "ColumnRef":
                col = scope.columns[scope.index_of(k)]
                fields.append((col.name, col.dtype, col.table))
            else:
                fields.append((render_expr(k), infer_dtype(k, scope), None))
                expr_columns[expr_key(k)] = i
        for j, (agg, alias, _) in enumerate(aggs):
            fields.append((alias, agg_result_dtype(agg, scope), None))
            expr_columns[expr_key(agg)] = len(node.group_keys) + j

        rows = [key + tuple(acc.result() for acc in accs) for key, accs in groups.items()]
        return _Rel(Table.from_rows(fields, rows), expr_columns)

    # ------------------------------------------------------------------ Sort
    def _sort(self, node) -> _Rel:
        rel = self.run(node.child)
        scope = rel.scope()
        rows = rel.table.to_rows()
        key_values = [[evaluate(expr, row, scope) for row in rows] for expr, _ in node.keys]

        order = list(range(len(rows)))
        # Stable sort by the last key first, so earlier keys take priority.
        for (_, descending), values in reversed(list(zip(node.keys, key_values))):
            present = [i for i in order if values[i] is not None]
            nulls = [i for i in order if values[i] is None]
            present.sort(key=lambda i: values[i], reverse=bool(descending))
            order = present + nulls
        return _Rel(rel.table.take(order), rel.expr_columns)

    # ------------------------------------------------------------------ Limit
    def _limit(self, node) -> _Rel:
        if node.n < 0:
            raise InterpreterError(f"LIMIT must be non-negative, got {node.n}")
        rel = self.run(node.child)
        return _Rel(rel.table.slice(0, node.n), rel.expr_columns)


_AGG_FUNCS = {"sum", "count", "avg", "min", "max"}


class _Acc:
    """Running state for one aggregate in one group."""

    def __init__(self, func: str):
        self.func = func
        self.count = 0      # non-NULL inputs (or rows, for count(*))
        self.total = None   # sum, for sum/avg
        self.best = None    # min/max

    def add(self, value, star: bool):
        if star:
            self.count += 1
            return
        if value is None:
            return
        self.count += 1
        if self.func in ("sum", "avg"):
            self.total = value if self.total is None else self.total + value
        elif self.func == "min":
            self.best = value if self.best is None or value < self.best else self.best
        elif self.func == "max":
            self.best = value if self.best is None or value > self.best else self.best

    def result(self):
        if self.func == "count":
            return self.count
        if self.func == "sum":
            return self.total
        if self.func == "avg":
            return None if self.count == 0 else self.total / self.count
        return self.best


# ---------------------------------------------------------------------- join candidates

def _candidate_finder(condition, left: Table, right: Table, right_rows: list):
    """`lrow -> right row indices worth evaluating the condition on`, in right order.

    All of them, unless the condition has equality conjuncts with one side per input
    and comparable types; then only the right rows whose keys equal the left row's.
    """
    everything = range(len(right_rows))
    keys = _equi_keys(condition, left, right)
    if not keys:
        return lambda lrow: everything
    lscope, rscope = Scope(left.take([])), Scope(right.take([]))
    index: dict[tuple, list[int]] = {}
    for j, rrow in enumerate(right_rows):
        key = tuple(evaluate(r, rrow, rscope) for _, r in keys)
        if None not in key:  # NULL = anything is never TRUE
            index.setdefault(key, []).append(j)

    def candidates(lrow):
        key = tuple(evaluate(l_expr, lrow, lscope) for l_expr, _ in keys)
        return () if None in key else index.get(key, ())
    return candidates


_NUMERIC = {"INT", "FLOAT"}


def _equi_keys(condition, left: Table, right: Table) -> list[tuple]:
    """(left expr, right expr) for each `a = b` conjunct that splits across the inputs.

    Only when both sides have the same type (or are both numeric): Python's == then
    agrees with SQL's on every pair the condition could accept. Anything else, a DATE
    compared to a STRING say, stays out of the index and is left to the condition.
    """
    if condition is None:
        return []
    lscope, rscope = Scope(left.take([])), Scope(right.take([]))

    def side(expr):
        refs = _column_refs(expr)
        in_left = bool(refs) and all(find_columns(left.columns, r.name, r.table) for r in refs)
        in_right = bool(refs) and all(find_columns(right.columns, r.name, r.table) for r in refs)
        return "L" if in_left and not in_right else "R" if in_right and not in_left else None

    keys = []
    for c in _conjuncts(condition):
        if node_kind(c) != "BinaryOp" or _norm(c.op) not in ("=", "=="):
            continue
        sides = (side(c.left), side(c.right))
        if sides not in (("L", "R"), ("R", "L")):
            continue
        l_expr, r_expr = (c.left, c.right) if sides == ("L", "R") else (c.right, c.left)
        try:
            lt, rt = infer_dtype(l_expr, lscope).name, infer_dtype(r_expr, rscope).name
        except InterpreterError:
            continue
        if lt == rt or {lt, rt} <= _NUMERIC:
            keys.append((l_expr, r_expr))
    return keys


def _conjuncts(expr) -> list:
    if node_kind(expr) == "BinaryOp" and _norm(expr.op) == "AND":
        return _conjuncts(expr.left) + _conjuncts(expr.right)
    return [expr]


def _column_refs(expr) -> list:
    kind = node_kind(expr)
    if kind == "ColumnRef":
        return [expr]
    if kind == "UnaryOp":
        return _column_refs(expr.operand)
    if kind == "BinaryOp":
        return _column_refs(expr.left) + _column_refs(expr.right)
    if kind == "AggCall":
        return [] if expr.arg is None else _column_refs(expr.arg)
    return []
