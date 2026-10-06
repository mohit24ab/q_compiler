"""Per-operator code emission. Each method emits numpy code and returns a `Rel` that
says which variables now hold each output column.

Implemented (Phase C3): Scan, Filter, Project.
"""
from __future__ import annotations

from codegen.emitter import Emitter
from codegen.exprgen import _NP_FULL_DTYPE, CVar, CodegenError, ExprGen, Rel, full_src
from runtime._compat import format_plan
from runtime.expr_eval import node_kind

COMPILED_KINDS = {"Scan", "Filter", "Project"}


def compilable(plan) -> bool:
    return node_kind(plan) in COMPILED_KINDS and all(compilable(c) for c in plan.children)


def never_true(predicate) -> bool:
    """A literal FALSE or NULL predicate: no row can ever pass it."""
    return (predicate is not None and node_kind(predicate) == "Literal"
            and (predicate.value is None or predicate.value is False))


def mask_src(gen, code, n: str) -> str:
    """Source for a boolean row mask: TRUE rows only (FALSE and NULL are dropped)."""
    if code.ok == "False":
        return f"np.zeros({n}, dtype=bool)"
    m = code.v if code.ok == "True" else f"({code.v} & {code.ok})"
    return f"np.full({n}, {m}, dtype=bool)" if gen.is_scalar(m) else m


class PlanCompiler:
    def __init__(self, em, catalog):
        self.em = em
        self.catalog = catalog

    def compile(self, node) -> Rel:
        kind = node_kind(node)
        method = getattr(self, f"_{kind.lower()}", None)
        if method is None or kind not in COMPILED_KINDS:
            raise CodegenError(f"{kind} is not compiled yet")
        return method(node)

    def _empty(self, columns, why: str) -> Rel:
        """Zero-row columns with the given layout; nothing below this point runs."""
        self.em.comment(why)
        out = []
        for col in columns:
            v = self.em.fresh(col.name)
            self.em.line(f"{v} = np.empty(0, dtype={_NP_FULL_DTYPE[col.dtype.name]})")
            out.append(CVar(col.name, col.dtype, col.table, v, "True"))
        return Rel(out, "0")

    def _layout_of(self, node) -> list[CVar]:
        """The output columns `node` would produce, without emitting any of its code."""
        return PlanCompiler(Emitter(), self.catalog).compile(node).columns

    def _section(self, node):
        self.em.blank()
        self.em.comment(format_plan(node).splitlines()[0].strip())

    def _bind_ok(self, gen, v: str, ok_src: str, n: str) -> str:
        """Give a non-trivial ok expression its own `<v>_ok` variable (always an array)."""
        if ok_src in ("True", "False"):
            return ok_src
        if gen.is_scalar(ok_src):  # e.g. TRUE OR <x>: NULL-ness doesn't depend on the row
            ok_src = f"np.full({n}, {ok_src}, dtype=bool)"
        elif ok_src.isidentifier():
            return ok_src
        name = f"{v}_ok"
        self.em.reserve(name)
        self.em.line(f"{name} = {ok_src}")
        return name

    # ------------------------------------------------------------------ Scan
    def _scan(self, node) -> Rel:
        schema = self._table_schema(node)
        self._section(node)
        src = self.em.fresh(node.table)
        full = Rel([CVar(name, dtype, node.table) for name, dtype in schema], n=f"{src}.num_rows")

        def load(col: CVar, rows: str | None = None):
            v = self.em.fresh(col.name)
            ok = f"{v}_ok"
            self.em.reserve(ok)
            extra = f", rows={rows}" if rows else ""
            self.em.line(f"{v}, {ok} = read_column({src}, {col.name!r}{extra})")
            col.v, col.ok = v, ok

        by_name = {c.name: c for c in full.columns}
        wanted = node.columns if node.columns is not None else [c.name for c in full.columns]
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise CodegenError(f"Scan[{node.table}] asks for unknown columns {missing}")

        if never_true(node.pushed_predicate):
            return self._empty([by_name[w] for w in wanted],
                               "pushed predicate is constant FALSE/NULL: the table is never read")
        self.em.line(f"{src} = as_table(tables[{node.table!r}])")

        if node.pushed_predicate is None:
            out = []
            for name in wanted:
                col = by_name[name]
                if col.v is None:
                    load(col)
                out.append(col)
            return Rel(out, full.n)

        # Pushed predicate: read just the columns it needs, build the mask, then read
        # the remaining output columns already filtered.
        gen = ExprGen(self.em, full, load)
        code = gen.gen(node.pushed_predicate)
        keep = self.em.fresh("pushed")
        self.em.line(f"{keep} = {mask_src(gen, code, full.n)}")
        out = []
        for name in wanted:
            col = by_name[name]
            if col.v is None:
                kept = CVar(col.name, col.dtype, col.table)
                load(kept, rows=keep)
            else:  # already read for the predicate: filter it
                v = self.em.fresh(col.name)
                self.em.reserve(f"{v}_ok")
                self.em.line(f"{v}, {v}_ok = {col.v}[{keep}], {col.ok}[{keep}]")
                kept = CVar(col.name, col.dtype, col.table, v, f"{v}_ok")
            out.append(kept)
        n = self.em.fresh("n")
        self.em.line(f"{n} = int(np.count_nonzero({keep}))")
        return Rel(out, n)

    def _table_schema(self, node):
        if getattr(node, "table_schema", None) is not None:
            return list(node.table_schema)
        if self.catalog is not None:
            return list(self.catalog.schema(node.table))
        raise CodegenError(f"no schema for table {node.table!r}: pass a catalog to generate()")

    # ------------------------------------------------------------------ Filter
    def _filter(self, node) -> Rel:
        if never_true(node.predicate):
            columns = self._layout_of(node.child)
            self._section(node)
            return self._empty(columns, "predicate is constant FALSE/NULL: the child never runs")
        rel = self.compile(node.child)
        self._section(node)
        gen = ExprGen(self.em, rel)
        code = gen.gen(node.predicate)
        keep = self.em.fresh("keep")
        self.em.line(f"{keep} = {mask_src(gen, code, rel.n)}")
        out = []
        for col in rel.columns:
            v = self.em.fresh(col.name)
            if col.ok in ("True", "False"):
                self.em.line(f"{v} = {col.v}[{keep}]")
                ok = col.ok
            else:
                ok = f"{v}_ok"
                self.em.reserve(ok)
                self.em.line(f"{v}, {ok} = {col.v}[{keep}], {col.ok}[{keep}]")
            out.append(CVar(col.name, col.dtype, col.table, v, ok))
        n = self.em.fresh("n")
        self.em.line(f"{n} = int(np.count_nonzero({keep}))")
        return Rel(out, n)

    # ------------------------------------------------------------------ Project
    def _project(self, node) -> Rel:
        rel = self.compile(node.child)
        self._section(node)
        gen = ExprGen(self.em, rel)
        out = []
        for expr, alias in node.exprs:
            dtype = gen.dtype(expr)
            if node_kind(expr) == "ColumnRef":
                # A plain column costs nothing: reuse its arrays under the new name.
                col = rel.columns[gen.scope.index_of(expr)]
                qualifier = col.table if expr.name == alias else None
                out.append(CVar(alias, col.dtype, qualifier, col.v, col.ok))
                continue
            code = gen.gen(expr)
            v = self.em.fresh(alias)
            self.em.line(f"{v} = {full_src(code.v, rel.n, dtype) if code.scalar else code.v}")
            out.append(CVar(alias, dtype, None, v, self._bind_ok(gen, v, code.ok, rel.n)))
        return Rel(out, rel.n)

    # ------------------------------------------------------------------ result
    def finish(self, rel: Rel) -> None:
        self.em.blank()
        self.em.line("return build_table([")
        with self.em.indent():
            for c in rel.columns:
                if c.ok == "True":
                    ok = "None"
                elif c.ok == "False":
                    ok = f"np.zeros({rel.n}, dtype=bool)"
                else:
                    ok = c.ok
                self.em.line(f"({c.name!r}, DType.{c.dtype.name}, {c.table!r}, {c.v}, {ok}),")
        self.em.line("])")
