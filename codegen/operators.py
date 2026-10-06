"""Per-operator code emission. Each method emits numpy code and returns a `Rel` that
says which variables now hold each output column.

Scan, Filter, Project (Phase C3); Join, Aggregate, Sort, Limit (Phase C4).
"""
from __future__ import annotations

from codegen.emitter import Emitter
from codegen.exprgen import (
    _FILLER_SRC, _NP_FULL_DTYPE, CVar, CodegenError, ExprGen, Rel, and_ok, column_refs,
    conjuncts, full_src,
)
from runtime._compat import BinaryOp, format_plan
from runtime.expr_eval import _norm, agg_result_dtype, expr_key, node_kind, render_expr

COMPILED_KINDS = {"Scan", "Filter", "Project", "Join", "Aggregate", "Sort", "Limit"}


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

    def _empty(self, columns, why: str, expr_columns=None) -> Rel:
        """Zero-row columns with the given layout; nothing below this point runs."""
        self.em.comment(why)
        out = []
        for col in columns:
            v = self.em.fresh(col.name)
            self.em.line(f"{v} = np.empty(0, dtype={_NP_FULL_DTYPE[col.dtype.name]})")
            out.append(CVar(col.name, col.dtype, col.table, v, "True"))
        return Rel(out, "0", expr_columns)

    def _layout_of(self, node) -> Rel:
        """The output layout `node` would produce, without emitting any of its code."""
        return PlanCompiler(Emitter(), self.catalog).compile(node)

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

    # ------------------------------------------------------------------ shared: gather rows
    def _take(self, rel: Rel, rows: str, n: str, keep_computed: bool = True) -> Rel:
        """Every column of `rel` indexed by `rows` (a bool mask or an index array)."""
        out = []
        for col in rel.columns:
            v = self.em.fresh(col.name)
            if col.ok in ("True", "False"):
                self.em.line(f"{v} = {col.v}[{rows}]")
                ok = col.ok
            else:
                ok = f"{v}_ok"
                self.em.reserve(ok)
                self.em.line(f"{v}, {ok} = {col.v}[{rows}], {col.ok}[{rows}]")
            out.append(CVar(col.name, col.dtype, col.table, v, ok))
        return Rel(out, n, rel.expr_columns if keep_computed else None)

    # ------------------------------------------------------------------ Filter
    def _filter(self, node) -> Rel:
        if never_true(node.predicate):
            layout = self._layout_of(node.child)
            self._section(node)
            return self._empty(layout.columns, "predicate is constant FALSE/NULL: the child never runs",
                               layout.expr_columns)
        rel = self.compile(node.child)
        self._section(node)
        gen = ExprGen(self.em, rel)
        code = gen.gen(node.predicate)
        keep = self.em.fresh("keep")
        self.em.line(f"{keep} = {mask_src(gen, code, rel.n)}")
        n = self.em.fresh("n")
        self.em.line(f"{n} = int(np.count_nonzero({keep}))")
        return self._take(rel, keep, n)

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

    # ------------------------------------------------------------------ Join
    def _join(self, node) -> Rel:
        kind = node.kind.lower()
        if kind not in ("inner", "left"):
            raise CodegenError(f"unsupported join kind {node.kind!r}")
        if kind == "inner" and never_true(node.condition):
            layout = Rel(self._layout_of(node.left).columns + self._layout_of(node.right).columns, "0")
            self._section(node)
            return self._empty(layout.columns, "join condition is constant FALSE/NULL: "
                                               "neither input runs")

        left = self.compile(node.left)
        right = self.compile(node.right)
        self._section(node)
        li, ri = self.em.fresh("li"), self.em.fresh("ri")
        keys, residual = ([], None) if never_true(node.condition) else             self._split_condition(node.condition, left, right)

        if never_true(node.condition):        # LEFT join that can never match
            self.em.comment("condition is constant FALSE/NULL: every left row is unmatched")
            self.em.line(f"{li} = np.zeros(0, dtype=np.int64)")
            self.em.line(f"{ri} = np.zeros(0, dtype=np.int64)")
        elif keys:
            self._hash_join(keys, left, right, li, ri)
        else:
            self.em.comment("no equality keys: nested loop, i.e. every (left, right) pair")
            self.em.line(f"{li} = np.repeat(np.arange({left.n}), {right.n})")
            self.em.line(f"{ri} = np.tile(np.arange({right.n}), {left.n})")

        if residual is not None:
            self._join_residual(residual, left, right, li, ri)

        if kind == "left":
            matched = self.em.fresh("matched")
            lonely = self.em.fresh("unmatched")
            self.em.comment("LEFT JOIN: left rows with no match get one row of NULLs on the right")
            self.em.line(f"{matched} = np.zeros({left.n}, dtype=bool)")
            self.em.line(f"{matched}[{li}] = True")
            self.em.line(f"{lonely} = np.flatnonzero(~{matched})")
            self.em.line(f"{li} = np.concatenate([{li}, {lonely}])")
            self.em.line(f"{ri} = np.concatenate([{ri}, np.full(len({lonely}), -1)])")

        order = self.em.fresh("order")
        self.em.comment("emit pairs in left-row order, then right-row order (the reference order)")
        self.em.line(f"{order} = np.lexsort(({ri}, {li}))")
        self.em.line(f"{li}, {ri} = {li}[{order}], {ri}[{order}]")
        n = self.em.fresh("n")
        self.em.line(f"{n} = len({li})")

        out = self._take(left, li, n, keep_computed=False).columns
        for col in right.columns:
            v = self.em.fresh(col.name)
            ok = f"{v}_ok"
            self.em.reserve(ok)
            if kind == "inner":
                if col.ok in ("True", "False"):
                    self.em.line(f"{v} = {col.v}[{ri}]")
                    ok = col.ok
                else:
                    self.em.line(f"{v}, {ok} = {col.v}[{ri}], {col.ok}[{ri}]")
            else:
                ok_arg = "None" if col.ok == "True" else (
                    f"np.zeros({right.n}, dtype=bool)" if col.ok == "False" else col.ok)
                self.em.line(f"{v}, {ok} = take_or_null({col.v}, {ok_arg}, {ri})")
            out.append(CVar(col.name, col.dtype, col.table, v, ok))
        return Rel(out, n)

    def _split_condition(self, condition, left: Rel, right: Rel):
        """Equality conjuncts with one side per input become hash keys; the rest is residual."""
        def side(expr):
            refs = column_refs(expr)
            if not refs:
                return None
            in_left = all(left.find(r.name, r.table) for r in refs)
            in_right = all(right.find(r.name, r.table) for r in refs)
            if in_left and not in_right:
                return "L"
            if in_right and not in_left:
                return "R"
            return None

        if condition is not None and node_kind(condition) == "Literal" and condition.value is True:
            return [], None
        keys, rest = [], []
        for c in conjuncts(condition):
            if node_kind(c) == "BinaryOp" and _norm(c.op) in ("=", "=="):
                sl, sr = side(c.left), side(c.right)
                if (sl, sr) == ("L", "R"):
                    keys.append((c.left, c.right))
                    continue
                if (sl, sr) == ("R", "L"):
                    keys.append((c.right, c.left))
                    continue
            rest.append(c)
        residual = None
        for c in rest:
            residual = c if residual is None else BinaryOp(op="AND", left=residual, right=c)
        return keys, residual

    def _key_list(self, rel: Rel, exprs, prefix: str):
        """Emit Python key lists (tuples for multi-column keys) and their combined ok mask."""
        gen = ExprGen(self.em, rel)
        codes = [gen.gen(e) for e in exprs]
        keys = self.em.fresh(f"{prefix}_keys")
        ok = self.em.fresh(f"{prefix}_keys_ok")
        lists = [f"{c.v}.tolist()" for c in codes]
        self.em.line(f"{keys} = {lists[0]}" if len(lists) == 1
                     else f"{keys} = list(zip({', '.join(lists)}))")
        oks = and_ok(*[c.ok for c in codes])
        self.em.line(f"{ok} = " + (f"np.ones({rel.n}, dtype=bool)" if oks == "True" else
                                   f"np.zeros({rel.n}, dtype=bool)" if oks == "False" else oks))
        return keys, ok

    def _hash_join(self, keys, left: Rel, right: Rel, li: str, ri: str):
        k = len(keys)
        self.em.comment(f"hash join on {k} key{'s' if k > 1 else ''}: "
                        f"build a dict on the smaller input, probe it with the larger")
        lk, lok = self._key_list(left, [l for l, _ in keys], "left")
        rk, rok = self._key_list(right, [r for _, r in keys], "right")
        build_left = self.em.fresh("build_left")
        table = self.em.fresh("hash_table")
        bi, pi = self.em.fresh("build_idx"), self.em.fresh("probe_idx")
        bk, bok, pk, pok = (self.em.fresh(x) for x in ("build_keys", "build_ok", "probe_keys", "probe_ok"))
        for name in ("i", "j", "key", "present", "match"):
            self.em.reserve(name)
        self.em.line(f"{build_left} = len({lk}) <= len({rk})")
        self.em.line(f"{bk}, {bok}, {pk}, {pok} = ({lk}, {lok}, {rk}, {rok}) if {build_left} "
                     f"else ({rk}, {rok}, {lk}, {lok})")
        self.em.line(f"{table} = {{}}")
        with self.em.block(f"for i, (key, present) in enumerate(zip({bk}, {bok}.tolist())):"):
            self.em.comment("NULL keys never match")
            with self.em.block("if present:"):
                self.em.line(f"{table}.setdefault(key, []).append(i)")
        self.em.line(f"{bi}, {pi} = [], []")
        with self.em.block(f"for j, (key, present) in enumerate(zip({pk}, {pok}.tolist())):"):
            with self.em.block("if present:"):
                with self.em.block(f"for match in {table}.get(key, ()):"):
                    self.em.line(f"{bi}.append(match)")
                    self.em.line(f"{pi}.append(j)")
        self.em.line(f"{li}, {ri} = ({bi}, {pi}) if {build_left} else ({pi}, {bi})")
        self.em.line(f"{li}, {ri} = np.array({li}, dtype=np.int64), np.array({ri}, dtype=np.int64)")

    def _join_residual(self, residual, left: Rel, right: Rel, li: str, ri: str):
        """Conditions that aren't simple equality keys: evaluate them on the candidate pairs."""
        self.em.comment("remaining join condition, checked on each candidate pair")
        pairs = Rel([CVar(c.name, c.dtype, c.table) for c in left.columns + right.columns], f"len({li})")
        sources = left.columns + right.columns
        n_left = len(left.columns)

        def load(col: CVar):
            src = sources[pairs.columns.index(col)]
            idx = li if pairs.columns.index(col) < n_left else ri
            v = self.em.fresh(col.name)
            if src.ok in ("True", "False"):
                self.em.line(f"{v} = {src.v}[{idx}]")
                col.v, col.ok = v, src.ok
            else:
                self.em.reserve(f"{v}_ok")
                self.em.line(f"{v}, {v}_ok = {src.v}[{idx}], {src.ok}[{idx}]")
                col.v, col.ok = v, f"{v}_ok"

        gen = ExprGen(self.em, pairs, load)
        code = gen.gen(residual)
        keep = self.em.fresh("keep")
        self.em.line(f"{keep} = {mask_src(gen, code, pairs.n)}")
        self.em.line(f"{li}, {ri} = {li}[{keep}], {ri}[{keep}]")

    # ------------------------------------------------------------------ Aggregate
    def _aggregate(self, node) -> Rel:
        rel = self.compile(node.child)
        self._section(node)
        gen = ExprGen(self.em, rel)
        aggs = [(call, alias, call.func.lower()) for call, alias in node.aggs]
        for _, _, func in aggs:
            if func not in ("sum", "count", "avg", "min", "max"):
                raise CodegenError(f"unsupported aggregate {func!r}")

        out, expr_columns = [], {}
        if node.group_keys:
            gid, ng = self._group_ids(gen, rel, node.group_keys, out, expr_columns)
        else:
            gid, ng = None, "1"
            self.em.comment("no GROUP BY: one output row, even when the input is empty")

        for call, alias, func in aggs:
            dtype = agg_result_dtype(call, gen.scope)
            v = self.em.fresh(alias)
            ok = self._accumulate(gen, rel, call, func, dtype, v, gid, ng)
            expr_columns[expr_key(call)] = len(out)
            out.append(CVar(alias, dtype, None, v, ok))
        return Rel(out, ng, expr_columns)

    def _group_ids(self, gen, rel, group_keys, out, expr_columns):
        """Hash every row's key tuple to a dense group id (first-seen order)."""
        self.em.comment("hash aggregation: each distinct key tuple gets a group id, "
                        "in first-seen order")
        codes = [gen.gen(k) for k in group_keys]
        parts = []
        for code in codes:
            if code.ok == "True":
                parts.append(f"{code.v}.tolist()")
            else:
                parts.append(f"[v if o else None for v, o in zip({code.v}.tolist(), {code.ok}.tolist())]")
        for name in ("v", "o", "key"):
            self.em.reserve(name)
        keys = self.em.fresh("group_keys")
        groups, gid, ng, first = (self.em.fresh(x) for x in ("groups", "gid", "n_groups", "first_row"))
        self.em.line(f"{keys} = {parts[0]}" if len(parts) == 1 else f"{keys} = list(zip({', '.join(parts)}))")
        self.em.line(f"{groups} = {{}}")
        self.em.line(f"{gid} = np.fromiter(({groups}.setdefault(key, len({groups})) for key in {keys}), "
                     f"dtype=np.int64, count=len({keys}))")
        self.em.line(f"{ng} = len({groups})")
        self.em.line(f"{first} = np.zeros({ng}, dtype=np.int64)")
        self.em.line(f"{first}[{gid}[::-1]] = np.arange(len({gid}))[::-1]  # each group's first row")

        for key_expr, code in zip(group_keys, codes):
            if node_kind(key_expr) == "ColumnRef":
                col = rel.columns[gen.scope.index_of(key_expr)]
                name, dtype, table = col.name, col.dtype, col.table
            else:
                name, dtype, table = render_expr(key_expr), gen.dtype(key_expr), None
                expr_columns[expr_key(key_expr)] = len(out)
            v = self.em.fresh(name)
            src = full_src(code.v, rel.n, dtype) if code.scalar else code.v
            if code.ok in ("True", "False"):
                self.em.line(f"{v} = {src}[{first}]")
                ok = code.ok
            else:
                ok = f"{v}_ok"
                self.em.reserve(ok)
                self.em.line(f"{v}, {ok} = {src}[{first}], {code.ok}[{first}]")
            out.append(CVar(name, dtype, table, v, ok))
        return gid, ng

    def _accumulate(self, gen, rel, call, func, dtype, v, gid, ng) -> str:
        """Emit one aggregate; return its ok (NULL-mask) source."""
        label = render_expr(call)
        if func == "count" and call.arg is None:
            self.em.line(f"{v} = np.bincount({gid}, minlength={ng})  # {label}" if gid
                         else f"{v} = np.array([{rel.n}], dtype=np.int64)  # {label}")
            return "True"

        code = gen.gen(call.arg)
        x = full_src(code.v, rel.n, gen.dtype(call.arg)) if code.scalar else code.v
        self.em.line(f"# {label}: NULL inputs are skipped")
        if code.ok == "True":
            sel = None
        elif code.ok == "False":
            sel = f"np.zeros({rel.n}, dtype=bool)"
        elif gen.is_scalar(code.ok):
            sel = f"np.full({rel.n}, {code.ok}, dtype=bool)"
        else:
            sel = code.ok
        base = v.rsplit("_", 1)[0]
        count = self.em.fresh(f"{base}_count")

        if gid is None:  # ---------------- global aggregate: plain reductions, one row
            vals = self.em.fresh("vals")
            self.em.line(f"{vals} = {x}" if sel is None else f"{vals} = {x}[{sel}]")
            self.em.line(f"{count} = len({vals})")
            if func == "count":
                self.em.line(f"{v} = np.array([{count}], dtype=np.int64)")
                return "True"
            if func == "sum":
                self.em.line(f"{v} = np.array([{vals}.sum()], dtype={_NP_FULL_DTYPE[dtype.name]})")
            elif func == "avg":
                self.em.line(f"{v} = np.array([{vals}.sum() / {count} if {count} else 0.0])"
                             f"  # sum and count, divided once at the end")
            else:
                self.em.line(f"{v} = np.array([{vals}.{func}() if {count} else {_FILLER_SRC[dtype.name]}], "
                             f"dtype={_NP_FULL_DTYPE[dtype.name]})")
            ok = f"{v}_ok"
            self.em.reserve(ok)
            self.em.line(f"{ok} = np.array([{count} > 0])")
            return ok

        # -------------------------------- grouped: vectorized per-group accumulators
        g, xs = self.em.fresh("g"), self.em.fresh("xs")
        self.em.line(f"{g}, {xs} = {gid}, {x}" if sel is None
                     else f"{g}, {xs} = {gid}[{sel}], {x}[{sel}]")
        self.em.line(f"{count} = np.bincount({g}, minlength={ng})")
        if func == "count":
            self.em.line(f"{v} = {count}")
            return "True"
        if func in ("sum", "avg"):
            acc_dtype = "np.float64" if func == "avg" else _NP_FULL_DTYPE[dtype.name]
            total = v if func == "sum" else self.em.fresh(f"{base}_sum")
            self.em.line(f"{total} = np.zeros({ng}, dtype={acc_dtype})")
            self.em.line(f"np.add.at({total}, {g}, {xs})")
            if func == "avg":
                self.em.line(f"{v} = {total} / np.where({count} == 0, 1, {count})"
                             f"  # sum and count, divided once at the end")
        else:  # min / max: seed each group with one of its own values, then fold
            # groups with no input keep a typed filler, never None (NULL slots must stay sortable)
            self.em.line(f"{v} = np.full({ng}, {_FILLER_SRC[dtype.name]}, dtype={_NP_FULL_DTYPE[dtype.name]})")
            self.em.line(f"{v}[{g}] = {xs}")
            self.em.line(f"np.{'minimum' if func == 'min' else 'maximum'}.at({v}, {g}, {xs})")
        ok = f"{v}_ok"
        self.em.reserve(ok)
        self.em.line(f"{ok} = {count} > 0")
        return ok

    # ------------------------------------------------------------------ Sort
    def _sort(self, node) -> Rel:
        rel = self.compile(node.child)
        self._section(node)
        gen = ExprGen(self.em, rel)
        ranks = []
        self.em.comment("rank each key (NULLs last, DESC negates), then one stable lexsort")
        for expr, descending in node.keys:
            code = gen.gen(expr)
            dtype = gen.dtype(expr)
            values = full_src(code.v, rel.n, dtype) if code.scalar else code.v
            r = self.em.fresh("rank")
            self.em.line(f"{r} = np.unique({values}, return_inverse=True)[1].reshape(-1)")
            if descending:
                self.em.line(f"{r} = -{r}")
            if code.ok != "True":
                ok = code.ok if code.ok != "False" else f"np.zeros({rel.n}, dtype=bool)"
                self.em.line(f"{r} = np.where({ok}, {r}, {rel.n} + 1)")
            ranks.append(r)
        order = self.em.fresh("order")
        # np.lexsort sorts by its LAST key first, so pass the keys in reverse
        self.em.line(f"{order} = np.lexsort(({', '.join(reversed(ranks))},))")
        return self._take(rel, order, rel.n)

    # ------------------------------------------------------------------ Limit
    def _limit(self, node) -> Rel:
        if node.n < 0:
            raise CodegenError(f"LIMIT must be non-negative, got {node.n}")
        rel = self.compile(node.child)
        self._section(node)
        n = self.em.fresh("n")
        self.em.line(f"{n} = min({node.n}, {rel.n})")
        return self._take(rel, f":{n}", n)

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
