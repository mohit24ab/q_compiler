"""qcompiler, end to end: one SQL query through every stage of the compiler.

    python demo.py "SELECT nation, COUNT(*) AS n FROM customer GROUP BY nation"
    python demo.py --query q18                 # one of the 20 golden queries
    python demo.py --query q10 --scale bench   # on the 100k-row dataset
    python demo.py --list

Prints, in order:
  1. the SQL
  2. the bound plan                  (frontend.binder.parse_and_bind, Person A)
  3. the optimized plan              (optimizer.optimize, Person B)
  4. what each optimizer pass did    (Person B's PassTrace diffs)
  5. the generated Python source     (codegen.generate, Person C)
  6. the result rows, checked against the reference interpreter
  7. timing: the naive interpreter vs the optimizer vs the compiler

Colour is used when stdout is a terminal (set NO_COLOR to turn it off), and box-drawing
characters when it can encode them; otherwise plain ASCII.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bench import runner  # noqa: E402
from bench.aliases import alias_map, resolve_aliases  # noqa: E402
from codegen import generate  # noqa: E402
from codegen.runner import compile_module  # noqa: E402
from runtime import compare_tables, interpret  # noqa: E402
from runtime._compat import format_plan  # noqa: E402

SAMPLES = {  # bench/samples/: one query per shape the compiler handles
    "q01": "scan_filter_sort_limit",
    "q05": "kleene_boolean_predicate",
    "q10": "three_way_join_topk",
    "q14": "group_by_having",
    "q20": "join_group_by_having_order_by",
}


# ---------------------------------------------------------------------- styling

class Style:
    def __init__(self, color: bool, unicode: bool):
        self.color, self.unicode = color, unicode

    def _c(self, code, text):
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def bold(self, t): return self._c("1", t)
    def dim(self, t): return self._c("2", t)
    def green(self, t): return self._c("32", t)
    def red(self, t): return self._c("31", t)
    def cyan(self, t): return self._c("36", t)
    def yellow(self, t): return self._c("33", t)

    @property
    def rule(self): return "─" if self.unicode else "-"
    @property
    def bar(self): return "█" if self.unicode else "#"
    @property
    def ok(self): return "✓" if self.unicode else "OK"
    @property
    def bad(self): return "✗" if self.unicode else "X"
    @property
    def arrow(self): return "→" if self.unicode else "->"


def make_style(force_ascii=False, force_color=None) -> Style:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    unicode = not force_ascii and encoding.replace("-", "").startswith("utf")
    color = force_color if force_color is not None else (
        sys.stdout.isatty() and "NO_COLOR" not in os.environ)
    if color and os.name == "nt":
        os.system("")  # turns on ANSI escape handling in the Windows console
    return Style(color, unicode)


class Printer:
    WIDTH = 92

    def __init__(self, style: Style, out=None):
        self.s, self.out = style, out or sys.stdout
        self.n = 0

    def __call__(self, text=""):
        print(text, file=self.out)

    def section(self, title, note=""):
        self.n += 1
        label = f" {self.n}. {title} "
        tail = f" {note} " if note else ""
        fill = self.s.rule * max(4, self.WIDTH - len(label) - len(tail) - 4)
        self()
        self(self.s.bold(self.s.cyan(f"{self.s.rule * 2}{label}{fill}{tail}{self.s.rule * 2}")))

    def block(self, text, indent=2, paint=None):
        for line in text.splitlines():
            self(" " * indent + (paint(line) if paint else line))


def paint_diff(s: Style):
    def paint(line):
        if line.startswith("+") and not line.startswith("+++"):
            return s.green(line)
        if line.startswith("-") and not line.startswith("---"):
            return s.red(line)
        if line.startswith("@@") or line.startswith("==="):
            return s.cyan(line)
        return line
    return paint


def paint_source(s: Style):
    def paint(line):
        num, _, code = line.partition(" | ")
        stripped = code.lstrip()
        if stripped.startswith("#"):
            code = s.dim(code)
        elif stripped.startswith(("def ", "return ", "import ", "from ")):
            code = s.bold(code)
        return f"{s.dim(num)} | {code}"
    return paint


# ---------------------------------------------------------------------- the stages

def traces_text(traces) -> str | None:
    try:
        from optimizer import render_traces
    except ImportError:
        return None
    return render_traces(traces)


def timing_table(p: Printer, rows):
    """rows: [(label, ms)], the first one is the baseline."""
    s = p.s
    base = rows[0][1]
    longest = max(ms for _, ms in rows)
    for label, ms in rows:
        width = max(1, round(36 * ms / longest)) if longest else 1
        speedup = base / ms if ms else float("inf")
        bar = s.bar * width
        bar = s.green(bar) if speedup > 1.5 else s.yellow(bar) if speedup > 1.05 else s.dim(bar)
        p(f"  {label:<26} {ms:>10.2f} ms  {bar:<36}  {speedup:>7.1f}x")


def run_demo(sql: str, scale: str = "tiny", runs: int = 3, max_rows: int = 15,
             p: Printer | None = None) -> bool:
    p = p or Printer(make_style())
    s = p.s
    catalog, tables = runner.load_data(scale)
    optimize, opt_label = runner.find_optimizer()

    p.section("SQL", f"scale={scale}")
    p.block(sql.strip(), paint=s.bold)

    from frontend.binder import parse_and_bind
    bound = parse_and_bind(sql, catalog)
    aliases = alias_map(bound, catalog)
    plan = resolve_aliases(bound, catalog)
    p.section("Bound plan", "frontend.binder.parse_and_bind")
    p.block(format_plan(plan))
    if aliases:
        resolved = ", ".join(f"{a} {s.arrow} {t}" for a, t in aliases.items())
        p(s.dim(f"  (alias qualifiers resolved to table names: {resolved}; see bench/aliases.py)"))

    optimized, traces = optimize(plan, catalog)
    p.section("Optimized plan", opt_label)
    p.block(format_plan(optimized))
    if optimized == plan:
        p(s.dim("  (unchanged)"))

    p.section("Optimizer, pass by pass")
    rendered = traces_text(traces) if traces else None
    if rendered is None:
        p(s.dim("  optimizer.render_traces is not available here: no pass trace to show."))
    else:
        p.block(rendered, paint=paint_diff(s))

    t0 = time.perf_counter()
    source = generate(optimized, catalog, mode="compiled")
    run = compile_module(source)["run"]
    codegen_ms = (time.perf_counter() - t0) * 1000
    from codegen.runner import number_source
    lines = source.count("\n")
    p.section("Generated code", f"{lines} lines, generated + compiled in {codegen_ms:.1f} ms")
    p.block(number_source(source), indent=1, paint=paint_source(s))

    result = run(tables)
    reference = interpret(plan, tables)
    ordered = runner.is_ordered(sql)
    ok, why = compare_tables(reference, result, ordered=ordered)
    p.section("Result", f"{result.num_rows} rows")
    p.block(result.format(max_rows=max_rows))
    verdict = (s.green(f"{s.ok} identical to the reference interpreter on the unoptimized plan")
               if ok else s.red(f"{s.bad} DIFFERENT from the reference interpreter: {why}"))
    p(f"\n  {verdict}" + s.dim(" (row order compared)" if ordered else " (as a multiset)"))

    p.section("Timing", f"median of {runs} round{'s' if runs != 1 else ''}, "
                        f"configurations taking turns (bench.runner)")
    unopt_run = compile_module(generate(plan, catalog, mode="compiled"))["run"]
    configs = {"naive: interpreted": lambda: interpret(plan, tables),
               "interpreted + optimizer": lambda: interpret(optimized, tables),
               "compiled": lambda: unopt_run(tables),
               "compiled + optimizer": lambda: run(tables)}
    medians, _, _ = runner.time_interleaved(configs, runs, warmup=1)
    timing_table(p, list(medians.items()))
    p(s.dim(f"  code generation happens once per query: {codegen_ms:.1f} ms"))
    if scale == "tiny":
        p(s.dim("  (tiny data: these are fractions of a millisecond; --scale bench shows the real gap)"))
    p()
    return ok


def write_samples(directory: Path, scale: str = "tiny") -> list[Path]:
    """Generated source for the SAMPLES queries, as committed in bench/samples/."""
    catalog, _ = runner.load_data(scale)
    optimize, label = runner.find_optimizer()
    queries = runner.load_queries(list(SAMPLES))
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, slug in SAMPLES.items():
        sql = queries[name]
        plan, _ = optimize(runner.bind(sql, catalog), catalog)
        source = generate(plan, catalog, mode="compiled")
        sql_comment = "\n".join(f"#   {line}" for line in sql.splitlines())
        head = (f"# {name}: generated by `python demo.py --samples {directory.as_posix()}`\n"
                f"# optimizer: {label}\n#\n{sql_comment}\n\n")
        path = directory / f"{name}_{slug}.py"
        path.write_text(head + source, encoding="utf-8")
        written.append(path)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One SQL query through every stage of qcompiler.")
    ap.add_argument("sql", nargs="?", help="a SQL query (quote it)")
    ap.add_argument("--query", help="a golden query by name, e.g. q18")
    ap.add_argument("--list", action="store_true", help="list the golden queries")
    ap.add_argument("--scale", default="tiny", choices=["tiny", "bench"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rows", type=int, default=15, help="result rows to print")
    ap.add_argument("--ascii", action="store_true", help="no box-drawing characters")
    ap.add_argument("--color", choices=["auto", "always", "never"], default="auto")
    ap.add_argument("--samples", metavar="DIR", help="write bench/samples-style sources to DIR")
    args = ap.parse_args(argv)

    if args.samples:
        for path in write_samples(Path(args.samples)):
            print(path)
        return 0
    queries = runner.load_queries()
    if args.list:
        for name, sql in queries.items():
            print(f"{name}  {' '.join(sql.split())[:100]}")
        return 0
    if args.query:
        if args.query not in queries:
            ap.error(f"unknown query {args.query!r}; try --list")
        sql = queries[args.query]
    elif args.sql:
        sql = args.sql
    else:
        ap.error("give a SQL string or --query NAME (or --list)")
    color = {"auto": None, "always": True, "never": False}[args.color]
    printer = Printer(make_style(force_ascii=args.ascii, force_color=color))
    return 0 if run_demo(sql, args.scale, args.runs, args.rows, printer) else 1


if __name__ == "__main__":
    raise SystemExit(main())
