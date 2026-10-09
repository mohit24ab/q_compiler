"""Ablation study: what each pass contributes, measured.

``run_ablation`` executes every query under several configurations of the
pipeline: all passes, each pass switched off in turn, and no passes at all
(the plan exactly as written). For every (query, configuration) it records:

  runtime_ms      fastest of ``repeat`` executions of the plan, excluding
                  code generation and compilation
  rows_scanned    rows read from base tables: the row count of every Scan
                  that runs. A subtree codegen knows to be empty (a FALSE
                  filter, pushed predicate or inner join condition) runs no
                  Scans.
  values_read     row count times the number of columns each running Scan
                  reads (its output columns plus those its pushed predicate
                  reads): the I/O a columnar engine does
  estimated_cost  the B5 cost model's estimate, in microseconds
  same_result     whether the result equals the unoptimized plan's

The executor is a parameter, so the same study runs on the reference
evaluator, on Person C's interpreter, or on generated code: ``prepare``
takes a plan and returns a function that executes it once and returns the
result rows.

``pass_order_study`` runs the suite through every ordering of the default
passes and reports the final plans' total estimated cost, how many pass
applications the fixed-point loop needed, and how many queries end in a
different plan from the default order.
"""

from __future__ import annotations

import csv
import io
import itertools
import time
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Iterable

from ir.nodes import Scan

from optimizer.columns import column_refs, table_schema
from optimizer.cost import CostModel, _short_circuits
from optimizer.manager import OptimizerManager, OptimizerWarning


@dataclass(frozen=True)
class Config:
    name: str
    passes: tuple[str, ...]  # enabled pass names, in pipeline order


def default_configs(pass_names: Iterable[str]) -> list[Config]:
    """All passes; each pass off in turn; no passes."""
    names = tuple(pass_names)
    configs = [Config("all passes", names)]
    configs += [Config(f"without {n}", tuple(p for p in names if p != n)) for n in names]
    configs.append(Config("no passes", ()))
    return configs


def optimize_with(plan: Any, catalog: Any, pass_names: Iterable[str], passes: list | None = None) -> Any:
    """Run the default pipeline restricted to ``pass_names``, in default order."""
    from optimizer import default_passes

    wanted = set(pass_names)
    chosen = [p for p in (passes if passes is not None else default_passes()) if p.name in wanted]
    if not chosen:
        return plan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizerWarning)
        final, _ = OptimizerManager(chosen).optimize(plan, catalog)
    return final


def scan_metrics(plan: Any, catalog: Any) -> tuple[int, int]:
    """``(rows_scanned, values_read)`` over the Scans that run."""
    rows = values = 0
    if _short_circuits(plan):
        return 0, 0
    if isinstance(plan, Scan):
        n = catalog.row_count(plan.table)
        if plan.columns is not None:
            out = set(plan.columns)
        else:
            out = {name for name, _ in table_schema(plan, catalog) or []}
        read = out | {name for _, name in column_refs(plan.pushed_predicate)}
        return n, n * len(read)
    for child in plan.children:
        r, v = scan_metrics(child, catalog)
        rows, values = rows + r, values + v
    return rows, values


@dataclass(frozen=True)
class Measurement:
    query: str
    config: str
    runtime_ms: float
    result_rows: int
    rows_scanned: int
    values_read: int
    estimated_cost_us: float
    same_result: bool
    workload: str = ""  # which query set the query belongs to (e.g. "golden", "suite")


def run_ablation(
    queries: list[tuple[str, Any]],
    catalog: Any,
    prepare: Callable[[Any], Callable[[], list]],
    repeat: int = 3,
    configs: list[Config] | None = None,
    same: Callable[[list, list], bool] | None = None,
    workload: str = "",
) -> list[Measurement]:
    """Execute every query under every configuration. See the module docstring."""
    from optimizer import default_passes

    configs = configs if configs is not None else default_configs(p.name for p in default_passes())
    same = same or _same_rows
    model = CostModel(catalog)
    out = []
    for name, plan in queries:
        reference = prepare(plan)()  # the plan as written
        for config in configs:
            optimized = optimize_with(plan, catalog, config.passes)
            execute = prepare(optimized)
            best, result = float("inf"), None
            for _ in range(repeat):
                start = time.perf_counter()
                result = execute()
                best = min(best, time.perf_counter() - start)
            rows_scanned, values_read = scan_metrics(optimized, catalog)
            out.append(Measurement(
                query=name, config=config.name, runtime_ms=best * 1000, result_rows=len(result),
                rows_scanned=rows_scanned, values_read=values_read,
                estimated_cost_us=model.cost(optimized).total / 1000,
                same_result=same(result, reference), workload=workload,
            ))
    return out


def _same_rows(a: list, b: list) -> bool:
    """Equal as multisets of rows, with floats compared to 9 decimal places."""
    def key(row):
        return tuple(round(v, 9) if isinstance(v, float) else v for v in row)
    return sorted(map(key, a), key=repr) == sorted(map(key, b), key=repr)


def to_csv(measurements: list[Measurement]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([f.name for f in fields(Measurement)])
    for m in measurements:
        writer.writerow([round(v, 4) if isinstance(v, float) else v for v in asdict(m).values()])
    return buf.getvalue()


def summary_table(measurements: list[Measurement]) -> str:
    """Totals per configuration, and each relative to all passes."""
    configs = list(dict.fromkeys(m.config for m in measurements))
    total = {c: {"runtime": 0.0, "scanned": 0, "values": 0, "cost": 0.0, "wrong": 0} for c in configs}
    for m in measurements:
        t = total[m.config]
        t["runtime"] += m.runtime_ms
        t["scanned"] += m.rows_scanned
        t["values"] += m.values_read
        t["cost"] += m.estimated_cost_us
        t["wrong"] += not m.same_result
    base = total[configs[0]]
    lines = [
        "| configuration | runtime (ms) | vs all passes | rows scanned | values read | vs all passes "
        "| estimated cost (ms) | wrong results |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for c in configs:
        t = total[c]
        lines.append(
            f"| {c} | {t['runtime']:.1f} | {_ratio(t['runtime'], base['runtime'])} | {t['scanned']:,} "
            f"| {t['values']:,} | {_ratio(t['values'], base['values'])} | {t['cost'] / 1000:.1f} | {t['wrong']} |"
        )
    return "\n".join(lines)


def per_query_table(measurements: list[Measurement], config_a: str, config_b: str, top: int = 10) -> str:
    """The queries whose runtime changes most between two configurations."""
    by = {(m.query, m.config): m for m in measurements}
    rows = []
    for q in dict.fromkeys(m.query for m in measurements):
        a, b = by.get((q, config_a)), by.get((q, config_b))
        if a and b and a.runtime_ms > 0:
            rows.append((b.runtime_ms / a.runtime_ms, q, a, b))
    rows.sort(key=lambda r: -r[0])
    lines = [f"| query | {config_a} (ms) | {config_b} (ms) | slowdown |", "|---|--:|--:|--:|"]
    for ratio, q, a, b in rows[:top]:
        lines.append(f"| {q} | {a.runtime_ms:.2f} | {b.runtime_ms:.2f} | {ratio:.1f}x |")
    return "\n".join(lines)


def rank_correlation(xs: list[float], ys: list[float]) -> float:
    """Spearman's rank correlation (ties get their average rank)."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0


def cost_model_check(measurements: list[Measurement]) -> str:
    """How well the cost model's estimates rank the measured runtimes."""
    by_query: dict[str, list[Measurement]] = {}
    for m in measurements:
        by_query.setdefault(m.query, []).append(m)
    overall = rank_correlation([m.estimated_cost_us for m in measurements], [m.runtime_ms for m in measurements])
    # Within one query: does the cheaper configuration by estimate also run faster?
    agree = total = 0
    for ms in by_query.values():
        for a, b in itertools.combinations(ms, 2):
            if abs(a.runtime_ms - b.runtime_ms) < 0.1 * max(a.runtime_ms, b.runtime_ms):
                continue  # within noise: not a meaningful comparison
            total += 1
            agree += (a.estimated_cost_us < b.estimated_cost_us) == (a.runtime_ms < b.runtime_ms)
    return (
        f"Rank correlation between estimated cost and measured runtime over all {len(measurements)} runs: "
        f"Spearman's ρ = {overall:.2f}. Within each query, for pairs of configurations whose runtimes "
        f"differ by more than 10%, the cost model picks the faster one in {agree} of {total} pairs "
        f"({agree / total:.0%})." if total else "Not enough distinct runtimes to compare."
    )


def _ratio(x: float, base: float) -> str:
    return f"{x / base:.2f}x" if base else "-"


# --------------------------------------------------------------------------
# Pass ordering
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OrderResult:
    order: tuple[str, ...]
    total_cost_us: float
    applications: int      # pass applications until the fixed point, summed over queries
    iterations: int        # fixed-point iterations, summed over queries
    differs: int           # queries whose final plan differs from the default order's
    unstable: int          # queries that hit oscillation or the iteration cap


def pass_order_study(queries: list[tuple[str, Any]], catalog: Any) -> list[OrderResult]:
    from optimizer import default_passes

    passes = default_passes()
    default_order = tuple(p.name for p in passes)
    model = CostModel(catalog)
    finals: dict[tuple[str, ...], list[Any]] = {}
    out = []
    for order in itertools.permutations(passes):
        names = tuple(p.name for p in order)
        cost = 0.0
        applications = iterations = unstable = 0
        plans = []
        for _, plan in queries:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", OptimizerWarning)
                final, traces = OptimizerManager(list(order)).optimize(plan, catalog)
            unstable += any(issubclass(w.category, OptimizerWarning) for w in caught)
            applications += len(traces)
            iterations += traces[-1].iteration if traces else 0
            cost += model.cost(final).total / 1000
            plans.append(final)
        finals[names] = plans
        out.append(OrderResult(names, cost, applications, iterations, 0, unstable))
    reference = finals[default_order]
    return [
        OrderResult(r.order, r.total_cost_us, r.applications, r.iterations,
                    sum(a != b for a, b in zip(finals[r.order], reference)), r.unstable)
        for r in out
    ]


def order_table(results: list[OrderResult]) -> str:
    from optimizer import default_passes

    default_order = tuple(p.name for p in default_passes())
    short = {"constant_folding": "fold", "predicate_pushdown": "push", "join_reordering": "reorder",
             "column_pruning": "prune"}
    lines = [
        "| order | total estimated cost (ms) | pass applications | iterations | plans differing from default "
        "| oscillation or cap |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for r in sorted(results, key=lambda r: (round(r.total_cost_us, 3), r.applications)):
        name = " → ".join(short.get(n, n) for n in r.order)
        if r.order == default_order:
            name = f"**{name} (default)**"
        lines.append(f"| {name} | {r.total_cost_us / 1000:.2f} | {r.applications} | {r.iterations} "
                     f"| {r.differs} | {r.unstable} |")
    return "\n".join(lines)
