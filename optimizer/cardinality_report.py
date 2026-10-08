"""Validation harness: estimated vs actual row counts, for every node of every query.

``compare`` pairs the estimator's row count for each node with the count
that a real execution of that node's subtree returns. The harness takes the
row counter as a function, so it runs with any executor: the tests'
reference evaluator now, and Person C's ``runtime.interpret`` later.

Errors are reported as the q-error (Moerkotte, Neumann and Steidl, 2009):
max(estimate / actual, actual / estimate), with both counts raised to at
least one row first. It is symmetric (2x too high and 2x too low both score
2), and a q-error of 1 means exact. Plan choice depends on ratios between
estimates, and the q-error bounds how far those ratios can be off.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from optimizer.stats import annotate


@dataclass(frozen=True)
class NodeEstimate:
    query: str
    path: tuple[int, ...]  # child indexes from the root
    label: str             # the node's line in the plan printout
    kind: str              # the node's type: Scan, Join, ...
    estimated: float
    actual: int

    @property
    def q_error(self) -> float:
        e, a = max(self.estimated, 1.0), max(float(self.actual), 1.0)
        return max(e / a, a / e)

    @property
    def direction(self) -> str:
        e, a = max(self.estimated, 1.0), max(float(self.actual), 1.0)
        return "over" if e > a * 1.1 else "under" if a > e * 1.1 else "close"


def compare(query: str, plan: Any, catalog: Any, count_rows: Callable[[Any], int],
            formatter: Callable[[Any], str] | None = None) -> list[NodeEstimate]:
    """Return one ``NodeEstimate`` per node of ``plan``, in pre-order."""
    from optimizer.trace import default_formatter

    nodes = annotate(plan, catalog)
    labels = [line.strip() for line in (formatter or default_formatter())(plan).splitlines()]
    if len(labels) != len(nodes):  # a formatter that doesn't print one line per node
        labels = [type(node).__name__ for _, node, _ in nodes]
    return [
        NodeEstimate(query, path, label, type(node).__name__, est.rows, count_rows(node))
        for label, (path, node, est) in zip(labels, nodes)
    ]


@dataclass(frozen=True)
class Summary:
    group: str
    nodes: int
    median: float
    p90: float
    worst: float
    within_2x: float  # fraction of nodes with q-error <= 2
    over: int
    under: int


def summarize(rows: Iterable[NodeEstimate], by_kind: bool = True) -> list[Summary]:
    """Q-error statistics for all nodes, then (if ``by_kind``) for each node type."""
    rows = list(rows)
    groups: dict[str, list[NodeEstimate]] = {"all nodes": rows}
    if by_kind:
        for r in rows:
            groups.setdefault(r.kind, []).append(r)
    out = []
    for name, members in groups.items():
        if not members:
            continue
        q = sorted(r.q_error for r in members)
        out.append(Summary(
            group=name,
            nodes=len(q),
            median=statistics.median(q),
            p90=q[min(len(q) - 1, int(0.9 * len(q)))],
            worst=q[-1],
            within_2x=sum(x <= 2.0 for x in q) / len(q),
            over=sum(r.direction == "over" for r in members),
            under=sum(r.direction == "under" for r in members),
        ))
    return out


def summary_table(rows: Iterable[NodeEstimate]) -> str:
    lines = [
        "| node type | nodes | median q-error | 90th percentile | worst | within 2x | over | under |",
        "|---|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for s in summarize(rows):
        lines.append(f"| {s.group} | {s.nodes} | {s.median:.2f} | {s.p90:.2f} | {s.worst:.2f} "
                     f"| {s.within_2x:.0%} | {s.over} | {s.under} |")
    return "\n".join(lines)


def detail_table(rows: Iterable[NodeEstimate], with_query: bool = True, indent: bool = True) -> str:
    """One row per node: estimated and actual rows and q-error. ``indent`` draws the plan tree (keep rows in pre-order)."""
    head = "| query | node | estimated | actual | q-error |" if with_query else "| node | estimated | actual | q-error |"
    lines = [head, "|---|---|--:|--:|--:|" if with_query else "|---|--:|--:|--:|"]
    last = None
    for r in rows:
        node = "&nbsp;&nbsp;" * len(r.path) * indent + f"`{cell(r.label)}`"
        cells = [node, f"{r.estimated:.1f}", str(r.actual), f"{r.q_error:.2f}"]
        if with_query:
            cells.insert(0, r.query if r.query != last else "")
            last = r.query
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cell(text: str) -> str:
    """Escape text for a markdown table cell."""
    return text.replace("|", "\\|")
