"""The validation harness (optimizer/cardinality_report.py) and the report it produces."""

import re
from pathlib import Path

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
from opt_reference_eval import evaluate
from optimizer.cardinality_report import NodeEstimate, cell, compare, detail_table, summarize, summary_table
from optimizer.trace import default_formatter


def node(estimated, actual, kind="Scan", query="q", path=()):
    return NodeEstimate(query, path, kind, kind, estimated, actual)


@pytest.mark.parametrize("estimated, actual, expected", [
    (10, 10, 1.0),
    (20, 10, 2.0),
    (10, 20, 2.0),   # symmetric
    (0.2, 0, 1.0),   # both below one row: exact
    (0, 3, 3.0),     # an estimate of nothing, against 3 rows
    (0.5, 4, 4.0),
])
def test_q_error(estimated, actual, expected):
    assert node(estimated, actual).q_error == pytest.approx(expected)


def test_direction():
    assert node(20, 10).direction == "over"
    assert node(10, 20).direction == "under"
    assert node(10.5, 10).direction == "close"


def test_compare_pairs_every_node_with_its_actual_count():
    query = next(q for q in S.QUERIES if q.name == "filter_over_join_unqualified")
    plan = query.plan
    rows = compare(query.name, plan, S.CATALOG, lambda n: len(evaluate(n, S.TABLES).rows))
    lines = default_formatter()(plan).splitlines()
    assert [r.label for r in rows] == [line.strip() for line in lines]
    assert [len(r.path) for r in rows] == [(len(line) - len(line.lstrip())) // 2 for line in lines]
    assert rows[0].path == () and rows[0].actual == len(evaluate(plan, S.TABLES).rows)
    assert {r.kind for r in rows} >= {"Join", "Scan"}
    assert all(r.query == query.name for r in rows)


def test_summarize():
    rows = [node(10, 10), node(30, 10), node(10, 40, kind="Join")]
    everything, scans, joins = summarize(rows)
    assert (everything.group, everything.nodes, everything.worst) == ("all nodes", 3, 4.0)
    assert everything.median == 3.0
    assert everything.within_2x == pytest.approx(1 / 3)
    assert (everything.over, everything.under) == (1, 1)
    assert (scans.group, scans.nodes, joins.group, joins.nodes) == ("Scan", 2, "Join", 1)
    assert len(summarize(rows, by_kind=False)) == 1


def test_tables_are_valid_markdown():
    rows = [node(10, 10, path=()), node(5, 4, path=(0,)), node(1, 1, query="other", path=())]
    detail = detail_table(rows).splitlines()
    assert all(line.count("|") == detail[0].count("|") for line in detail)
    assert "&nbsp;&nbsp;`Scan`" in detail[3]
    assert detail[3].startswith("|  |")  # the query name is printed once
    assert "&nbsp;" not in detail_table(rows, indent=False)
    summary = summary_table(rows).splitlines()
    assert all(line.count("|") == summary[0].count("|") for line in summary)
    assert cell("a | b") == "a \\| b"


def test_committed_report_is_current():
    """docs/cardinality_estimates.md must be regenerated when the estimator changes.

    Plan labels (in backticks) are left out of the comparison: they come from
    ``ir.printer``, and the numbers must match whichever printer is installed.
    """
    import opt_cardinality_table

    def numbers_only(text):
        return re.sub(r"`[^`]*`", "`plan`", text)

    committed = Path(opt_cardinality_table.OUT).read_text(encoding="utf-8")
    assert numbers_only(committed) == numbers_only(opt_cardinality_table.render()), \
        "stale report: run `python tests/opt_cardinality_table.py`"
