"""Rendering plans, plan diffs, and pass traces.

These tests pin ``formatter=generic_format`` so they don't depend on whether
Person A's ``ir.printer`` is installed. The fallback logic itself is tested
separately, at the bottom of this file.
"""

import sys
import types

import pytest

from optimizer.manager import OptimizerManager
from optimizer.trace import (
    PassTrace,
    default_formatter,
    diff_plans,
    generic_format,
    render_trace,
    render_traces,
    side_by_side,
)

from opt_stub_ir import Filter, Join, Project, Scan, transform_up


def diff(before, after):
    return diff_plans(before, after, formatter=generic_format)


# --------------------------------------------------------------------------
# generic_format
# --------------------------------------------------------------------------


def test_generic_format_renders_an_indented_tree():
    plan = Project(Filter(Scan("sales", ["id", "amount"]), "amount > 100"), [("id", "id")])

    assert generic_format(plan) == (
        "Project[exprs=[('id', 'id')]]\n"
        "  Filter[predicate='amount > 100']\n"
        "    Scan[table='sales', columns=['id', 'amount']]"
    )


def test_generic_format_omits_child_and_none_fields():
    plan = Join(Scan("orders"), Scan("customer"), "orders.cust_id = customer.id")

    assert generic_format(plan) == (
        "Join[condition='orders.cust_id = customer.id', kind='inner']\n"
        "  Scan[table='orders']\n"
        "  Scan[table='customer']"
    )


def test_generic_format_falls_back_to_repr_for_non_tree_plans():
    assert generic_format("just a string") == "'just a string'"


# --------------------------------------------------------------------------
# diff_plans
# --------------------------------------------------------------------------


def test_diff_of_identical_plans_marks_nothing():
    plan = Filter(Scan("sales"), "amount > 100")

    assert diff(plan, plan) == (
        "  Filter[predicate='amount > 100']\n"
        "    Scan[table='sales']"
    )


def test_diff_shows_a_changed_node_as_remove_plus_add():
    before = Filter(Scan("sales"), "amount > 100")
    after = Filter(Scan("sales", ["amount"]), "amount > 100")

    assert diff(before, after) == (
        "  Filter[predicate='amount > 100']\n"
        "-   Scan[table='sales']\n"
        "+   Scan[table='sales', columns=['amount']]"
    )


def test_diff_of_a_removed_node_shows_its_subtree_at_both_depths():
    scan = Scan("sales")
    before = Project(Filter(scan, "true"), [("id", "id")])
    after = Project(scan, [("id", "id")])

    assert diff(before, after) == (
        "  Project[exprs=[('id', 'id')]]\n"
        "-   Filter[predicate='true']\n"
        "-     Scan[table='sales']\n"
        "+   Scan[table='sales']"
    )


def _pushdown_example():
    orders, customer = Scan("orders"), Scan("customer")
    cond = "orders.cust_id = customer.id"
    before = Filter(Join(orders, customer, cond), "orders.total > 100")
    after = Join(Filter(orders, "orders.total > 100"), customer, cond)
    return before, after


def test_diff_of_a_predicate_moved_below_a_join():
    before, after = _pushdown_example()

    assert diff(before, after) == (
        "- Filter[predicate='orders.total > 100']\n"
        "-   Join[condition='orders.cust_id = customer.id', kind='inner']\n"
        "+ Join[condition='orders.cust_id = customer.id', kind='inner']\n"
        "+   Filter[predicate='orders.total > 100']\n"
        "      Scan[table='orders']\n"
        "-     Scan[table='customer']\n"
        "+   Scan[table='customer']"
    )


def test_both_trees_can_be_read_back_out_of_any_diff():
    """Dropping the + lines gives the before tree; dropping the - lines gives the after tree."""
    before, after = _pushdown_example()
    cases = [
        (before, after),
        (after, before),
        (Project(Filter(Scan("s"), "true"), [("a", "a")]), Project(Scan("s"), [("a", "a")])),
        (Scan("s"), Join(Scan("s"), Scan("t"), "s.k = t.k")),
    ]
    for old, new in cases:
        lines = diff(old, new).splitlines()
        assert "\n".join(l[2:] for l in lines if not l.startswith("+")) == generic_format(old)
        assert "\n".join(l[2:] for l in lines if not l.startswith("-")) == generic_format(new)


def test_diff_uses_a_custom_formatter():
    out = diff_plans(1, 2, formatter=lambda plan: f"plan #{plan}")
    assert out == "- plan #1\n+ plan #2"


# --------------------------------------------------------------------------
# side_by_side
# --------------------------------------------------------------------------


def test_side_by_side_marks_differing_rows():
    before = Filter(Scan("sales"), "a > 1")
    after = Filter(Scan("sales", ["a"]), "a > 1")

    assert side_by_side(before, after, generic_format) == (
        "  before                    | after\n"
        "  Filter[predicate='a > 1'] | Filter[predicate='a > 1']\n"
        "!   Scan[table='sales']     |   Scan[table='sales', columns=['a']]"
    )


def test_side_by_side_leaves_the_shorter_side_blank():
    scan = Scan("t")
    before = Filter(scan, "true")
    after = scan

    assert side_by_side(before, after, generic_format) == (
        "  before                   | after\n"
        "! Filter[predicate='true'] | Scan[table='t']\n"
        "!   Scan[table='t']        |"
    )


# --------------------------------------------------------------------------
# render_trace / render_traces
# --------------------------------------------------------------------------


def test_render_unchanged_trace_is_header_only():
    plan = Scan("sales")
    trace = PassTrace("pruning", plan, plan, changed=False, iteration=2)

    assert render_trace(trace, generic_format) == "[iteration 2] pruning: no change"


def test_render_changed_trace_is_header_plus_diff():
    before = Scan("sales")
    after = Scan("sales", ["id"])
    trace = PassTrace("pruning", before, after, changed=True, iteration=1)

    assert render_trace(trace, generic_format) == (
        "[iteration 1] pruning: changed\n"
        "- Scan[table='sales']\n"
        "+ Scan[table='sales', columns=['id']]"
    )
    assert render_trace(trace, generic_format, style="side_by_side") == (
        "[iteration 1] pruning: changed\n"
        "  before              | after\n"
        "! Scan[table='sales'] | Scan[table='sales', columns=['id']]"
    )


def test_render_trace_rejects_unknown_style():
    trace = PassTrace("p", "a", "b", changed=True)
    with pytest.raises(ValueError, match="unknown style"):
        render_trace(trace, generic_format, style="fancy")


def _drop_true_filters(plan, catalog):
    return transform_up(
        plan, lambda n: n.child if isinstance(n, Filter) and n.predicate == "true" else n
    )


class DropTrueFilters:
    name = "drop_true_filters"

    def apply(self, plan, catalog):
        return _drop_true_filters(plan, catalog)


class Identity:
    name = "identity"

    def apply(self, plan, catalog):
        return plan


def _run_toy_pipeline():
    plan = Project(Filter(Scan("sales"), "true"), [("id", "id")])
    return OptimizerManager([Identity(), DropTrueFilters()]).optimize(plan)


def test_render_traces_end_to_end_skips_no_op_traces_by_default():
    _, traces = _run_toy_pipeline()

    assert render_traces(traces, generic_format) == (
        "4 pass application(s) over 2 iteration(s), 1 changed the plan: drop_true_filters x1\n"
        "\n"
        "[iteration 1] drop_true_filters: changed\n"
        "  Project[exprs=[('id', 'id')]]\n"
        "-   Filter[predicate='true']\n"
        "-     Scan[table='sales']\n"
        "+   Scan[table='sales']"
    )


def test_render_traces_can_include_every_trace():
    _, traces = _run_toy_pipeline()

    out = render_traces(traces, generic_format, only_changed=False)

    headers = [line for line in out.splitlines() if line.startswith("[iteration")]
    assert headers == [
        "[iteration 1] identity: no change",
        "[iteration 1] drop_true_filters: changed",
        "[iteration 2] identity: no change",
        "[iteration 2] drop_true_filters: no change",
    ]


def test_render_traces_of_an_empty_run():
    assert render_traces([]) == "0 pass application(s) over 0 iteration(s), 0 changed the plan"


# --------------------------------------------------------------------------
# default_formatter
# --------------------------------------------------------------------------


def test_default_formatter_falls_back_when_ir_printer_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "ir.printer", None)  # makes the import fail

    assert default_formatter() is generic_format


def test_default_formatter_uses_ir_printer_when_present(monkeypatch):
    def format_plan(plan):
        return "from ir.printer"

    fake_ir = types.ModuleType("ir")
    fake_printer = types.ModuleType("ir.printer")
    fake_printer.format_plan = format_plan
    monkeypatch.setitem(sys.modules, "ir", fake_ir)
    monkeypatch.setitem(sys.modules, "ir.printer", fake_printer)

    assert default_formatter() is format_plan
