"""OptimizerManager behaviour beyond the basics in test_opt_pass_framework.py.

Most tests use strings or ints as plans. They are immutable, comparable, and
make each pass a one-line lookup. The tree-shaped tests use the stub IR.
"""

import warnings

import pytest

import opt_ir  # noqa: F401  (default_passes() imports the IR)
import optimizer
from optimizer.manager import (
    IterationCapWarning,
    OptimizerManager,
    OscillationWarning,
    PlanMutationError,
)
from optimizer.pass_base import OptimizerPass
from optimizer.trace import PassTrace

from opt_stub_ir import Filter, Project, Scan, transform_up


class FnPass:
    """Wrap a function as a pass so a test can define a rule in one line."""

    def __init__(self, name, fn):
        self.name = name
        self._fn = fn

    def apply(self, plan, catalog):
        return self._fn(plan, catalog)


def rewrite(name, mapping):
    """A pass that replaces plan ``p`` with ``mapping[p]`` and leaves anything else alone."""
    return FnPass(name, lambda plan, catalog: mapping.get(plan, plan))


def run_strict(manager, plan, catalog=None):
    """Optimize with every warning promoted to an error. Use this where the run must converge cleanly."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return manager.optimize(plan, catalog)


# --------------------------------------------------------------------------
# Ordering and fixed point
# --------------------------------------------------------------------------


def test_passes_run_in_declared_order_within_each_iteration():
    manager = OptimizerManager([rewrite("p1", {"a": "b"}), rewrite("p2", {"b": "c"})])

    final, traces = run_strict(manager, "a")

    assert final == "c"
    assert [(t.iteration, t.pass_name, t.changed) for t in traces] == [
        (1, "p1", True),
        (1, "p2", True),
        (2, "p1", False),
        (2, "p2", False),
    ]


def test_a_later_pass_can_enable_an_earlier_one_on_the_next_iteration():
    # p2 turns a into b, which only p1 can rewrite. So p1 needs a second
    # iteration to fire, and a third iteration confirms the fixed point.
    manager = OptimizerManager([rewrite("p1", {"b": "c"}), rewrite("p2", {"a": "b"})])

    final, traces = run_strict(manager, "a")

    assert final == "c"
    assert [t.changed for t in traces] == [False, True, True, False, False, False]
    assert traces[-1].iteration == 3


def test_trace_chain_is_continuous():
    """Each trace starts from the exact plan object the previous trace ended with."""
    plan = "a"
    manager = OptimizerManager([rewrite("p1", {"b": "c"}), rewrite("p2", {"a": "b"})])

    final, traces = run_strict(manager, plan)

    assert traces[0].plan_before is plan
    for prev, nxt in zip(traces, traces[1:]):
        assert nxt.plan_before is prev.plan_after
    assert traces[-1].plan_after is final


def test_structurally_equal_result_counts_as_unchanged():
    # This pass rebuilds an identical tree, a fresh object every time.
    # If the manager compared identity, it would never reach a fixed point.
    rebuild = FnPass("rebuild", lambda plan, catalog: Scan(plan.table, list(plan.columns)))
    plan = Scan("sales", ["id", "amount"])

    final, traces = run_strict(OptimizerManager([rebuild]), plan)

    assert final is plan
    assert len(traces) == 1
    assert traces[0].changed is False


def test_catalog_is_passed_to_every_pass():
    seen = []
    record = FnPass("record", lambda plan, catalog: seen.append(catalog) or plan)
    catalog = object()

    run_strict(OptimizerManager([record, record]), "a", catalog)

    assert seen == [catalog, catalog]


def test_empty_pipeline_returns_input_and_no_traces():
    plan = Scan("sales")
    final, traces = run_strict(OptimizerManager([]), plan)
    assert final is plan
    assert traces == []


# --------------------------------------------------------------------------
# Non-termination: oscillation and the iteration cap
# --------------------------------------------------------------------------


def test_oscillation_warning_names_the_passes_that_undo_each_other():
    manager = OptimizerManager([rewrite("to_b", {"a": "b"}), rewrite("to_a", {"b": "a"})])

    with pytest.warns(OscillationWarning, match=r"oscillation detected.*\(to_b, to_a\)"):
        final, traces = manager.optimize("a")

    assert final == "a"
    assert len(traces) == 2


def test_oscillation_spanning_several_iterations_is_detected():
    # A three-cycle in which each iteration makes "progress" but goes round in a circle.
    rotate = rewrite("rotate", {"a": "b", "b": "c", "c": "a"})

    with pytest.warns(OscillationWarning, match="start of iteration 1.*rotate"):
        final, traces = OptimizerManager([rotate], max_iterations=50).optimize("a")

    assert final == "a"
    assert len(traces) == 3  # detected on the first repeat, not at the cap


def test_cycle_that_does_not_include_the_input_is_detected():
    # a -> b, then b -> c -> b -> ... The cycle starts at iteration 2.
    flip = rewrite("flip", {"a": "b", "b": "c", "c": "b"})

    with pytest.warns(OscillationWarning, match="start of iteration 2"):
        _, traces = OptimizerManager([flip], max_iterations=50).optimize("a")

    assert len(traces) == 3


def test_revisiting_a_state_mid_iteration_is_not_oscillation():
    # x and y undo each other within the iteration, but z still makes
    # progress, so the run converges. Only a repeat at an iteration boundary
    # proves an infinite loop.
    manager = OptimizerManager(
        [rewrite("x", {"a": "b"}), rewrite("y", {"b": "a"}), rewrite("z", {"a": "c"})]
    )

    final, _ = run_strict(manager, "a")

    assert final == "c"


def test_iteration_cap_stops_a_pass_that_never_converges():
    increment = FnPass("increment", lambda plan, catalog: plan + 1)

    with pytest.warns(IterationCapWarning, match="4 iteration.*increment"):
        final, traces = OptimizerManager([increment], max_iterations=4).optimize(0)

    assert final == 4
    assert len(traces) == 4
    assert all(t.changed for t in traces)


# --------------------------------------------------------------------------
# Guard rails on misbehaving passes
# --------------------------------------------------------------------------


def _append_column(plan, catalog):
    plan.columns.append("leaked")
    return plan


def _setattr_on_frozen(plan, catalog):
    object.__setattr__(plan, "table", "other")
    return plan


@pytest.mark.parametrize("mutate", [_append_column, _setattr_on_frozen])
def test_pass_that_mutates_its_input_is_rejected(mutate):
    plan = Scan("sales", ["id"])

    with pytest.raises(PlanMutationError, match="'mutator' mutated its input"):
        OptimizerManager([FnPass("mutator", mutate)]).optimize(plan)


def test_pass_returning_none_is_rejected():
    forgot_return = FnPass("forgot_return", lambda plan, catalog: None)

    with pytest.raises(TypeError, match="'forgot_return' returned None"):
        OptimizerManager([forgot_return]).optimize("a")


def test_exception_inside_a_pass_is_annotated_with_the_pass_name():
    def boom(plan, catalog):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom") as info:
        OptimizerManager([FnPass("exploder", boom)]).optimize("a")

    assert "raised by optimizer pass 'exploder' in iteration 1" in info.value.__notes__


def test_constructor_rejects_non_passes_and_bad_caps():
    with pytest.raises(TypeError, match="is not an OptimizerPass"):
        OptimizerManager([lambda plan, catalog: plan])
    with pytest.raises(ValueError, match="max_iterations"):
        OptimizerManager([], max_iterations=0)


# --------------------------------------------------------------------------
# Hand-built trees
# --------------------------------------------------------------------------


def _drop_true_filters(plan, catalog):
    """Toy tree rewrite: remove every ``Filter[predicate='true']``."""
    return transform_up(
        plan, lambda n: n.child if isinstance(n, Filter) and n.predicate == "true" else n
    )


def test_tree_rewrite_reaches_fixed_point_and_leaves_input_untouched():
    plan = Project(
        Filter(Filter(Scan("sales", ["id", "amount"]), "true"), "amount > 100"),
        [("id", "id")],
    )
    snapshot = repr(plan)

    final, traces = run_strict(
        OptimizerManager([FnPass("drop_true_filters", _drop_true_filters)]), plan
    )

    assert final == Project(Filter(Scan("sales", ["id", "amount"]), "amount > 100"), [("id", "id")])
    assert repr(plan) == snapshot
    assert [t.changed for t in traces] == [True, False]
    # Subtrees the pass did not touch are shared, not copied.
    assert final.child.child is plan.child.child.child


# --------------------------------------------------------------------------
# Contract §5 entry point
# --------------------------------------------------------------------------


def test_optimize_entry_point_runs_the_default_pipeline(monkeypatch):
    monkeypatch.setattr(
        optimizer, "default_passes", lambda: [rewrite("p", {"a": "b"})]
    )

    final, traces = optimizer.optimize("a", None)

    assert final == "b"
    assert all(isinstance(t, PassTrace) for t in traces)
    assert [t.pass_name for t in traces] == ["p", "p"]


def test_default_pipeline_is_well_formed():
    passes = optimizer.default_passes()
    assert all(isinstance(p, OptimizerPass) for p in passes)
    names = [p.name for p in passes]
    assert len(names) == len(set(names)), "pass names must be unique for traces and ablation"
