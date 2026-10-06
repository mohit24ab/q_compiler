import pytest

from optimizer.manager import OptimizerManager
from optimizer.trace import PassTrace


class Plan:
    """Temporary plan used until Person A provides the real IR."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Plan({self.value!r})"

    def __eq__(self, other):
        return isinstance(other, Plan) and self.value == other.value


class NoOpPass:
    """A pass that deliberately makes no changes."""

    name = "no_op"

    def apply(self, plan, catalog):
        return plan


class ToyPass:
    """Changes A -> B, then leaves B unchanged."""

    name = "toy_pass"

    def apply(self, plan, catalog):
        if plan.value == "A":
            return Plan("B")
        return plan


class ToBPass:
    """Changes A -> B."""

    name = "to_b"

    def apply(self, plan, catalog):
        if plan.value == "A":
            return Plan("B")
        return plan


class ToAPass:
    """Changes B -> A."""

    name = "to_a"

    def apply(self, plan, catalog):
        if plan.value == "B":
            return Plan("A")
        return plan


def test_pass_trace_records_information():
    before = "Plan A"
    after = "Plan B"

    trace = PassTrace(
        pass_name="toy_pass",
        plan_before=before,
        plan_after=after,
        changed=True,
    )

    assert trace.pass_name == "toy_pass"
    assert trace.plan_before == "Plan A"
    assert trace.plan_after == "Plan B"
    assert trace.changed is True


def test_pass_trace_is_immutable():
    trace = PassTrace(
        pass_name="toy_pass",
        plan_before="Plan A",
        plan_after="Plan B",
        changed=True,
    )

    with pytest.raises(AttributeError):
        trace.changed = False


def test_manager_no_op_pass_leaves_plan_unchanged():
    plan = Plan("A")

    manager = OptimizerManager(
        passes=[NoOpPass()]
    )

    final_plan, traces = manager.optimize(plan)

    assert final_plan == plan
    assert len(traces) == 1

    assert traces[0].pass_name == "no_op"
    assert traces[0].plan_before == Plan("A")
    assert traces[0].plan_after == Plan("A")
    assert traces[0].changed is False


def test_manager_reaches_fixed_point():
    plan = Plan("A")

    manager = OptimizerManager(
        passes=[ToyPass()]
    )

    final_plan, traces = manager.optimize(plan)

    # First iteration: A -> B
    # Second iteration: B -> B
    # Therefore B is the fixed point.
    assert final_plan == Plan("B")

    assert len(traces) == 2

    assert traces[0].pass_name == "toy_pass"
    assert traces[0].plan_before == Plan("A")
    assert traces[0].plan_after == Plan("B")
    assert traces[0].changed is True

    assert traces[1].pass_name == "toy_pass"
    assert traces[1].plan_before == Plan("B")
    assert traces[1].plan_after == Plan("B")
    assert traces[1].changed is False


def test_manager_detects_oscillation():
    plan = Plan("A")

    manager = OptimizerManager(
        passes=[ToBPass(), ToAPass()]
    )

    with pytest.warns(
        UserWarning,
        match="oscillation detected",
    ):
        final_plan, traces = manager.optimize(plan)

    # The exact final plan is not important here.
    # The important thing is that the optimizer detected
    # the A -> B -> A -> B cycle and stopped.
    assert len(traces) > 0