"""Runs an ordered list of passes to a fixed point, recording a trace of each step.

Each iteration applies every pass once, in order. The run stops when one of
these happens:

* **Fixed point.** A full iteration in which no pass changed the plan.
* **Oscillation.** At the end of an iteration, the plan equals the plan at
  the start of an earlier iteration. Passes are deterministic, so the run
  would repeat the same cycle forever. The manager warns, naming the passes
  involved, and stops.
* **Iteration cap.** ``max_iterations`` iterations ran without either of the
  above. The manager warns and stops.

Plans are compared structurally (``==``). IR nodes are dataclasses, so
equality compares whole trees. The manager never hashes plans, because the IR
has list-valued fields and so its nodes are unhashable.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable

from optimizer.pass_base import OptimizerPass
from optimizer.trace import PassTrace

DEFAULT_MAX_ITERATIONS = 10


class OptimizerWarning(UserWarning):
    """Base class for warnings raised by the pass pipeline."""


class OscillationWarning(OptimizerWarning):
    """The pipeline returned the plan to a state it had already been in."""


class IterationCapWarning(OptimizerWarning):
    """The iteration cap was hit before the pipeline reached a fixed point."""


class PlanMutationError(RuntimeError):
    """A pass modified its input plan in place instead of returning a new tree."""


def plans_equal(a: Any, b: Any) -> bool:
    """Return True if two plans are structurally equal. ``a is b`` is checked first as a fast path."""
    return a is b or a == b


class OptimizerManager:
    """Applies ``passes`` repeatedly until the plan stops changing.

    ``check_immutability`` takes a ``repr`` fingerprint of each pass's input
    before the pass runs and compares it afterwards. If the input changed,
    the manager raises ``PlanMutationError``. Plan nodes are frozen
    dataclasses, but their list-valued fields (``Scan.columns``,
    ``Project.exprs``, and others) can still be mutated in place. This check
    catches that.
    """

    def __init__(
        self,
        passes: Iterable[OptimizerPass],
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        check_immutability: bool = True,
    ):
        self.passes = tuple(passes)
        for opt_pass in self.passes:
            if not isinstance(opt_pass, OptimizerPass):
                raise TypeError(
                    f"{opt_pass!r} is not an OptimizerPass: it needs a `name` "
                    f"attribute and an `apply(plan, catalog)` method"
                )
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1, got {max_iterations}")
        self.max_iterations = max_iterations
        self.check_immutability = check_immutability

    def optimize(self, plan: Any, catalog: Any = None) -> tuple[Any, list[PassTrace]]:
        """Run the pipeline. Returns ``(final_plan, traces)``, with one trace per pass application."""
        traces: list[PassTrace] = []
        # boundary_states[k] is the plan at the start of iteration k + 1.
        boundary_states = [plan]
        current = plan

        for iteration in range(1, self.max_iterations + 1):
            first = len(traces)
            for opt_pass in self.passes:
                current = self._apply(opt_pass, current, catalog, iteration, traces)

            if not any(t.changed for t in traces[first:]):
                return current, traces

            for k, earlier in enumerate(boundary_states):
                if plans_equal(current, earlier):
                    cycle_start = k + 1
                    involved = _changed_pass_names(traces, cycle_start)
                    warnings.warn(
                        f"oscillation detected: after iteration {iteration} the plan "
                        f"is identical to the plan at the start of iteration "
                        f"{cycle_start}. The passes that changed it in that span "
                        f"({', '.join(involved)}) undo each other, so the pipeline "
                        f"would never reach a fixed point. Stopping early.",
                        OscillationWarning,
                        stacklevel=2,
                    )
                    return current, traces
            boundary_states.append(current)

        warnings.warn(
            f"iteration cap reached: {self.max_iterations} iteration(s) ran without "
            f"reaching a fixed point. The last iteration changed the plan via "
            f"{', '.join(_changed_pass_names(traces, self.max_iterations))}. "
            f"Returning the current plan.",
            IterationCapWarning,
            stacklevel=2,
        )
        return current, traces

    def _apply(
        self,
        opt_pass: OptimizerPass,
        plan: Any,
        catalog: Any,
        iteration: int,
        traces: list[PassTrace],
    ) -> Any:
        fingerprint = repr(plan) if self.check_immutability else None
        try:
            result = opt_pass.apply(plan, catalog)
        except Exception as exc:
            exc.add_note(f"raised by optimizer pass {opt_pass.name!r} in iteration {iteration}")
            raise
        if result is None:
            raise TypeError(
                f"optimizer pass {opt_pass.name!r} returned None. A pass must "
                f"return a plan. To signal no change, return the input unchanged."
            )
        if fingerprint is not None and repr(plan) != fingerprint:
            raise PlanMutationError(
                f"optimizer pass {opt_pass.name!r} mutated its input plan in place "
                f"(iteration {iteration}). Passes must return a new tree and leave "
                f"their input untouched (Contract §4)."
            )
        changed = not plans_equal(plan, result)
        # When nothing changed, keep the input object. That way each trace's
        # plan_after *is* the next trace's plan_before.
        after = result if changed else plan
        traces.append(PassTrace(opt_pass.name, plan, after, changed, iteration))
        return after


def _changed_pass_names(traces: list[PassTrace], from_iteration: int) -> list[str]:
    """List the passes that changed the plan in ``from_iteration`` or later, in first-seen order, without duplicates."""
    names: dict[str, None] = {}
    for t in traces:
        if t.changed and t.iteration >= from_iteration:
            names.setdefault(t.pass_name)
    return list(names)
