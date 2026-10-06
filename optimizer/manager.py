import warnings
from typing import Any, Iterable

from optimizer.trace import PassTrace


class OptimizerManager:
    """
    Runs optimizer passes repeatedly until a fixed point is reached,
    an iteration limit is hit, or oscillation is detected.
    """

    def __init__(
        self,
        passes: Iterable[Any],
        max_iterations: int = 20,
    ) -> None:
        self.passes = list(passes)
        self.max_iterations = max_iterations

    def optimize(
        self,
        plan: Any,
        catalog: Any = None,
    ) -> tuple[Any, list[PassTrace]]:
        """
        Run all optimizer passes until the plan reaches a fixed point.

        Returns:
            The final plan and a trace entry for every pass execution.
        """

        current_plan = plan
        traces: list[PassTrace] = []

        # Keep track of plans we have already seen.
        seen_plans: set[str] = set()

        for iteration in range(self.max_iterations):
            plan_key = self._plan_key(current_plan)

            if plan_key in seen_plans:
                warnings.warn(
                    "Optimizer oscillation detected; stopping optimization."
                )
                break

            seen_plans.add(plan_key)

            iteration_changed = False

            for optimizer_pass in self.passes:
                before = current_plan
                after = optimizer_pass.apply(before, catalog)

                changed = after != before

                traces.append(
                    PassTrace(
                        pass_name=optimizer_pass.name,
                        plan_before=before,
                        plan_after=after,
                        changed=changed,
                    )
                )

                current_plan = after

                if changed:
                    iteration_changed = True

            if not iteration_changed:
                break

        else:
            warnings.warn(
                f"Optimizer reached iteration cap "
                f"({self.max_iterations}); stopping."
            )

        return current_plan, traces

    @staticmethod
    def _plan_key(plan: Any) -> str:
        """
        Create a stable representation of a plan for oscillation detection.

        The temporary implementation uses repr(). Once Person A's IR is
        available, this can be replaced with a proper structural plan key.
        """
        return repr(plan)