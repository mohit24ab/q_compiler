from typing import Any, Protocol


class OptimizerPass(Protocol):
    """
    Interface that every optimizer pass must follow.

    A pass receives a plan and catalog and returns a NEW plan.
    It must never modify the input plan in place.
    """

    name: str

    def apply(self, plan: Any, catalog: Any) -> Any:
        """
        Apply this optimization pass to a plan.

        Parameters
        ----------
        plan:
            The input PlanNode.

        catalog:
            The catalog containing table metadata and statistics.

        Returns
        -------
        PlanNode
            A new plan after applying the optimization.
        """
        ...