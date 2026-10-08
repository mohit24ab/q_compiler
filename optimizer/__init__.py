"""Query optimizer (Person B).

``optimize`` is the entry point from Contract §5. The framework is
deliberately IR-agnostic: it needs plans to be immutable and comparable with
``==``, and it never imports Person A's node classes.
"""

from __future__ import annotations

from typing import Any

from optimizer.manager import (
    IterationCapWarning,
    OptimizerManager,
    OptimizerWarning,
    OscillationWarning,
    PlanMutationError,
)
from optimizer.pass_base import OptimizerPass
from optimizer.trace import PassTrace, diff_plans, render_trace, render_traces, side_by_side


def default_passes() -> list[OptimizerPass]:
    """Return the production pipeline, in order. It is empty until the first rewrite lands in B2."""
    return []


def optimize(plan: Any, catalog: Any) -> tuple[Any, list[PassTrace]]:
    """Run the default pipeline to a fixed point (Contract §5)."""
    return OptimizerManager(default_passes()).optimize(plan, catalog)


__all__ = [
    "IterationCapWarning",
    "OptimizerManager",
    "OptimizerPass",
    "OptimizerWarning",
    "OscillationWarning",
    "PassTrace",
    "PlanMutationError",
    "default_passes",
    "diff_plans",
    "optimize",
    "render_trace",
    "render_traces",
    "side_by_side",
]
