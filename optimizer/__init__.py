"""Query optimizer (Person B).

``optimize`` is the entry point from Contract §5. The framework (manager,
pass protocol, traces) is deliberately IR-agnostic: it needs plans to be
immutable and comparable with ``==``. The rewrite passes are built on
Person A's ``ir`` package.
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
    """Return the production pipeline, in order."""
    # Imported here, not at module level: rewrite passes need Person A's IR,
    # while the framework itself (manager, traces) must work without it.
    from optimizer.column_pruning import ColumnPruning

    return [ColumnPruning()]


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
