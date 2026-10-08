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
    from optimizer.constant_folding import ConstantFolding
    from optimizer.join_reordering import JoinReordering
    from optimizer.predicate_pushdown import PredicatePushdown

    # Folding first: a simplified predicate exposes more conjuncts to push.
    # Pushdown before reordering: filters at the scans make the estimates the
    # reordering is costed from reflect them, and WHERE conjuncts become join
    # conditions it can place.
    # Reordering before pruning: pruning inserts Projects between joins, which
    # would split the join trees reordering works on.
    return [ConstantFolding(), PredicatePushdown(), JoinReordering(), ColumnPruning()]


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
