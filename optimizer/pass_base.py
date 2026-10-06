"""The interface every optimizer pass implements.

A pass is a pure function from plan to plan: it takes a tree and returns a
NEW tree (Contract §4). Returning the input object unchanged is how a pass
says "nothing to do here". A pass must never mutate its input; the manager
checks for this and raises ``PlanMutationError`` if it happens.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OptimizerPass(Protocol):
    """A single rewrite rule, applied by ``OptimizerManager``.

    ``name`` identifies the pass in traces, warnings, and the ablation study,
    so it should be stable and unique within a pipeline.

    ``apply`` receives the whole plan and the catalog (which may be ``None``
    for passes and tests that do not need statistics). It must be
    deterministic: the same plan and catalog always produce the same result.
    The manager's oscillation detection relies on this.
    """

    name: str

    def apply(self, plan: Any, catalog: Any) -> Any: ...
