from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PassTrace:
    """
    Records what happened during one optimizer pass.
    """

    pass_name: str
    plan_before: Any
    plan_after: Any
    changed: bool