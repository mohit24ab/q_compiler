"""Hand-built plan nodes for optimizer tests, used until Person A's IR lands.

Field names mirror Contract §4 and tests/test_ir_nodes.py, so tests written
against these stubs port to the real IR by changing one import. Expressions
are plain strings: the B1 framework never looks inside them.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scan:
    table: str
    columns: list[str] | None = None
    pushed_predicate: Any = None

    @property
    def children(self) -> tuple:
        return ()

    def replace_children(self, new_children) -> "Scan":
        assert not new_children
        return dataclasses.replace(self)


@dataclass(frozen=True)
class Filter:
    child: Any
    predicate: Any

    @property
    def children(self) -> tuple:
        return (self.child,)

    def replace_children(self, new_children) -> "Filter":
        (child,) = new_children
        return dataclasses.replace(self, child=child)


@dataclass(frozen=True)
class Project:
    child: Any
    exprs: list[tuple[Any, str]]

    @property
    def children(self) -> tuple:
        return (self.child,)

    def replace_children(self, new_children) -> "Project":
        (child,) = new_children
        return dataclasses.replace(self, child=child)


@dataclass(frozen=True)
class Join:
    left: Any
    right: Any
    condition: Any
    kind: str = "inner"

    @property
    def children(self) -> tuple:
        return (self.left, self.right)

    def replace_children(self, new_children) -> "Join":
        left, right = new_children
        return dataclasses.replace(self, left=left, right=right)


def transform_up(node: Any, fn) -> Any:
    """Rebuild the tree bottom-up, applying ``fn`` to each node. Never mutates."""
    new_children = tuple(transform_up(c, fn) for c in node.children)
    if any(new is not old for new, old in zip(new_children, node.children)):
        node = node.replace_children(new_children)
    return fn(node)
