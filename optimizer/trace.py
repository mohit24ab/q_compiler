"""Pass traces and how to render them.

Every time the manager applies a pass it records one ``PassTrace``
(Contract §5). The before/after evidence in the final report and the
ablation study are both built from these traces.

Traces keep references to plans, not copies. Plans are immutable, so
consecutive traces share structure, which keeps storing them cheap. That is
also why the manager rejects any pass that mutates its input: an in-place
edit would silently rewrite the history held in earlier traces.
"""

from __future__ import annotations

import dataclasses
import difflib
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from typing import Any, Callable, Iterable

PlanFormatter = Callable[[Any], str]


@dataclass(frozen=True)
class PassTrace:
    """One application of one pass.

    ``changed`` is structural: a pass that rebuilds an identical tree did not
    change the plan. ``iteration`` is the 1-based manager iteration the pass
    ran in. It is 0 for traces built outside the manager.
    """

    pass_name: str
    plan_before: Any
    plan_after: Any
    changed: bool
    iteration: int = 0


# --------------------------------------------------------------------------
# Rendering a single plan
# --------------------------------------------------------------------------


def default_formatter() -> PlanFormatter:
    """Return ``ir.printer.format_plan`` once Person A ships it, else ``generic_format``."""
    try:
        from ir.printer import format_plan
    except ModuleNotFoundError as exc:
        # Fall back only if the IR package itself is missing. A broken import
        # *inside* the printer is a real bug and should surface.
        if exc.name not in ("ir", "ir.printer"):
            raise
        return generic_format
    return format_plan


def generic_format(plan: Any) -> str:
    """Render any tree of objects that expose ``children`` as an indented outline.

    Use this for hand-built trees, and for any plan while the IR printer does
    not exist yet. A dataclass node prints as ``Name[field=value, ...]``.
    Fields that hold children, fields set to ``None``, and fields declared
    with ``repr=False`` are left out. An object without a ``children``
    tuple prints as its ``repr``.
    """
    lines: list[str] = []
    _outline(plan, 0, lines)
    return "\n".join(lines)


def _outline(node: Any, depth: int, lines: list[str]) -> None:
    indent = "  " * depth
    children = getattr(node, "children", None)
    if not isinstance(children, tuple):
        lines.extend(indent + line for line in repr(node).splitlines())
        return
    lines.append(indent + _label(node, children))
    for child in children:
        _outline(child, depth + 1, lines)


def _label(node: Any, children: tuple) -> str:
    name = type(node).__name__
    if not dataclasses.is_dataclass(node):
        return name
    child_ids = {id(child) for child in children}
    parts = []
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if not field.repr or value is None or id(value) in child_ids:
            continue
        parts.append(f"{field.name}={value!r}")
    return f"{name}[{', '.join(parts)}]"


# --------------------------------------------------------------------------
# Diffing two plans
# --------------------------------------------------------------------------


def _line_opcodes(before: Any, after: Any, formatter: PlanFormatter | None):
    render = formatter or default_formatter()
    old = render(before).splitlines()
    new = render(after).splitlines()
    # Match whole lines, indentation included. If indentation were ignored,
    # a node that swapped places with its parent (as happens in predicate
    # pushdown) could be shown as "unchanged" at only one of its two depths,
    # and one of the two trees read back from the diff would be wrong.
    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    return old, new, matcher.get_opcodes()


def diff_plans(before: Any, after: Any, formatter: PlanFormatter | None = None) -> str:
    """Unified line diff of two rendered plans, showing the full tree.

    Each output line starts with a two-character marker: ``"- "`` for a
    removed line, ``"+ "`` for an added line, and two spaces for an
    unchanged line. The full tree is always shown, not just the hunks
    around each change, because a plan only makes sense whole. Dropping the
    ``+`` lines gives back the before tree exactly; dropping the ``-``
    lines gives the after tree.
    """
    old, new, opcodes = _line_opcodes(before, after, formatter)
    out: list[str] = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            out.extend("  " + line for line in new[j1:j2])
        else:
            out.extend("- " + line for line in old[i1:i2])
            out.extend("+ " + line for line in new[j1:j2])
    return "\n".join(out)


def side_by_side(before: Any, after: Any, formatter: PlanFormatter | None = None) -> str:
    """Show the before and after trees in two columns. Rows that differ are marked ``!``.

    Both trees appear whole and in their own shape, which suits screenshots.
    Rows are aligned with the same line matching as ``diff_plans``. Where
    one side has more lines than the other, the shorter side is left blank.
    """
    old, new, opcodes = _line_opcodes(before, after, formatter)
    width = max([len("before"), *map(len, old)])
    rows = [f"  {'before':<{width}} | after"]
    for tag, i1, i2, j1, j2 in opcodes:
        marker = " " if tag == "equal" else "!"
        for left, right in zip_longest(old[i1:i2], new[j1:j2], fillvalue=""):
            rows.append(f"{marker} {left:<{width}} | {right}".rstrip())
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Rendering traces
# --------------------------------------------------------------------------


_STYLES = {"unified": diff_plans, "side_by_side": side_by_side}


def render_trace(
    trace: PassTrace,
    formatter: PlanFormatter | None = None,
    style: str = "unified",
) -> str:
    """Return a header line for the trace, followed by its diff if the pass changed the plan.

    ``style`` is ``"unified"`` (see ``diff_plans``) or ``"side_by_side"``.
    """
    if style not in _STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {sorted(_STYLES)}")
    status = "changed" if trace.changed else "no change"
    header = f"[iteration {trace.iteration}] {trace.pass_name}: {status}"
    if not trace.changed:
        return header
    return header + "\n" + _STYLES[style](trace.plan_before, trace.plan_after, formatter)


def render_traces(
    traces: Iterable[PassTrace],
    formatter: PlanFormatter | None = None,
    only_changed: bool = True,
    style: str = "unified",
) -> str:
    """Render a whole optimizer run: a summary line, then one block per trace.

    By default, traces where the pass did nothing are left out of the body.
    They still count towards the totals in the summary line.
    """
    traces = list(traces)
    fired = Counter(t.pass_name for t in traces if t.changed)
    iterations = max((t.iteration for t in traces), default=0)
    summary = (
        f"{len(traces)} pass application(s) over {iterations} iteration(s), "
        f"{sum(fired.values())} changed the plan"
    )
    if fired:
        summary += ": " + ", ".join(f"{name} x{count}" for name, count in fired.items())
    blocks = [summary]
    blocks.extend(
        render_trace(t, formatter, style) for t in traces if t.changed or not only_changed
    )
    return "\n\n".join(blocks)
