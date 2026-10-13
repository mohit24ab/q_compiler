"""Source-text builder for generated Python modules.

    em = Emitter()
    em.import_("import numpy as np")
    with em.block("def run(tables):"):
        t = em.fresh("scan")              # -> "scan_1"
        em.line(f"{t} = tables['sales']")
        em.line(f"return {t}")
    print(em.source())

Tracks indentation, hands out unique variable names, and collects imports into a
de-duplicated block at the top of the module.

Plan text reaches the source only through comment(), a line's `note`, or the module
docstring, and all three escape it: a SQL string literal may hold a newline, and a raw
newline in a comment would end the comment and turn the rest into code. line() refuses
any text that would span more than one physical line.
"""
from __future__ import annotations

import keyword
import re
from contextlib import contextmanager

INDENT = "    "
_LINE_BREAKS = re.compile(r"[\r\n\x00]")  # what ends a line (or the source) for Python


def escape_unprintable(text: str, keep: str = "") -> str:
    """`text` with every unprintable character (newline, CR, NUL, tab, U+2028...) except
    those in `keep` written as its escape sequence, e.g. a newline becomes a backslash
    and an `n`."""
    return "".join(ch if ch.isprintable() or ch in keep
                   else ch.encode("unicode_escape").decode("ascii") for ch in text)


class Emitter:
    def __init__(self):
        self._lines: list[str] = []
        self._depth = 0
        self._imports: list[str] = []
        self._counters: dict[str, int] = {}
        self._reserved: set[str] = {"tables", "run"}

    # ---------------------------------------------------------------- body
    def line(self, text: str = "", note: str | None = None) -> None:
        """Append one line at the current indentation (blank lines stay blank), with an
        optional trailing `# note` escaped like comment()."""
        if _LINE_BREAKS.search(text):
            raise ValueError(f"an emitted line must be one physical line: {text!r}")
        if note is not None:
            text = f"{text}  # {escape_unprintable(note)}"
        self._lines.append(f"{INDENT * self._depth}{text}" if text else "")

    def lines(self, text: str) -> None:
        """Append a multi-line snippet, re-indenting each line."""
        for ln in text.splitlines():
            self.line(ln)

    def comment(self, text: str) -> None:
        """One comment line; unprintable characters in `text` are escaped, never emitted."""
        self.line(f"# {escape_unprintable(text)}".rstrip())

    def blank(self) -> None:
        if self._lines and self._lines[-1] != "" and not self._lines[-1].endswith(":"):
            self._lines.append("")

    @contextmanager
    def block(self, header: str):
        """`header` ends in ':'; everything emitted inside the `with` is indented."""
        if not header.rstrip().endswith(":"):
            raise ValueError(f"block header must end with ':' — got {header!r}")
        self.line(header)
        self._depth += 1
        start = len(self._lines)
        try:
            yield
        finally:
            if len(self._lines) == start:  # an empty block would be a syntax error
                self.line("pass")
            self._depth -= 1

    @contextmanager
    def indent(self):
        """Indent without a block header, e.g. the items of a multi-line list."""
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    def body_text(self) -> str:
        return "\n".join(self._lines)

    # ---------------------------------------------------------------- names
    def fresh(self, hint: str) -> str:
        """A new identifier never handed out before, derived from `hint`."""
        base = re.sub(r"\W+", "_", hint).strip("_").lower() or "v"
        if base[0].isdigit():
            base = f"v_{base}"
        while True:
            n = self._counters.get(base, 0) + 1
            self._counters[base] = n
            name = f"{base}_{n}"
            if name not in self._reserved and not keyword.iskeyword(name):
                self._reserved.add(name)
                return name

    def reserve(self, name: str) -> None:
        """Mark a name (e.g. an imported module alias) as off-limits for fresh()."""
        self._reserved.add(name)

    # ---------------------------------------------------------------- imports
    def import_(self, statement: str) -> None:
        if statement not in self._imports:
            self._imports.append(statement)
        for alias in re.findall(r"\bas\s+(\w+)", statement):
            self.reserve(alias)

    # ---------------------------------------------------------------- output
    def source(self, docstring: str | None = None) -> str:
        parts = []
        if docstring is not None:
            # Backslashes first, so the escapes added after them stay escapes: the
            # module's __doc__ is exactly `docstring` again.
            text = docstring.rstrip().replace("\\", "\\\\").replace('"""', r"\"\"\"")
            text = escape_unprintable(text, keep="\n")
            parts.append(f'"""{text}\n"""')
        if self._imports:
            parts.append("\n".join(self._imports))
        body = "\n".join(self._lines).strip("\n")
        if body:
            parts.append(body)
        return "\n\n\n".join(parts) + "\n"
