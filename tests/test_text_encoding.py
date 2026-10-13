"""Every text file this repo reads or writes names its encoding.

Leaving it to the platform means cp1252 on a Windows laptop and utf-8 on the CI
runner, so a report carrying the selectivity rho, an arrow or a section sign
round-trips on one machine and dies on the other. That has already happened to
the cardinality report (unreadable bytes), the two ablation reports and
pass_order.md (UnicodeEncodeError before anything was written), so it is checked
here rather than remembered.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".venv", "__pycache__", ".git", "build", "dist", "node_modules"}
TEXT_METHODS = {"read_text", "write_text"}


def sources() -> list[Path]:
    return [p for p in sorted(ROOT.rglob("*.py")) if not SKIP.intersection(p.parts)]


def unencoded(path: Path) -> list[tuple[int, str]]:
    """Text reads and writes in ``path`` that do not say which encoding."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or any(k.arg == "encoding" for k in node.keywords):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name in TEXT_METHODS:
            found.append((node.lineno, name))
        elif name == "open":
            # open(path) and open(path, "w") are text; "rb"/"wb" carry no encoding.
            mode = next((a.value for a in node.args[1:2]
                         if isinstance(a, ast.Constant) and isinstance(a.value, str)), "r")
            if "b" not in mode:
                found.append((node.lineno, "open"))
    return found


@pytest.mark.parametrize("path", sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_text_io_names_its_encoding(path):
    bad = unencoded(path)
    assert not bad, (f"{path.relative_to(ROOT)}: text I/O with no encoding= at "
                     f"{', '.join(f'line {n} ({call})' for n, call in bad)}")
