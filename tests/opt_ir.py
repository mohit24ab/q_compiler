"""Import this before anything that imports ``ir``.

If Person A's ``ir`` package is importable, this does nothing. Otherwise it
appends the stand-in in tests/opt_standin to the *end* of sys.path. That
makes the real package take precedence as soon as it exists, so optimizer
tests switch over to it with no edits.
"""

import sys
from pathlib import Path

try:
    import ir  # noqa: F401
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).parent / "opt_standin"))
    import ir  # noqa: F401

USING_STANDIN = "opt_standin" in (ir.__file__ or "")
