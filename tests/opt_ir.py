"""Import this before anything that imports ``ir``.

If Person A's ``ir`` package is complete (every module the optimizer uses
imports cleanly), this does nothing and the tests run against the real IR.

Otherwise it switches to the stand-in in tests/opt_standin. That includes
a partially committed ``ir/``, as on main at c08198f, where ``ir/nodes.py``
imports ``ir.dtype`` and ``ir.expr`` but neither file exists. To switch,
it drops the half-imported ``ir`` modules and puts the stand-in first on
sys.path. The check is the same one Person C's runtime/_compat.py uses, so
all optimizer and runtime tests move to the real IR in the same commit.
"""

import sys
from pathlib import Path

_REQUIRED = ("ir.dtype", "ir.expr", "ir.nodes", "ir.printer")

try:
    for _name in _REQUIRED:
        __import__(_name)
    USING_STANDIN = False
except ImportError:
    for _name in [m for m in sys.modules if m == "ir" or m.startswith("ir.")]:
        del sys.modules[_name]
    sys.path.insert(0, str(Path(__file__).parent / "opt_standin"))
    for _name in _REQUIRED:
        __import__(_name)
    USING_STANDIN = True
