"""`compile_and_run(source, tables) -> Table`: execute a generated module.

The source is compiled under a unique pseudo-filename and registered with `linecache`,
so ordinary tracebacks show the generated lines. On any failure — syntax error, missing
`run`, or an exception raised inside the generated code — a `GeneratedCodeError` is
raised whose message contains the full numbered source with the failing line marked:

    GeneratedCodeError: ZeroDivisionError: division by zero  (generated line 7)
          5 |     x_1 = 1
          6 |     y_1 = 0
    -->   7 |     z_1 = x_1 / y_1
          8 |     return z_1

The original exception is chained (`raise ... from`), so nothing is lost.
"""
from __future__ import annotations

import itertools
import linecache
import traceback

from runtime.table import Table

_counter = itertools.count(1)


class GeneratedCodeError(Exception):
    def __init__(self, message: str, source: str, lineno: int | None):
        self.source = source
        self.lineno = lineno
        where = f"  (generated line {lineno})" if lineno else ""
        super().__init__(f"{message}{where}\n\n{number_source(source, lineno)}")


def number_source(source: str, highlight: int | None = None) -> str:
    """The source with line numbers; `highlight` (1-based) gets a `-->` marker."""
    lines = source.splitlines()
    width = len(str(len(lines)))
    out = []
    for i, text in enumerate(lines, start=1):
        marker = "-->" if i == highlight else "   "
        out.append(f"{marker} {i:>{width}} | {text}")
    return "\n".join(out)


def compile_module(source: str) -> dict:
    """Compile and execute the module body; return its namespace."""
    filename = f"<qcompiler-generated-{next(_counter)}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as e:
        raise GeneratedCodeError(f"SyntaxError: {e.msg}", source, e.lineno) from e

    namespace = {"__name__": "qcompiler_generated", "__file__": filename}
    try:
        exec(code, namespace)
    except Exception as e:
        raise _runtime_error(e, source, filename) from e
    if not callable(namespace.get("run")):
        raise GeneratedCodeError("generated module defines no run(tables)", source, None)
    return namespace


def compile_and_run(source: str, tables: dict) -> Table:
    namespace = compile_module(source)
    filename = namespace["__file__"]
    try:
        result = namespace["run"](tables)
    except GeneratedCodeError:
        raise
    except Exception as e:
        raise _runtime_error(e, source, filename) from e
    if not isinstance(result, Table):
        raise GeneratedCodeError(
            f"run() returned {type(result).__name__}, expected runtime.Table", source, None)
    return result


def _runtime_error(exc: Exception, source: str, filename: str) -> GeneratedCodeError:
    # Innermost frame that belongs to the generated module = the generated line at fault.
    lineno = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == filename:
            lineno = frame.lineno
    return GeneratedCodeError(f"{type(exc).__name__}: {exc}", source, lineno)
