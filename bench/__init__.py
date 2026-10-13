"""Benchmark suite for q_compiler."""
# The names below are re-exported lazily (PEP 562). Importing them eagerly meant
# `from bench import runner` dragged in bench.report -- and with it the frontend,
# the catalog and the IR -- to reach a module that needs none of them, and it put
# bench.report in sys.modules before `python -m bench.report` ran it, which runpy
# reports as "found in sys.modules ... may result in unpredictable behaviour".
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers and editors only; no import at runtime
    from bench.report import (
        BenchmarkRecord,
        export_results_to_csv,
        generate_charts,
        run_benchmark_matrix,
    )

__all__ = [
    "BenchmarkRecord",
    "export_results_to_csv",
    "generate_charts",
    "run_benchmark_matrix",
]


def __getattr__(name: str):
    if name in __all__:
        from bench import report
        return getattr(report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
