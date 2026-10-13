"""Dataset generators for benchmarking and testing."""
# Re-exported lazily, as in bench/__init__.py: an eager import here put
# bench.data.generate in sys.modules before `python -m bench.data.generate`
# executed it, and runpy warns that the module then runs twice.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers and editors only; no import at runtime
    from bench.data.generate import create_test_catalog, generate_dataset

__all__ = ["create_test_catalog", "generate_dataset"]


def __getattr__(name: str):
    if name in __all__:
        from bench.data import generate
        return getattr(generate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
