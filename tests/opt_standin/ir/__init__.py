"""STAND-IN for Person A's ``ir`` package, for optimizer tests only.

This is not the project's IR. It mirrors the API pinned by
tests/test_ir_nodes.py and tests/test_catalog.py, so the optimizer can be
written against ``ir.nodes`` / ``ir.expr`` now and tested before the real
package lands. tests/opt_ir.py appends this directory to the *end* of
sys.path, so the real ``ir/`` at the repo root always wins once it exists.
Delete this directory when it does.
"""
