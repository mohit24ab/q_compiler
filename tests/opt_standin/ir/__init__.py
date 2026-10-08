"""STAND-IN for Person A's ``ir`` package, for optimizer tests only.

This is not the project's IR. It mirrors the API pinned by
tests/test_ir_nodes.py and tests/test_catalog.py, so the optimizer can be
written against ``ir.nodes`` / ``ir.expr`` now and tested before the real
package is complete. tests/opt_ir.py only puts this on sys.path when the
real ``ir`` package cannot supply every module the optimizer imports.
Delete this directory once it can.
"""
