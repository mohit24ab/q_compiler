"""Tests for optimizer.expressions: conjunct handling, substitution, and null rejection."""

import pytest

import opt_ir  # noqa: F401
from ir.dtype import DType
from ir.expr import Literal, UnaryOp
from opt_query_suite import col, lit, op
from optimizer.expressions import conjoin, can_be_true, rebuild, split_conjuncts, substitute

A, B, C = (op(">", col(n), lit(1)) for n in "abc")
NULL = Literal(value=None, dtype=DType.INT)


def test_split_flattens_any_nesting_and_conjoin_builds_left_deep():
    assert split_conjuncts(op("AND", A, op("AND", B, C))) == [A, B, C]
    assert split_conjuncts(op("and", op("AND", A, B), C)) == [A, B, C]
    assert split_conjuncts(op("OR", A, B)) == [op("OR", A, B)]
    assert split_conjuncts(None) == []
    assert conjoin([A, B, C]) == op("AND", op("AND", A, B), C)
    assert conjoin([A]) is A
    assert conjoin([]) is None


def test_rebuild_reuses_the_original_when_conjuncts_are_unchanged():
    right_deep = op("AND", A, op("AND", B, C))
    assert rebuild([A, B, C], right_deep) is right_deep
    assert rebuild([A, C], right_deep) == op("AND", A, C)


def test_substitute_replaces_matching_refs_and_shares_the_rest():
    expr = op("AND", op(">", col("value"), lit(10)), op("=", col("region"), lit("EU")))
    out = substitute(expr, {(None, "value"): op("*", col("qty"), col("amount"))})
    assert out == op("AND", op(">", op("*", col("qty"), col("amount")), lit(10)), op("=", col("region"), lit("EU")))
    assert out.right is expr.right  # untouched subtrees are shared, not copied
    assert substitute(expr, {(None, "other"): lit(1)}) is expr
    # Qualified and unqualified references are different keys.
    assert substitute(col("x", "t"), {(None, "x"): lit(1)}) == col("x", "t")


def right_is_null(ref):
    return ref.table == "r"


R, L = col("v", "r"), col("v", "l")


@pytest.mark.parametrize("pred, possible", [
    # NULL-propagating comparisons and arithmetic on the right side reject NULLs.
    (op(">", R, lit(1)), False),
    (op("=", L, R), False),
    (op(">", op("+", R, lit(1)), lit(0)), False),
    (UnaryOp("NOT", op("=", R, lit(1))), False),
    # AND rejects NULLs if either side does; OR only if both do.
    (op("AND", op(">", L, lit(1)), op(">", R, lit(1))), False),
    (op("OR", op(">", R, lit(1)), op("<", R, lit(0))), False),
    (op("OR", op(">", R, lit(1)), op(">", L, lit(1))), True),
    # IS NULL is TRUE exactly for NULL-extended rows; IS NOT NULL rejects them.
    (UnaryOp("IS NULL", R), True),
    (UnaryOp("IS NOT NULL", R), False),
    (UnaryOp("NOT", UnaryOp("IS NULL", R)), False),
    # Predicates that do not read the right side say nothing about its NULLs.
    (op(">", L, lit(1)), True),
    (lit(True), True),
    (lit(False), False),  # never TRUE at all
    (NULL, False),
    # Operators the analysis doesn't know are not assumed to return NULL.
    (op("IS DISTINCT FROM", R, lit(1)), True),
    (UnaryOp("ISNULL", R), True),
    (op(">", UnaryOp("ISNULL", R), lit(0)), True),
])
def test_null_rejection(pred, possible):
    assert can_be_true(pred, right_is_null) is possible
