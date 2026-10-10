"""Plan tests for constant folding: expression rules, predicate rules, plan rules, idempotence.

Correctness for every query in the suite is in test_opt_differential.py,
including negative controls for the rewrites that must NOT happen.
"""

import dataclasses

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
from ir.dtype import DType
from ir.expr import Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort
from opt_query_suite import CATALOG, agg, col, date, join, keep, lit, op, project, scan
from optimizer.constant_folding import FALSE, ConstantFolding, fold, is_empty, simplify_predicate
from optimizer.expressions import TRUE

FOLD = ConstantFolding()
X, Y = col("x"), col("y")
P = op(">", X, lit(1))  # an arbitrary non-constant predicate
NULL_INT = Literal(value=None, dtype=DType.INT)
NULL_BOOL = Literal(value=None, dtype=DType.BOOL)


def run(plan, catalog=CATALOG):
    return FOLD.apply(plan, catalog)


def find(plan, kind):
    found = [plan] if isinstance(plan, kind) else []
    for child in plan.children:
        found += find(child, kind)
    return found


def empties(plan) -> int:
    return sum(is_empty(f) for f in find(plan, Filter))


# --------------------------------------------------------------------------
# Expression folding: exact under three-valued logic, valid in any context
# --------------------------------------------------------------------------

FOLDS = [
    # literal arithmetic and comparison
    ("int arithmetic", op("*", lit(2), lit(3)), lit(6)),
    ("int and float mix", op("+", lit(1.5), lit(2)), lit(3.5)),
    ("float division", op("/", lit(7.0), lit(2)), lit(3.5)),
    ("nested arithmetic", op("*", op("+", lit(1), lit(2)), X), op("*", lit(3), X)),
    ("date comparison", op("<", date("2024-01-01"), date("2024-06-01")), TRUE),
    ("string comparison", op(">=", lit("b"), lit("a")), TRUE),
    ("numeric equality across types", op("=", lit(1), lit(1.0)), TRUE),
    ("unary minus", UnaryOp("-", lit(4)), lit(-4)),
    # boolean algebra
    ("x AND true", op("AND", P, TRUE), P),
    ("true AND x", op("AND", TRUE, P), P),
    ("x AND false", op("AND", P, FALSE), FALSE),
    ("x OR true", op("OR", P, TRUE), TRUE),
    ("x OR false", op("OR", P, FALSE), P),
    ("x AND x", op("AND", P, P), P),
    ("x OR x", op("or", P, P), P),
    ("double negation", UnaryOp("NOT", UnaryOp("NOT", P)), P),
    ("NOT true", UnaryOp("NOT", TRUE), FALSE),
    ("NOT NULL", UnaryOp("NOT", NULL_BOOL), NULL_BOOL),
    # NULL handling
    ("comparison with NULL", op("=", X, NULL_INT), NULL_BOOL),
    ("arithmetic with NULL", op("+", NULL_INT, lit(1)), NULL_INT),
    ("NULL IS NULL", UnaryOp("IS NULL", NULL_INT), TRUE),
    ("literal IS_NOT_NULL", UnaryOp("IS_NOT_NULL", lit(5)), TRUE),
    ("NULL AND x stays: FALSE or NULL depending on x", op("AND", NULL_BOOL, P), op("AND", NULL_BOOL, P)),
    # deliberately left unfolded
    ("integer division (semantics not pinned)", op("/", lit(7), lit(2)), op("/", lit(7), lit(2))),
    ("division by zero", op("/", lit(1.0), lit(0)), op("/", lit(1.0), lit(0))),
    ("modulo", op("%", lit(7), lit(2)), op("%", lit(7), lit(2))),
    ("mixed-type comparison", op("<", lit("a"), lit(1)), op("<", lit("a"), lit(1))),
    ("date not written YYYY-MM-DD", op("<", date("19950101"), date("1995-02-01")),
     op("<", date("19950101"), date("1995-02-01"))),
    ("int64 overflow", op("*", lit(2**62), lit(4)), op("*", lit(2**62), lit(4))),
    ("unknown operator", op("LIKE", lit("ab"), lit("a%")), op("LIKE", lit("ab"), lit("a%"))),
    ("contradictions are a predicate rule, not a value rule",
     op("AND", op("=", X, lit(1)), op("=", X, lit(2))), op("AND", op("=", X, lit(1)), op("=", X, lit(2)))),
]


@pytest.mark.parametrize("before, after", [(b, a) for _, b, a in FOLDS], ids=[n for n, _, _ in FOLDS])
def test_fold(before, after):
    assert fold(before) == after
    assert fold(fold(before)) == fold(before)  # idempotent


def test_fold_keeps_literal_types():
    assert fold(op("*", lit(2), lit(3))).dtype == DType.INT
    assert fold(op("+", lit(1.5), lit(2))).dtype == DType.FLOAT
    assert fold(op("=", X, NULL_INT)).dtype == DType.BOOL


def test_fold_returns_the_same_object_when_nothing_changes():
    expr = op("AND", op(">", X, lit(1)), op("<", Y, lit(5)))
    assert fold(expr) is expr


# --------------------------------------------------------------------------
# Predicate simplification: NULL and FALSE both reject the row
# --------------------------------------------------------------------------

PREDICATES = [
    ("NULL conjunct", op("AND", P, op("=", X, NULL_INT)), FALSE),
    ("bare NULL", NULL_BOOL, FALSE),
    ("two equalities", op("AND", op("=", X, lit(1)), op("=", X, lit(2))), FALSE),
    ("empty range", op("AND", op(">", X, lit(5)), op("<", X, lit(3))), FALSE),
    ("flipped literal", op("AND", op("<", lit(5), X), op("<", X, lit(5))), FALSE),
    ("equality vs not-equal", op("AND", op("=", X, lit(3)), op("<>", X, lit(3))), FALSE),
    ("equality outside range", op("AND", op("=", X, lit(9)), op("<=", X, lit(5))), FALSE),
    ("open point range", op("AND", op(">", X, lit(2)), op("<=", X, lit(2))), FALSE),
    ("strings", op("AND", op("=", X, lit("EU")), op("=", X, lit("US"))), FALSE),
    ("all TRUE", op("AND", TRUE, op("=", lit(1), lit(1))), TRUE),
    ("duplicates removed", op("AND", op("AND", P, op("<", Y, lit(2))), P), op("AND", P, op("<", Y, lit(2)))),
]
SATISFIABLE = [
    ("closed point range", op("AND", op(">=", X, lit(2)), op("<=", X, lit(2)))),
    ("overlapping bounds", op("AND", op(">", X, lit(2)), op(">=", X, lit(2)))),
    ("same value, int and float", op("AND", op("=", X, lit(1)), op("=", X, lit(1.0)))),
    ("different columns", op("AND", op("=", X, lit(1)), op("=", Y, lit(2)))),
    ("mixed types are not guessed at", op("AND", op("=", X, lit("a")), op("=", X, lit(1)))),
    # as strings '19950101' > '1995-02-01', but a DATE column reads it as 1995-01-01
    ("a date not written YYYY-MM-DD is not guessed at",
     op("AND", op(">=", X, lit("19950101")), op("<", X, lit("1995-02-01")))),
    ("qualified and unqualified are different columns",
     op("AND", op("=", col("x", "t"), lit(1)), op("=", X, lit(2)))),
]


@pytest.mark.parametrize("before, after", [(b, a) for _, b, a in PREDICATES], ids=[n for n, _, _ in PREDICATES])
def test_simplify_predicate(before, after):
    assert simplify_predicate(before) == after
    assert simplify_predicate(simplify_predicate(before)) == simplify_predicate(before)


@pytest.mark.parametrize("pred", [p for _, p in SATISFIABLE], ids=[n for n, _ in SATISFIABLE])
def test_satisfiable_predicates_are_left_alone(pred):
    assert simplify_predicate(pred) is pred


# --------------------------------------------------------------------------
# Plan rules
# --------------------------------------------------------------------------


def test_true_filter_is_removed():
    plan = project(Filter(child=scan("sales"), predicate=op("=", lit(1), lit(1))), *keep("sale_id"))
    assert run(plan) == project(scan("sales"), *keep("sale_id"))


def test_filter_predicate_is_folded_in_place():
    plan = Filter(child=scan("sales"), predicate=op(">", col("qty"), op("+", lit(1), lit(2))))
    assert run(plan) == Filter(child=scan("sales"), predicate=op(">", col("qty"), lit(3)))


def test_false_is_lifted_to_the_root_through_every_emptiness_preserving_node():
    out = run(S.false_filter_over_join_and_sort())
    assert is_empty(out) and isinstance(out.child, Sort)
    assert empties(out) == 1


@pytest.mark.parametrize("query", [S.empty_side_of_inner_join, S.grouped_aggregate_over_empty, S.limit_zero])
def test_emptiness_propagates_to_the_root(query):
    out = run(query())
    assert is_empty(out) and empties(out) == 1


def test_inner_join_with_false_condition_is_empty():
    plan = join(scan("orders"), scan("customer"), op("AND", op("=", col("o_custkey"), col("c_id")), FALSE))
    out = run(plan)
    assert is_empty(out) and isinstance(out.child, Join)


def test_left_join_with_empty_left_side_is_empty():
    plan = join(Filter(child=scan("customer"), predicate=FALSE), scan("orders"),
                op("=", col("c_id"), col("o_custkey")), kind="left")
    out = run(plan)
    assert is_empty(out) and isinstance(out.child, Join) and empties(out) == 1


def test_emptiness_stops_at_the_right_side_of_a_left_join():
    out = run(S.empty_right_side_of_left_join_must_not_lift())
    (j,) = find(out, Join)
    assert not is_empty(out) and is_empty(j.right)


def test_left_join_with_false_on_condition_is_not_empty():
    plan = join(scan("customer"), scan("orders"), op("=", lit(1), lit(0)), kind="left")
    out = run(plan)
    assert isinstance(out, Join) and out.condition == FALSE


def test_emptiness_stops_at_a_global_aggregate():
    out = run(S.global_aggregate_over_empty_must_not_lift())
    assert isinstance(out, Aggregate) and is_empty(out.child)


def test_scan_pushed_predicate_true_is_dropped_and_false_becomes_empty():
    true_scan = scan("sales", pushed=op("=", lit(1), lit(1)))
    false_scan = scan("sales", pushed=op("AND", op("=", col("qty"), lit(1)), op("=", col("qty"), lit(2))))
    assert run(true_scan) == scan("sales")
    assert run(false_scan) == Filter(child=scan("sales"), predicate=FALSE)


def test_join_condition_is_simplified():
    out = run(S.constant_conjuncts_in_join_and_scan())
    (j,) = find(out, Join)
    assert j.condition == op("=", col("o_custkey"), col("c_id"))


def test_select_expressions_fold_but_a_contradiction_there_stays():
    folded = run(S.folded_select_expressions())
    assert [e for e, _ in folded.exprs][1:] == [lit(6), lit(3.0), op("/", lit(7), lit(2)), lit(-4)]
    plan = S.contradiction_in_select_must_stay()
    assert run(plan) is plan


# --------------------------------------------------------------------------
# No-op Project removal
# --------------------------------------------------------------------------


def test_noop_project_over_scan_is_removed():
    assert run(S.noop_project_over_scan()) == scan("emp")


def test_noop_project_over_explicit_columns_is_removed():
    base = Scan(table="sales", columns=["qty", "sale_id"], pushed_predicate=None)
    assert run(project(base, *keep("qty", "sale_id"))) is base


@pytest.mark.parametrize("exprs", [
    keep("name", "id", "dept_id", "salary"),                               # reordered
    keep("id", "name", "dept_id"),                                         # subset
    [(col("id"), "emp_id")] + keep("name", "dept_id", "salary"),           # renamed
    [(op("+", col("id"), lit(0)), "id")] + keep("name", "dept_id", "salary"),  # computed
], ids=["reordered", "subset", "renamed", "computed"])
def test_projects_that_change_something_are_kept(exprs):
    plan = project(scan("emp"), *exprs)
    assert run(plan) is plan


def test_project_over_a_join_is_kept_even_if_it_passes_everything_through():
    # Codegen's naming of join output columns isn't pinned down; the
    # Project may be what gives them their plain names.
    j = join(Scan("orders", ["o_id", "o_custkey"], None), Scan("customer", ["c_id"], None),
             op("=", col("o_custkey"), col("c_id")))
    plan = project(j, *keep("o_id", "o_custkey", "c_id"))
    assert run(plan) is plan


def test_project_over_unknown_schema_is_kept():
    plan = project(scan("emp"), *keep("id", "name", "dept_id", "salary"))
    assert run(plan, catalog=None) is plan


# --------------------------------------------------------------------------
# Fixed point
# --------------------------------------------------------------------------


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_folding_is_idempotent(query):
    once = run(query.plan)
    assert run(once) == once


def test_untouched_plan_is_returned_as_the_same_object():
    plan = S.join_condition_only_columns()
    assert run(plan) is plan
