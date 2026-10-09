"""Contract §7, end to end: interpret(plan) against compile_and_run(generate(optimize(plan))).

The reference is Person C's interpreter running the plan as written; the
candidate is Person C's generated code running the fully optimized plan.
These tests need Person C's ``runtime`` and ``codegen`` packages, which are
not on main yet, so they skip until those land. With all three members'
code merged they run on every query of the suite.
"""

import warnings

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
import optimizer
from opt_reference_eval import _is_ordered

try:
    from codegen import compile_and_run, generate
    from runtime import Table, compare_tables, interpret
except ImportError as e:  # Person C's code isn't in this checkout
    pytest.skip(f"Person C's runtime/codegen not available: {e}", allow_module_level=True)


def runtime_tables(tables, schemas):
    """The suite's (names, rows) tables as Person C's runtime Tables."""
    return {
        name: Table.from_rows([(col, dtype, name) for col, dtype in schemas[name]], rows)
        for name, (_, rows) in tables.items()
    }


TABLES = runtime_tables(S.TABLES, S.SCHEMAS)


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_generated_code_for_the_optimized_plan_matches_the_interpreter(query):
    plan = query.plan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        optimized, _ = optimizer.optimize(plan, S.CATALOG)
    expected = interpret(plan, TABLES)
    actual = compile_and_run(generate(optimized, S.CATALOG, mode="compiled"), TABLES)
    equal, why = compare_tables(expected, actual, ordered=_is_ordered(plan))
    assert equal, why


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_interpreter_agrees_with_the_reference_evaluator(query):
    """The tests' reference evaluator and Person C's interpreter: same rows."""
    from opt_reference_eval import evaluate

    plan = query.plan
    ours = evaluate(plan, S.TABLES)
    theirs = interpret(plan, TABLES)
    assert theirs.num_rows == len(ours.rows)
    canon = sorted((repr(_canon(r)) for r in ours.rows))
    assert canon == sorted(repr(_canon(r)) for r in theirs.to_rows())


def _canon(row):
    import datetime

    out = []
    for v in row:
        if isinstance(v, float):
            v = round(v, 6) + 0.0
        elif isinstance(v, datetime.date):
            v = v.isoformat()
        out.append(v)
    return tuple(out)


# SQL bound by Person A's frontend. ORDER BY on an aggregate the SELECT list
# leaves out binds to a hidden aggregate column: Aggregate (with the hidden
# column), Project (keeps it), Sort (on it), Project (drops it).
ORDER_BY_AGGREGATE = [
    "SELECT c.nation FROM customer c GROUP BY c.nation ORDER BY SUM(c.acctbal) DESC",
    "SELECT c.nation, SUM(c.acctbal) AS s FROM customer c GROUP BY c.nation ORDER BY SUM(c.acctbal)",
    "SELECT c.nation, COUNT(*) AS n FROM customer c GROUP BY c.nation "
    "HAVING SUM(c.acctbal) > 0 ORDER BY AVG(c.acctbal), c.nation",
    "SELECT c.nation FROM customer c JOIN orders o ON o.cust_id = c.id WHERE o.total_price > 100 "
    "GROUP BY c.nation ORDER BY MAX(o.total_price) DESC LIMIT 3",
]


@pytest.fixture(scope="module")
def tiny():
    from bench.data.generate import create_test_catalog, generate_dataset

    arrow = generate_dataset("tiny", seed=42)
    tables = {name: Table.from_arrow(t, table=name) for name, t in arrow.items()}
    return create_test_catalog("tiny", seed=42), tables


@pytest.mark.parametrize("sql", ORDER_BY_AGGREGATE)
def test_order_by_an_aggregate_bound_from_sql(sql, tiny):
    from frontend import parse_and_bind

    catalog, tables = tiny
    plan = parse_and_bind(sql, catalog)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        optimized, _ = optimizer.optimize(plan, catalog)
    expected = interpret(plan, tables)
    actual = compile_and_run(generate(optimized, catalog, mode="compiled"), tables)
    assert actual.num_rows > 0
    equal, why = compare_tables(expected, actual, ordered=True)
    assert equal, why
