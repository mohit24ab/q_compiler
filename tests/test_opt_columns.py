"""Tests for optimizer.columns: which columns an expression reads, and which a node produces."""

import pytest

import opt_ir  # noqa: F401
from ir.expr import UnaryOp
from ir.nodes import Aggregate, Scan
from opt_query_suite import CATALOG, SCHEMAS, agg, col, join, keep, lit, op, project, scan
from optimizer.columns import column_refs, output_columns, scanned_tables, table_schema


def test_column_refs_walks_nested_expressions_and_containers():
    expr = op("AND", op(">", col("a", "t"), lit(1)), UnaryOp("NOT", op("=", col("b"), col("c", "u"))))
    assert column_refs(expr) == {("t", "a"), (None, "b"), ("u", "c")}
    assert column_refs([(agg("sum", col("x")), "s"), (agg("count"), "n")]) == {(None, "x")}
    assert column_refs(None) == set()


def test_column_refs_refuses_expressions_it_cannot_see_inside():
    with pytest.raises(TypeError, match="cannot find column references"):
        column_refs(op("=", col("a"), object()))


def test_output_columns_of_scans_joins_projects_and_aggregates():
    j = join(Scan("emp", ["id", "name"], None), scan("dept"), lit(True))
    assert output_columns(j, CATALOG) == [
        ("emp", "id"), ("emp", "name"), ("dept", "id"), ("dept", "name"), ("dept", "budget"),
    ]
    assert output_columns(project(j, *keep("budget")), CATALOG) == [(None, "budget")]
    grouped = Aggregate(scan("emp"), [col("dept_id", "emp")], [(agg("count"), "n")])
    assert output_columns(grouped, CATALOG) == [("emp", "dept_id"), (None, "n")]


def test_output_columns_is_none_when_unknowable():
    assert output_columns(scan("emp"), catalog=None) is None
    computed_key = Aggregate(scan("emp"), [op("+", col("id"), lit(1))], [])
    assert output_columns(computed_key, CATALOG) is None


def test_table_schema_prefers_the_bound_schema_over_the_catalog():
    assert table_schema(scan("emp", bound=True), catalog=None) == SCHEMAS["emp"]
    assert table_schema(scan("emp"), CATALOG) == SCHEMAS["emp"]
    assert table_schema(scan("emp"), catalog=None) is None


def test_scanned_tables():
    plan = project(join(scan("emp"), join(scan("dept"), scan("emp"), lit(True)), lit(True)), *keep("id"))
    assert scanned_tables(plan) == {"emp", "dept"}
