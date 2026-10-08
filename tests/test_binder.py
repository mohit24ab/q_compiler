from __future__ import annotations

import pyarrow as pa
import pytest

from catalog.catalog import Catalog
from frontend.binder import parse_and_bind
from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, PlanNode, Project, Scan, Sort
from ir.printer import format_plan


@pytest.fixture
def catalog() -> Catalog:
    cat = Catalog()

    sales_table = pa.table({
        "id": [1, 2, 3],
        "amount": [10.0, 20.0, 30.0],
        "region": ["US", "EU", "US"],
        "qty": [1, 2, 3],
    })
    cat.register_table("sales", sales_table)

    orders_table = pa.table({
        "id": [100, 101, 102],
        "cust_id": [1, 2, 1],
        "total": [50.0, 70.0, 90.0],
    })
    cat.register_table("orders", orders_table)

    customer_table = pa.table({
        "id": [1, 2, 3],
        "name": ["ann", "bob", "charlie"],
        "c_segment": ["AUTO", "BUILDING", "AUTO"],
    })
    cat.register_table("customer", customer_table)

    simple_table = pa.table({
        "col": [1, 2],
        "col2": [10, 20],
        "a": [1, 2],
        "b": [3, 4],
    })
    cat.register_table("table", simple_table)

    return cat


def _assert_canonical_scan_properties(node: PlanNode) -> None:
    """Verifies that no optimization (pushdown or pruning) occurred on any Scan node."""
    if isinstance(node, Scan):
        assert node.columns is None, f"Scan.columns must be None, got: {node.columns}"
        assert (
            node.pushed_predicate is None
        ), f"Scan.pushed_predicate must be None, got: {node.pushed_predicate}"
        assert node.table_schema is not None, "Scan.table_schema must be populated"

    for child in node.children:
        _assert_canonical_scan_properties(child)


def test_simple_select(catalog: Catalog) -> None:
    sql = "SELECT col FROM table"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Scan)
    assert plan.exprs == [(ColumnRef(table="table", name="col"), "col")]
    assert isinstance(plan.exprs, list)

    expected = "Project[table.col AS col]\n  Scan[table]"
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_select_aliases(catalog: Catalog) -> None:
    sql = "SELECT col AS alias, col2 FROM table"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Scan)
    assert plan.exprs == [
        (ColumnRef(table="table", name="col"), "alias"),
        (ColumnRef(table="table", name="col2"), "col2"),
    ]
    assert isinstance(plan.exprs, list)

    expected = "Project[table.col AS alias, table.col2 AS col2]\n  Scan[table]"
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_select_wildcard(catalog: Catalog) -> None:
    sql = "SELECT * FROM table"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Scan)
    assert plan.exprs == [
        (ColumnRef(table="table", name="col"), "col"),
        (ColumnRef(table="table", name="col2"), "col2"),
        (ColumnRef(table="table", name="a"), "a"),
        (ColumnRef(table="table", name="b"), "b"),
    ]

    expected = "Project[table.col AS col, table.col2 AS col2, table.a AS a, table.b AS b]\n  Scan[table]"
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_select_where(catalog: Catalog) -> None:
    sql = "SELECT col FROM table WHERE col > 10"
    plan = parse_and_bind(sql, catalog)

    # Verify Filter sits directly above Scan, and Project sits directly above Filter
    assert isinstance(plan, Project)
    assert isinstance(plan.child, Filter)
    assert isinstance(plan.child.child, Scan)

    pred = plan.child.predicate
    assert isinstance(pred, BinaryOp)
    assert pred.op == ">"
    assert pred.left == ColumnRef(table="table", name="col")
    assert pred.right == Literal(value=10, dtype=DType.INT)

    expected = (
        "Project[table.col AS col]\n"
        "  Filter[table.col > 10]\n"
        "    Scan[table]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_arithmetic_expressions(catalog: Catalog) -> None:
    sql = "SELECT a + b * 2 AS res FROM table"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Scan)
    assert len(plan.exprs) == 1

    expr, alias = plan.exprs[0]
    assert alias == "res"
    assert isinstance(expr, BinaryOp)
    assert expr.op == "+"
    assert expr.left == ColumnRef(table="table", name="a")
    assert isinstance(expr.right, BinaryOp)
    assert expr.right.op == "*"
    assert expr.right.left == ColumnRef(table="table", name="b")
    assert expr.right.right == Literal(value=2, dtype=DType.INT)

    expected = "Project[table.a + table.b * 2 AS res]\n  Scan[table]"
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_group_by_and_aggregates(catalog: Catalog) -> None:
    sql = "SELECT region, SUM(amount), COUNT(*) FROM sales GROUP BY region"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Aggregate)
    assert isinstance(plan.child.child, Scan)

    agg_node = plan.child
    assert isinstance(agg_node.group_keys, list)
    assert agg_node.group_keys == [ColumnRef(table="sales", name="region")]

    assert isinstance(agg_node.aggs, list)
    assert len(agg_node.aggs) == 2
    assert agg_node.aggs[0] == (
        AggCall(func="sum", arg=ColumnRef(table="sales", name="amount")),
        "sum(amount)",
    )
    assert agg_node.aggs[1] == (
        AggCall(func="count", arg=None),
        "count(*)",
    )

    expected = (
        "Project[sales.region AS region, sum(amount), count(*)]\n"
        "  Aggregate[group=sales.region, aggs=sum(sales.amount) AS sum(amount), count(*) AS count(*)]\n"
        "    Scan[sales]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_having_clause(catalog: Catalog) -> None:
    sql = "SELECT region, SUM(amount) FROM sales GROUP BY region HAVING SUM(amount) > 100"
    plan = parse_and_bind(sql, catalog)

    # Verify canonical hierarchy: Filter sits directly above Aggregate
    assert isinstance(plan, Project)
    assert isinstance(plan.child, Filter)
    assert isinstance(plan.child.child, Aggregate)
    assert isinstance(plan.child.child.child, Scan)

    having_filter = plan.child
    pred = having_filter.predicate
    assert isinstance(pred, BinaryOp)
    assert pred.op == ">"
    assert pred.left == AggCall(func="sum", arg=ColumnRef(table="sales", name="amount"))
    assert pred.right == Literal(value=100, dtype=DType.INT)

    expected = (
        "Project[sales.region AS region, sum(amount)]\n"
        "  Filter[sum(sales.amount) > 100]\n"
        "    Aggregate[group=sales.region, aggs=sum(sales.amount) AS sum(amount)]\n"
        "      Scan[sales]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_order_by(catalog: Catalog) -> None:
    sql = "SELECT id, amount FROM sales ORDER BY amount DESC, id ASC"
    plan = parse_and_bind(sql, catalog)

    # Sort sits above Project
    assert isinstance(plan, Sort)
    assert isinstance(plan.child, Project)
    assert isinstance(plan.child.child, Scan)

    assert isinstance(plan.keys, list)
    assert plan.keys == [
        (ColumnRef(table="sales", name="amount"), True),
        (ColumnRef(table="sales", name="id"), False),
    ]

    expected = (
        "Sort[keys=sales.amount DESC, sales.id ASC]\n"
        "  Project[sales.id AS id, sales.amount AS amount]\n"
        "    Scan[sales]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_limit(catalog: Catalog) -> None:
    sql = "SELECT id FROM sales LIMIT 5"
    plan = parse_and_bind(sql, catalog)

    # Limit sits above Project
    assert isinstance(plan, Limit)
    assert isinstance(plan.child, Project)
    assert isinstance(plan.child.child, Scan)
    assert plan.n == 5

    expected = (
        "Limit[n=5]\n"
        "  Project[sales.id AS id]\n"
        "    Scan[sales]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_inner_join(catalog: Catalog) -> None:
    sql = "SELECT o.id, c.name FROM orders o INNER JOIN customer c ON o.cust_id = c.id"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Join)
    join_node = plan.child
    assert join_node.kind == "inner"
    assert isinstance(join_node.left, Scan)
    assert isinstance(join_node.right, Scan)
    assert join_node.left.table == "orders"
    assert join_node.right.table == "customer"

    cond = join_node.condition
    assert isinstance(cond, BinaryOp)
    assert cond.op == "="
    assert cond.left == ColumnRef(table="orders", name="cust_id")
    assert cond.right == ColumnRef(table="customer", name="id")

    expected = (
        "Project[orders.id AS id, customer.name AS name]\n"
        "  Join[kind=inner, cond=orders.cust_id = customer.id]\n"
        "    Scan[orders]\n"
        "    Scan[customer]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_left_join(catalog: Catalog) -> None:
    sql = "SELECT o.id, c.name FROM orders o LEFT JOIN customer c ON o.cust_id = c.id"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Join)
    assert plan.child.kind == "left"

    expected = (
        "Project[orders.id AS id, customer.name AS name]\n"
        "  Join[kind=left, cond=orders.cust_id = customer.id]\n"
        "    Scan[orders]\n"
        "    Scan[customer]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_multi_table_unqualified_resolution(catalog: Catalog) -> None:
    # cust_id belongs uniquely to orders (aliased o), and name belongs uniquely to customer (aliased c)
    sql = "SELECT cust_id, name FROM orders o INNER JOIN customer c ON o.cust_id = c.id"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert plan.exprs == [
        (ColumnRef(table="orders", name="cust_id"), "cust_id"),
        (ColumnRef(table="customer", name="name"), "name"),
    ]

    expected = (
        "Project[orders.cust_id AS cust_id, customer.name AS name]\n"
        "  Join[kind=inner, cond=orders.cust_id = customer.id]\n"
        "    Scan[orders]\n"
        "    Scan[customer]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_is_null_predicate(catalog: Catalog) -> None:
    sql = "SELECT id FROM sales WHERE amount IS NULL"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Filter)
    pred = plan.child.predicate
    assert isinstance(pred, UnaryOp)
    assert pred.op == "IS NULL"
    assert pred.operand == ColumnRef(table="sales", name="amount")

    expected = (
        "Project[sales.id AS id]\n"
        "  Filter[sales.amount IS NULL]\n"
        "    Scan[sales]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_is_not_null_predicate(catalog: Catalog) -> None:
    sql = "SELECT id FROM sales WHERE amount IS NOT NULL"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Filter)
    pred = plan.child.predicate
    assert isinstance(pred, UnaryOp)
    assert pred.op == "IS NOT NULL"
    assert pred.operand == ColumnRef(table="sales", name="amount")

    expected = (
        "Project[sales.id AS id]\n"
        "  Filter[sales.amount IS NOT NULL]\n"
        "    Scan[sales]"
    )
    assert format_plan(plan) == expected
    _assert_canonical_scan_properties(plan)


def test_canonical_full_pipeline_hierarchy(catalog: Catalog) -> None:
    """Tests the full stack: Limit -> Sort -> Project -> Filter(HAVING) -> Aggregate -> Filter(WHERE) -> Join -> Scan."""
    sql = (
        "SELECT c.name, SUM(o.total) AS spent "
        "FROM orders o INNER JOIN customer c ON o.cust_id = c.id "
        "WHERE o.total > 5 "
        "GROUP BY c.name "
        "HAVING SUM(o.total) > 0 "
        "ORDER BY spent DESC "
        "LIMIT 1"
    )
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Limit)
    assert plan.n == 1

    assert isinstance(plan.child, Sort)
    assert plan.child.keys == [(ColumnRef(table=None, name="spent"), True)]

    assert isinstance(plan.child.child, Project)
    assert plan.child.child.exprs == [
        (ColumnRef(table="customer", name="name"), "name"),
        (ColumnRef(table=None, name="spent"), "spent"),
    ]

    assert isinstance(plan.child.child.child, Filter)  # HAVING Filter
    having_pred = plan.child.child.child.predicate
    assert isinstance(having_pred, BinaryOp)
    assert having_pred.left == AggCall(func="sum", arg=ColumnRef(table="orders", name="total"))

    assert isinstance(plan.child.child.child.child, Aggregate)
    agg_node = plan.child.child.child.child
    assert agg_node.group_keys == [ColumnRef(table="customer", name="name")]
    assert agg_node.aggs == [
        (AggCall(func="sum", arg=ColumnRef(table="orders", name="total")), "spent")
    ]

    assert isinstance(agg_node.child, Filter)  # WHERE Filter
    where_pred = agg_node.child.predicate
    assert isinstance(where_pred, BinaryOp)
    assert where_pred.left == ColumnRef(table="orders", name="total")

    assert isinstance(agg_node.child.child, Join)
    join_node = agg_node.child.child
    assert isinstance(join_node.left, Scan)
    assert isinstance(join_node.right, Scan)

    _assert_canonical_scan_properties(plan)


# ---------------------------------------------------------------------------
# Error Handling Assertions
# ---------------------------------------------------------------------------


def test_missing_table_raises_value_error(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="not found in catalog"):
        parse_and_bind("SELECT id FROM missing_table", catalog)

    with pytest.raises(ValueError, match="not found in catalog"):
        parse_and_bind("SELECT o.id FROM orders o JOIN missing_t m ON o.id = m.id", catalog)


def test_unknown_column_raises_value_error(catalog: Catalog) -> None:
    # Unqualified unknown column
    with pytest.raises(ValueError, match="Unknown column 'nonexistent'"):
        parse_and_bind("SELECT nonexistent FROM sales", catalog)

    # Qualified unknown column
    with pytest.raises(ValueError, match="Column 'nonexistent' not found in table 'sales'"):
        parse_and_bind("SELECT sales.nonexistent FROM sales", catalog)

    # Qualified table not in scope
    with pytest.raises(ValueError, match="Table 'other' not found in active scope"):
        parse_and_bind("SELECT other.id FROM sales", catalog)


def test_ambiguous_unqualified_column_raises_value_error(catalog: Catalog) -> None:
    # Both orders and customer have an 'id' column
    with pytest.raises(ValueError, match="Ambiguous column reference 'id'"):
        parse_and_bind(
            "SELECT id FROM orders o INNER JOIN customer c ON o.cust_id = c.id", catalog
        )


def test_non_select_query_raises_value_error(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="Expected a SELECT query"):
        parse_and_bind("INSERT INTO sales VALUES (1)", catalog)


def test_syntax_error_raises_value_error(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="SQL parsing error"):
        parse_and_bind("SELECT FROM", catalog)


def test_missing_from_clause_raises_value_error(catalog: Catalog) -> None:
    with pytest.raises(ValueError, match="Query must specify a FROM clause"):
        parse_and_bind("SELECT 1", catalog)


def test_order_by_aggregate_and_alias_binding(catalog: Catalog) -> None:
    """Verifies that ORDER BY binds successfully on an aggregate expression and an alias."""
    sql_agg = "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY SUM(amount) DESC"
    plan_agg = parse_and_bind(sql_agg, catalog)
    assert isinstance(plan_agg, Sort)
    assert isinstance(plan_agg.child, Project)
    assert isinstance(plan_agg.child.child, Aggregate)
    assert (AggCall(func="sum", arg=ColumnRef(table="sales", name="amount")), "total") in plan_agg.child.child.aggs
    assert plan_agg.keys == [(ColumnRef(table=None, name="total"), True)]

    sql_alias = "SELECT region, SUM(amount) AS total FROM sales GROUP BY region ORDER BY total DESC"
    plan_alias = parse_and_bind(sql_alias, catalog)
    assert isinstance(plan_alias, Sort)
    assert isinstance(plan_alias.child, Project)
    assert isinstance(plan_alias.child.child, Aggregate)
    assert (AggCall(func="sum", arg=ColumnRef(table="sales", name="amount")), "total") in plan_alias.child.child.aggs
    assert plan_alias.keys == [(ColumnRef(table=None, name="total"), True)]

