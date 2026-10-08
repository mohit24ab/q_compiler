from __future__ import annotations

import datetime
import pyarrow as pa
import pytest

from catalog.catalog import Catalog
from frontend.binder import parse_and_bind
from frontend.resolver import Resolver, SemanticError
from frontend.typecheck import TypeChecker, SemanticTypeError
from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Project, Scan, Sort


@pytest.fixture
def catalog() -> Catalog:
    cat = Catalog()

    sales_table = pa.table({
        "id": [1, 2, 3],
        "amount": [10.0, 20.5, 30.0],
        "region": ["US", "EU", "US"],
        "qty": [1, 2, 3],
        "sale_date": pa.array(
            [datetime.date(2026, 1, 1), datetime.date(2026, 1, 2), datetime.date(2026, 1, 3)],
            type=pa.date32(),
        ),
    })
    cat.register_table("sales", sales_table)

    orders_table = pa.table({
        "id": [100, 101, 102],
        "cust_id": [1, 2, 1],
        "total": [50.0, 70.5, 90.0],
        "order_date": pa.array(
            [datetime.date(2026, 2, 1), datetime.date(2026, 2, 2), datetime.date(2026, 2, 3)],
            type=pa.date32(),
        ),
    })
    cat.register_table("orders", orders_table)

    customer_table = pa.table({
        "id": [1, 2, 3],
        "name": ["ann", "bob", "charlie"],
        "c_segment": ["AUTO", "BUILDING", "AUTO"],
    })
    cat.register_table("customer", customer_table)

    table_table = pa.table({
        "a": [1, 2],
        "b": [3, 4],
        "rate": [1.5, 2.5],
    })
    cat.register_table("table", table_table)

    return cat


# ===========================================================================
# 8 VALID COMPLEX SEMANTIC QUERIES
# ===========================================================================


def test_valid_arithmetic_type_coercion(catalog: Catalog) -> None:
    """1. Arithmetic expressions with type coercion (INT + FLOAT -> FLOAT, division -> FLOAT)."""
    sql = "SELECT qty + amount AS total_val, qty * 2.5 AS scaled_qty, amount / 2 AS half_amt FROM sales"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Scan)

    tc = TypeChecker()
    child_schema = plan.child.schema()

    # Verify type inference on expressions
    total_val_expr = plan.exprs[0][0]
    scaled_qty_expr = plan.exprs[1][0]
    half_amt_expr = plan.exprs[2][0]

    assert tc.infer_type(total_val_expr, child_schema) == DType.FLOAT
    assert tc.infer_type(scaled_qty_expr, child_schema) == DType.FLOAT
    assert tc.infer_type(half_amt_expr, child_schema) == DType.FLOAT

    # Verify output schema
    assert plan.schema() == [
        ("total_val", DType.FLOAT),
        ("scaled_qty", DType.FLOAT),
        ("half_amt", DType.FLOAT),
    ]


def test_valid_multi_table_joins_with_aliases(catalog: Catalog) -> None:
    """2. Multi-table joins with qualified column references and aliases."""
    sql = (
        "SELECT o.id, c.name, o.total, s.region "
        "FROM orders o "
        "INNER JOIN customer c ON o.cust_id = c.id "
        "INNER JOIN sales s ON o.cust_id = s.id"
    )
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Join)
    assert isinstance(plan.child.left, Join)

    assert plan.schema() == [
        ("id", DType.INT),
        ("name", DType.STRING),
        ("total", DType.FLOAT),
        ("region", DType.STRING),
    ]
    assert plan.exprs == [
        (ColumnRef(table="orders", name="id"), "id"),
        (ColumnRef(table="customer", name="name"), "name"),
        (ColumnRef(table="orders", name="total"), "total"),
        (ColumnRef(table="sales", name="region"), "region"),
    ]


def test_valid_group_by_with_aggregates(catalog: Catalog) -> None:
    """3. Proper GROUP BY with valid aggregated and grouped columns."""
    sql = (
        "SELECT region, SUM(amount) AS total_amt, AVG(qty) AS avg_qty, COUNT(*) AS count_sales "
        "FROM sales GROUP BY region"
    )
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Aggregate)
    agg_node = plan.child

    assert agg_node.group_keys == [ColumnRef(table="sales", name="region")]
    assert len(agg_node.aggs) == 3
    assert plan.schema() == [
        ("region", DType.STRING),
        ("total_amt", DType.FLOAT),
        ("avg_qty", DType.FLOAT),
        ("count_sales", DType.INT),
    ]


def test_valid_having_filtering_on_aggregates(catalog: Catalog) -> None:
    """4. HAVING filtering on aggregate expressions."""
    sql = (
        "SELECT region, SUM(amount) AS total_amt "
        "FROM sales "
        "GROUP BY region "
        "HAVING SUM(amount) > 25.0 AND COUNT(*) >= 1"
    )
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Filter)  # HAVING Filter node above Aggregate
    assert isinstance(plan.child.child, Aggregate)

    having_filter = plan.child
    assert isinstance(having_filter.predicate, BinaryOp)
    assert having_filter.predicate.op == "AND"
    assert plan.schema() == [
        ("region", DType.STRING),
        ("total_amt", DType.FLOAT),
    ]


def test_valid_date_and_string_comparisons(catalog: Catalog) -> None:
    """5. Comparisons between compatible dates and strings."""
    sql = "SELECT id, sale_date FROM sales WHERE sale_date >= '2026-01-01' AND '2026-12-31' > sale_date"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Filter)
    assert plan.schema() == [
        ("id", DType.INT),
        ("sale_date", DType.DATE),
    ]

    tc = TypeChecker()
    # Check that date vs string comparison is accepted and yields BOOL
    assert tc.infer_type(plan.child.predicate, plan.child.child.schema()) == DType.BOOL


def test_valid_nested_boolean_logic(catalog: Catalog) -> None:
    """6. Nested boolean logic with AND/OR/NOT."""
    sql = "SELECT id FROM sales WHERE NOT (region = 'US' AND qty <= 2) OR (amount > 15.0 AND NOT (qty IS NULL))"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Filter)
    assert plan.schema() == [("id", DType.INT)]

    tc = TypeChecker()
    assert tc.infer_type(plan.child.predicate, plan.child.child.schema()) == DType.BOOL


def test_valid_order_by_grouped_projection_columns(catalog: Catalog) -> None:
    """7. ORDER BY referencing grouped projection columns and aggregate aliases."""
    sql = (
        "SELECT region, SUM(amount) AS total_sales "
        "FROM sales "
        "GROUP BY region "
        "ORDER BY region ASC, total_sales DESC"
    )
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Sort)
    assert isinstance(plan.child, Project)
    assert plan.keys == [
        (ColumnRef(table="sales", name="region"), False),
        (ColumnRef(table=None, name="total_sales"), True),
    ]
    assert plan.schema() == [
        ("region", DType.STRING),
        ("total_sales", DType.FLOAT),
    ]


def test_valid_unqualified_columns_resolved_across_joins(catalog: Catalog) -> None:
    """8. Unqualified columns cleanly resolved across joins."""
    sql = "SELECT cust_id, name, total FROM orders o INNER JOIN customer c ON o.cust_id = c.id"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert plan.exprs == [
        (ColumnRef(table="orders", name="cust_id"), "cust_id"),
        (ColumnRef(table="customer", name="name"), "name"),
        (ColumnRef(table="orders", name="total"), "total"),
    ]
    assert plan.schema() == [
        ("cust_id", DType.INT),
        ("name", DType.STRING),
        ("total", DType.FLOAT),
    ]


# ===========================================================================
# 8 INVALID QUERIES ASSERTING SEMANTIC EXCEPTIONS
# ===========================================================================


def test_invalid_arithmetic_on_incompatible_types(catalog: Catalog) -> None:
    """1. Arithmetic on incompatible types (e.g., string + int)."""
    with pytest.raises(SemanticTypeError, match=r"operator '\+' requires numeric operands"):
        parse_and_bind("SELECT region + 10 FROM sales", catalog)

    with pytest.raises(SemanticTypeError, match=r"operator '\*' requires numeric operands"):
        parse_and_bind("SELECT name * 2 FROM customer", catalog)


def test_invalid_aggregate_inside_where(catalog: Catalog) -> None:
    """2. Aggregate function (SUM or COUNT) embedded inside a WHERE clause."""
    with pytest.raises(SemanticError, match=r"Aggregate function SUM\(\) is not allowed in WHERE clause"):
        parse_and_bind("SELECT id FROM sales WHERE SUM(amount) > 10", catalog)

    with pytest.raises(SemanticError, match=r"Aggregate function COUNT\(\) is not allowed in WHERE clause"):
        parse_and_bind("SELECT id FROM sales WHERE COUNT(*) > 0", catalog)


def test_invalid_non_grouped_unaggregated_column_in_select(catalog: Catalog) -> None:
    """3. Non-grouped, unaggregated column in SELECT during a GROUP BY query."""
    with pytest.raises(
        SemanticError,
        match=r"Column 'amount' must appear in the GROUP BY clause or be used in an aggregate function",
    ):
        parse_and_bind("SELECT region, amount FROM sales GROUP BY region", catalog)


def test_invalid_non_grouped_column_in_order_by(catalog: Catalog) -> None:
    """4. Non-grouped column in ORDER BY during a GROUP BY query."""
    with pytest.raises(
        SemanticError,
        match=r"Column 'amount' in ORDER BY must appear in the GROUP BY clause or be used in an aggregate function",
    ):
        parse_and_bind(
            "SELECT region, SUM(amount) FROM sales GROUP BY region ORDER BY amount", catalog
        )

    with pytest.raises(
        SemanticError,
        match=r"Column 'qty' in ORDER BY must appear in the GROUP BY clause or be used in an aggregate function",
    ):
        parse_and_bind(
            "SELECT region, SUM(amount) FROM sales GROUP BY region ORDER BY qty", catalog
        )


def test_invalid_unknown_table_reference(catalog: Catalog) -> None:
    """5. Unknown table reference."""
    with pytest.raises(SemanticError, match=r"Table 'non_existent_table' not found in catalog"):
        parse_and_bind("SELECT id FROM non_existent_table", catalog)

    with pytest.raises(SemanticError, match=r"Table 'missing_join_table' not found in catalog"):
        parse_and_bind("SELECT s.id FROM sales s JOIN missing_join_table m ON s.id = m.id", catalog)


def test_invalid_unknown_column_reference(catalog: Catalog) -> None:
    """6. Unknown column reference."""
    with pytest.raises(SemanticError, match=r"Unknown column 'unknown_col'"):
        parse_and_bind("SELECT unknown_col FROM sales", catalog)

    with pytest.raises(SemanticError, match=r"Column 'bad_col' not found in table 'sales'"):
        parse_and_bind("SELECT sales.bad_col FROM sales", catalog)


def test_invalid_ambiguous_unqualified_column(catalog: Catalog) -> None:
    """7. Ambiguous unqualified column reference across joined tables."""
    # Both orders and customer have an 'id' column
    with pytest.raises(
        SemanticError,
        match=r"Ambiguous column reference 'id' found across active tables",
    ):
        parse_and_bind("SELECT id FROM orders o INNER JOIN customer c ON o.cust_id = c.id", catalog)


def test_invalid_non_boolean_condition_in_where_or_join(catalog: Catalog) -> None:
    """8. Non-boolean condition in WHERE or JOIN ... ON."""
    with pytest.raises(SemanticTypeError, match=r"WHERE condition must evaluate to BOOL"):
        parse_and_bind("SELECT id FROM sales WHERE amount", catalog)

    with pytest.raises(SemanticTypeError, match=r"WHERE condition must evaluate to BOOL"):
        parse_and_bind("SELECT id FROM sales WHERE region", catalog)

    with pytest.raises(SemanticTypeError, match=r"JOIN ON condition must evaluate to BOOL"):
        parse_and_bind("SELECT o.id FROM orders o JOIN customer c ON o.cust_id", catalog)


# ===========================================================================
# COMPREHENSIVE UNIT TESTS FOR RESOLVER AND TYPECHECKER STANDALONE
# ===========================================================================


def test_resolver_standalone(catalog: Catalog) -> None:
    """Directly verifies Resolver scoping, alias resolution, and errors."""
    resolver = Resolver(catalog)
    resolver.add_table("sales", alias="s")

    # Qualified via alias
    col1 = resolver.resolve_column("amount", table="s")
    assert col1 == ColumnRef(table="sales", name="amount")

    # Qualified via original table name
    col2 = resolver.resolve_column("amount", table="sales")
    assert col2 == ColumnRef(table="sales", name="amount")

    # Unqualified
    col3 = resolver.resolve_column("amount")
    assert col3 == ColumnRef(table="sales", name="amount")

    # Unknown table
    with pytest.raises(SemanticError, match=r"Table 'unknown' not found in active scope"):
        resolver.resolve_column("amount", table="unknown")

    # Unknown column
    with pytest.raises(SemanticError, match=r"Column 'bad' not found in table 's'"):
        resolver.resolve_column("bad", table="s")


def test_typechecker_standalone_infer_type() -> None:
    """Directly verifies TypeChecker type inference, coercion, and validation."""
    tc = TypeChecker()

    # Division returns FLOAT
    div_expr = BinaryOp(op="/", left=Literal(10, DType.INT), right=Literal(2, DType.INT))
    assert tc.infer_type(div_expr) == DType.FLOAT

    # Int + Float coercion
    add_expr = BinaryOp(op="+", left=Literal(10, DType.INT), right=Literal(2.5, DType.FLOAT))
    assert tc.infer_type(add_expr) == DType.FLOAT

    # Date = String comparison
    date_cmp = BinaryOp(
        op="=",
        left=Literal("2026-01-01", DType.DATE),
        right=Literal("2026-01-01", DType.STRING),
    )
    assert tc.infer_type(date_cmp) == DType.BOOL

    # Incompatible comparison
    bad_cmp = BinaryOp(
        op="=",
        left=Literal(10, DType.INT),
        right=Literal("text", DType.STRING),
    )
    with pytest.raises(SemanticTypeError, match=r"cannot compare incompatible types"):
        tc.infer_type(bad_cmp)
