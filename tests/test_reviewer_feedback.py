from __future__ import annotations

import pyarrow as pa
import pytest

from catalog.catalog import Catalog
from codegen import compile_and_run, generate
from frontend.binder import parse_and_bind
from frontend.resolver import SemanticError
from ir.dtype import DType
from ir.expr import BinaryOp, ColumnRef, Literal
from ir.nodes import Aggregate, Join, Limit, Project, Scan, Sort
from runtime.interpreter import interpret


@pytest.fixture
def catalog() -> Catalog:
    cat = Catalog()

    sales_table = pa.table({
        "id": [1, 2, 3],
        "amount": [10.0, 30.0, 20.0],
        "region": ["US", "EU", "US"],
        "qty": [1, 2, 3],
    })
    cat.register_table("sales", sales_table)

    orders_table = pa.table({
        "id": [1, 2, 4],
        "cust_id": [1, 2, 1],
        "total": [50.0, 70.0, 90.0],
    })
    cat.register_table("orders", orders_table)

    dup_table = pa.table({
        "x": [1, 1, 2, 2, 3],
        "y": ["a", "a", "b", "c", "a"],
    })
    cat.register_table("dups", dup_table)

    return cat


# ==============================================================================
# CATEGORY 1: FIX SQL THAT RETURNS WRONG ROWS WITH NO ERROR
# ==============================================================================


def test_select_distinct_lowering_and_execution(catalog: Catalog) -> None:
    """1. SELECT DISTINCT lowers to an Aggregate node and eliminates duplicates."""
    sql = "SELECT DISTINCT x, y FROM dups"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Aggregate)
    assert plan.aggs == []
    assert plan.group_keys == [ColumnRef(table=None, name="x"), ColumnRef(table=None, name="y")]
    assert isinstance(plan.child, Project)

    tables = {"dups": catalog._tables["dups"]}
    rel = interpret(plan, tables)
    rows = rel.to_rows()
    assert len(rows) == 4
    assert sorted(rows) == [(1, "a"), (2, "b"), (2, "c"), (3, "a")]

    # Also test code generator
    code = generate(plan, catalog)
    cg_res = compile_and_run(code, tables)
    assert sorted(cg_res.to_rows()) == [(1, "a"), (2, "b"), (2, "c"), (3, "a")]


def test_positional_order_by(catalog: Catalog) -> None:
    """2. Positional ORDER BY (e.g. ORDER BY 2) resolves to n-th projected item."""
    # ORDER BY 2 resolves to amount DESC
    sql1 = "SELECT id, amount FROM sales ORDER BY 2 DESC"
    plan1 = parse_and_bind(sql1, catalog)
    assert isinstance(plan1, Sort)
    assert plan1.keys == [(ColumnRef(table="sales", name="amount"), True)]

    # ORDER BY 2 with alias resolves to alias
    sql2 = "SELECT id, amount AS spent FROM sales ORDER BY 2 DESC"
    plan2 = parse_and_bind(sql2, catalog)
    assert isinstance(plan2, Sort)
    assert plan2.keys == [(ColumnRef(table=None, name="spent"), True)]

    # Out of bounds position raises SemanticError
    with pytest.raises(SemanticError, match=r"ORDER BY position 5 is out of range"):
        parse_and_bind("SELECT id, amount FROM sales ORDER BY 5", catalog)

    with pytest.raises(SemanticError, match=r"ORDER BY position 0 is out of range"):
        parse_and_bind("SELECT id, amount FROM sales ORDER BY 0", catalog)


def test_unsupported_joins_raise_cleanly(catalog: Catalog) -> None:
    """3a. RIGHT, FULL, SEMI, ANTI joins raise explicit NotImplementedError."""
    with pytest.raises(NotImplementedError, match=r"RIGHT JOIN is not supported"):
        parse_and_bind("SELECT * FROM sales RIGHT JOIN orders ON sales.id = orders.cust_id", catalog)

    with pytest.raises(NotImplementedError, match=r"FULL JOIN is not supported"):
        parse_and_bind("SELECT * FROM sales FULL JOIN orders ON sales.id = orders.cust_id", catalog)

    with pytest.raises(NotImplementedError, match=r"SEMI JOIN is not supported"):
        parse_and_bind("SELECT * FROM sales LEFT SEMI JOIN orders ON sales.id = orders.cust_id", catalog)

    with pytest.raises(NotImplementedError, match=r"ANTI JOIN is not supported"):
        parse_and_bind("SELECT * FROM sales LEFT ANTI JOIN orders ON sales.id = orders.cust_id", catalog)


def test_join_using_constructs_equi_join(catalog: Catalog) -> None:
    """3b. USING (c1, ...) constructs explicit equi-join predicate."""
    sql = "SELECT s.region, o.total FROM sales s INNER JOIN orders o USING (id)"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Join)
    join_node = plan.child
    assert isinstance(join_node.condition, BinaryOp)
    assert join_node.condition.op == "="
    assert join_node.condition.left == ColumnRef(table="sales", name="id")
    assert join_node.condition.right == ColumnRef(table="orders", name="id")

    tables = {"sales": catalog._tables["sales"], "orders": catalog._tables["orders"]}
    rows = interpret(plan, tables).to_rows()
    assert len(rows) == 2
    assert rows == [("US", 50.0), ("EU", 70.0)]


def test_natural_join_constructs_equi_join(catalog: Catalog) -> None:
    """3c. NATURAL JOIN finds common columns and builds equi-join condition."""
    sql = "SELECT sales.region, orders.total FROM sales NATURAL JOIN orders"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Join)
    join_node = plan.child
    assert isinstance(join_node.condition, BinaryOp)
    assert join_node.condition.op == "="
    assert join_node.condition.left == ColumnRef(table="sales", name="id")
    assert join_node.condition.right == ColumnRef(table="orders", name="id")


def test_offset_and_nulls_ordering_raise_error(catalog: Catalog) -> None:
    """4. OFFSET and NULLS FIRST / NULLS LAST raise explicit validation errors."""
    with pytest.raises(SemanticError, match=r"OFFSET is not supported"):
        parse_and_bind("SELECT id FROM sales LIMIT 2 OFFSET 1", catalog)

    with pytest.raises(SemanticError, match=r"NULLS FIRST / NULLS LAST ordering is not supported"):
        parse_and_bind("SELECT id FROM sales ORDER BY amount ASC NULLS FIRST", catalog)

    with pytest.raises(SemanticError, match=r"NULLS FIRST / NULLS LAST ordering is not supported"):
        parse_and_bind("SELECT id FROM sales ORDER BY amount DESC NULLS LAST", catalog)


def test_wildcard_expansion_with_extra_expressions(catalog: Catalog) -> None:
    """5. SELECT *, expr preserves all expanded wildcard columns AND appends extra exprs."""
    sql = "SELECT *, amount * 2.0 AS double_amt FROM sales"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    col_names = [alias for _, alias in plan.exprs]
    assert col_names == ["id", "amount", "region", "qty", "double_amt"]

    tables = {"sales": catalog._tables["sales"]}
    rows = interpret(plan, tables).to_rows()
    assert rows[0] == (1, 10.0, "US", 1, 20.0)


def test_self_joins_with_isolated_alias_scopes(catalog: Catalog) -> None:
    """6. Self-joins with aliases bind strictly to isolated scopes without collapsing to underlying table."""
    sql = "SELECT s1.id, s2.id, s1.region FROM sales s1 INNER JOIN sales s2 ON s1.id = s2.id"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    assert isinstance(plan.child, Join)
    join_node = plan.child

    # Condition must preserve s1 and s2 aliases
    assert join_node.condition == BinaryOp(
        op="=",
        left=ColumnRef(table="s1", name="id"),
        right=ColumnRef(table="s2", name="id"),
    )

    # Projections must bind to s1 and s2 respectively
    assert plan.exprs[0][0] == ColumnRef(table="s1", name="id")
    assert plan.exprs[1][0] == ColumnRef(table="s2", name="id")
    assert plan.exprs[2][0] == ColumnRef(table="s1", name="region")

    # Ambiguous underlying table name reference raises SemanticError
    with pytest.raises(SemanticError, match=r"Ambiguous table reference 'sales'"):
        parse_and_bind("SELECT sales.id FROM sales s1 INNER JOIN sales s2 ON s1.id = s2.id", catalog)

    # Ambiguous unqualified column reference raises SemanticError
    with pytest.raises(SemanticError, match=r"Ambiguous column reference 'id'"):
        parse_and_bind("SELECT id FROM sales s1 INNER JOIN sales s2 ON s1.id = s2.id", catalog)


# ==============================================================================
# CATEGORY 2: FIX VALID SQL THAT CRASHES
# ==============================================================================


def test_order_by_column_not_in_select_list(catalog: Catalog) -> None:
    """1. ORDER BY column not in SELECT list sorts correctly and projects only requested columns."""
    sql = "SELECT region FROM sales ORDER BY amount DESC"
    plan = parse_and_bind(sql, catalog)

    # Top plan should project only region
    assert [col_name for col_name, _ in plan.schema()] == ["region"]

    tables = {"sales": catalog._tables["sales"]}
    rows = interpret(plan, tables).to_rows()
    # sales amounts: id=2: 30.0, id=3: 20.0, id=1: 10.0
    # regions in descending amount order: EU (30.0), US (20.0), US (10.0)
    assert rows == [("EU",), ("US",), ("US",)]

    code = generate(plan, catalog)
    cg_res = compile_and_run(code, tables)
    assert cg_res.to_rows() == [("EU",), ("US",), ("US",)]


def test_date_literals_and_typed_constructors(catalog: Catalog) -> None:
    """2. DATE literals and typed string constructors map to Literal(dtype=DType.DATE)."""
    sql1 = "SELECT DATE '2024-01-01' AS d1 FROM sales"
    plan1 = parse_and_bind(sql1, catalog)
    assert isinstance(plan1, Project)
    assert plan1.exprs[0][0] == Literal(value="2024-01-01", dtype=DType.DATE)

    sql2 = "SELECT CAST('2024-12-31' AS DATE) AS d2 FROM sales"
    plan2 = parse_and_bind(sql2, catalog)
    assert isinstance(plan2, Project)
    assert plan2.exprs[0][0] == Literal(value="2024-12-31", dtype=DType.DATE)

    sql3 = "SELECT DATE('2025-05-15') AS d3 FROM sales"
    plan3 = parse_and_bind(sql3, catalog)
    assert isinstance(plan3, Project)
    assert plan3.exprs[0][0] == Literal(value="2025-05-15", dtype=DType.DATE)


def test_name_collision_deduplication(catalog: Catalog) -> None:
    """3. Unaliased projections with name collisions are deduplicated."""
    sql = "SELECT MAX(id), MAX(id) FROM sales"
    plan = parse_and_bind(sql, catalog)

    assert isinstance(plan, Project)
    col_names = [alias for _, alias in plan.exprs]
    assert len(col_names) == 2
    assert col_names[0] == "max(id)"
    assert col_names[1] == "max(id)_1"

    sql2 = "SELECT id, id FROM sales"
    plan2 = parse_and_bind(sql2, catalog)
    col_names2 = [alias for _, alias in plan2.exprs]
    assert col_names2 == ["id", "id_1"]
