import pytest
from dataclasses import FrozenInstanceError

from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort, _infer_expr_dtype
from ir.visitor import transform_post_order
from ir.printer import format_expr, format_plan


class MockScan(Scan):
    """Scan subclass with synthetic schema for A1 schema propagation tests."""

    def __init__(self, table: str, mock_schema: list[tuple[str, DType]]):
        super().__init__(table=table, columns=None, pushed_predicate=None)
        object.__setattr__(self, "_mock_schema", mock_schema)

    def schema(self) -> list[tuple[str, DType]]:
        return getattr(self, "_mock_schema")


def test_dtype_definitions():
    assert set(DType.__members__.keys()) == {"INT", "FLOAT", "STRING", "BOOL", "DATE"}


def test_expression_node_construction():
    col = ColumnRef(table="sales", name="amount")
    assert col.table == "sales" and col.name == "amount"

    lit = Literal(value=42, dtype=DType.INT)
    assert lit.value == 42 and lit.dtype == DType.INT

    bin_op = BinaryOp(op="+", left=col, right=lit)
    assert bin_op.op == "+" and bin_op.left == col and bin_op.right == lit

    un_op = UnaryOp(op="NOT", operand=Literal(value=True, dtype=DType.BOOL))
    assert un_op.op == "NOT"

    agg_count_star = AggCall(func="count", arg=None)
    assert agg_count_star.func == "count" and agg_count_star.arg is None


def test_plan_node_children_and_schema_derivation():
    leaf = MockScan("sales", [("amount", DType.FLOAT), ("region", DType.STRING)])
    
    filt = Filter(child=leaf, predicate=BinaryOp(">", ColumnRef(None, "amount"), Literal(100.0, DType.FLOAT)))
    assert filt.children == (leaf,)
    assert filt.schema() == [("amount", DType.FLOAT), ("region", DType.STRING)]

    proj = Project(child=filt, exprs=[(ColumnRef(None, "region"), "region"), (ColumnRef(None, "amount"), "amount")])
    assert proj.children == (filt,)
    assert proj.schema() == [("region", DType.STRING), ("amount", DType.FLOAT)]

    limit = Limit(child=proj, n=10)
    assert limit.children == (proj,)
    assert limit.schema() == [("region", DType.STRING), ("amount", DType.FLOAT)]


def test_scan_schema_raises_not_implemented_in_a1():
    scan = Scan(table="sales", columns=None, pushed_predicate=None)
    assert scan.children == ()
    with pytest.raises(NotImplementedError):
        scan.schema()


def test_replace_children_immutability():
    scan_a = MockScan("t1", [("id", DType.INT)])
    scan_b = MockScan("t2", [("id", DType.INT)])
    filt = Filter(child=scan_a, predicate=Literal(True, DType.BOOL))

    new_filt = filt.replace_children((scan_b,))
    assert new_filt is not filt
    assert type(new_filt) is Filter
    assert new_filt.child == scan_b
    assert filt.child == scan_a


def test_frozen_dataclass_reassignment_raises():
    scan = Scan(table="sales", columns=None, pushed_predicate=None)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        scan.table = "other"  # type: ignore


def test_transform_post_order_traversal():
    visited = []
    scan1 = MockScan("t1", [("id", DType.INT)])
    scan2 = MockScan("t2", [("id", DType.INT)])
    join = Join(left=scan1, right=scan2, condition=Literal(True, DType.BOOL), kind="inner")

    def visitor(node):
        visited.append(type(node))
        return node

    transform_post_order(join, visitor)
    assert visited == [MockScan, MockScan, Join]


def test_contract_list_fields_remain_lists():
    scan = Scan(table="sales", columns=["id", "amount"], pushed_predicate=None)
    assert isinstance(scan.columns, list)

    proj = Project(child=scan, exprs=[(ColumnRef(None, "id"), "id")])
    assert isinstance(proj.exprs, list)

    agg = Aggregate(child=scan, group_keys=[ColumnRef(None, "id")], aggs=[(AggCall("count", None), "cnt")])
    assert isinstance(agg.group_keys, list)
    assert isinstance(agg.aggs, list)

    sort = Sort(child=scan, keys=[(ColumnRef(None, "id"), True)])
    assert isinstance(sort.keys, list)


def test_three_hand_built_plans_printer_output():
    # Plan 1: Aggregate over Filter over Scan
    scan1 = Scan(table="sales", columns=None, pushed_predicate=None)
    filt1 = Filter(
        child=scan1,
        predicate=BinaryOp(op=">=", left=ColumnRef(table=None, name="date"), right=Literal(value="2024-01-01", dtype=DType.DATE))
    )
    agg1 = Aggregate(
        child=filt1,
        group_keys=[ColumnRef(table=None, name="region")],
        aggs=[(AggCall(func="avg", arg=ColumnRef(table=None, name="amount")), "avg_amt")]
    )
    expected_1 = (
        "Aggregate[group=region, aggs=avg(amount) AS avg_amt]\n"
        "  Filter[date >= '2024-01-01']\n"
        "    Scan[sales]"
    )
    assert format_plan(agg1) == expected_1

    # Plan 2: Limit over Sort over Project over Scan
    scan2 = Scan(table="sales", columns=None, pushed_predicate=None)
    proj2 = Project(child=scan2, exprs=[(ColumnRef(None, "region"), "region"), (ColumnRef(None, "amount"), "amount")])
    sort2 = Sort(child=proj2, keys=[(ColumnRef(None, "amount"), True)])
    limit2 = Limit(child=sort2, n=10)
    expected_2 = (
        "Limit[n=10]\n"
        "  Sort[keys=amount DESC]\n"
        "    Project[region, amount]\n"
        "      Scan[sales]"
    )
    assert format_plan(limit2) == expected_2

    # Plan 3: Project over Join over 2 Scans
    scan3_l = Scan(table="orders", columns=None, pushed_predicate=None)
    scan3_r = Scan(table="customer", columns=None, pushed_predicate=None)
    join3 = Join(
        left=scan3_l,
        right=scan3_r,
        condition=BinaryOp(op="=", left=ColumnRef("orders", "cust_id"), right=ColumnRef("customer", "id")),
        kind="inner"
    )
    proj3 = Project(child=join3, exprs=[(ColumnRef("orders", "id"), "order_id"), (ColumnRef("customer", "name"), "name")])
    expected_3 = (
        "Project[orders.id AS order_id, customer.name AS name]\n"
        "  Join[kind=inner, cond=orders.cust_id = customer.id]\n"
        "    Scan[orders]\n"
        "    Scan[customer]"
    )
    assert format_plan(proj3) == expected_3


def test_unary_op_is_null_format_and_type_inference():
    col_x = ColumnRef(table=None, name="x")

    assert format_expr(UnaryOp(op="IS NULL", operand=ColumnRef(None, "x"))) == "x IS NULL"
    assert format_expr(UnaryOp(op="IS NOT NULL", operand=ColumnRef(None, "x"))) == "x IS NOT NULL"

    # Additional case-insensitive and underscore variants
    assert format_expr(UnaryOp(op="is null", operand=col_x)) == "x IS NULL"
    assert format_expr(UnaryOp(op="is not null", operand=col_x)) == "x IS NOT NULL"
    assert format_expr(UnaryOp(op="IS_NULL", operand=col_x)) == "x IS NULL"
    assert format_expr(UnaryOp(op="IS_NOT_NULL", operand=col_x)) == "x IS NOT NULL"
    assert format_expr(UnaryOp(op="is_null", operand=col_x)) == "x IS NULL"
    assert format_expr(UnaryOp(op="is_not_null", operand=col_x)) == "x IS NOT NULL"

    # Type inference tests
    child_schema = [("x", DType.INT)]
    assert _infer_expr_dtype(UnaryOp(op="IS NULL", operand=col_x), child_schema) == DType.BOOL
    assert _infer_expr_dtype(UnaryOp(op="IS NOT NULL", operand=col_x), child_schema) == DType.BOOL
    assert _infer_expr_dtype(UnaryOp(op="IS_NULL", operand=col_x), child_schema) == DType.BOOL
    assert _infer_expr_dtype(UnaryOp(op="IS_NOT_NULL", operand=col_x), child_schema) == DType.BOOL


def test_printer_parentheses_precedence():
    x = ColumnRef(table=None, name="x")
    y = ColumnRef(table=None, name="y")
    z = ColumnRef(table=None, name="z")
    a = ColumnRef(table=None, name="a")
    b = ColumnRef(table=None, name="b")
    c = ColumnRef(table=None, name="c")

    # BinaryOp("AND", BinaryOp("OR", x, y), z) formats as (x OR y) AND z
    assert format_expr(BinaryOp("AND", BinaryOp("OR", x, y), z)) == "(x OR y) AND z"

    # BinaryOp("AND", z, BinaryOp("OR", x, y)) formats as z AND (x OR y)
    assert format_expr(BinaryOp("AND", z, BinaryOp("OR", x, y))) == "z AND (x OR y)"

    # BinaryOp("*", BinaryOp("+", x, y), z) formats as (x + y) * z
    assert format_expr(BinaryOp("*", BinaryOp("+", x, y), z)) == "(x + y) * z"

    # BinaryOp("-", a, BinaryOp("-", b, c)) formats as a - (b - c)
    assert format_expr(BinaryOp("-", a, BinaryOp("-", b, c))) == "a - (b - c)"

    # Standard unparenthesized expressions like x AND y and x * y + z format cleanly without unnecessary double parentheses
    assert format_expr(BinaryOp("AND", x, y)) == "x AND y"
    assert format_expr(BinaryOp("+", BinaryOp("*", x, y), z)) == "x * y + z"
    assert format_expr(BinaryOp("+", x, BinaryOp("*", y, z))) == "x + y * z"
    assert format_expr(BinaryOp("-", BinaryOp("-", a, b), c)) == "a - b - c"

    # Unary NOT wraps BinaryOp in parentheses
    assert format_expr(UnaryOp("NOT", BinaryOp("AND", x, y))) == "NOT (x AND y)"
    assert format_expr(UnaryOp("NOT", x)) == "NOT x"


def test_type_inference_division_yields_float():
    schema = [("a", DType.INT), ("b", DType.INT), ("c", DType.FLOAT)]
    assert _infer_expr_dtype(BinaryOp("/", ColumnRef(None, "a"), ColumnRef(None, "b")), schema) == DType.FLOAT
    assert _infer_expr_dtype(BinaryOp("/", ColumnRef(None, "a"), ColumnRef(None, "c")), schema) == DType.FLOAT
    assert _infer_expr_dtype(BinaryOp("/", Literal(10, DType.INT), Literal(2, DType.INT)), schema) == DType.FLOAT


def test_type_inference_numeric_coercion():
    schema = [("a", DType.INT), ("b", DType.FLOAT), ("s", DType.STRING)]
    # INT + INT -> INT
    assert _infer_expr_dtype(BinaryOp("+", ColumnRef(None, "a"), ColumnRef(None, "a")), schema) == DType.INT
    # INT * INT -> INT
    assert _infer_expr_dtype(BinaryOp("*", ColumnRef(None, "a"), ColumnRef(None, "a")), schema) == DType.INT
    # INT + FLOAT -> FLOAT
    assert _infer_expr_dtype(BinaryOp("+", ColumnRef(None, "a"), ColumnRef(None, "b")), schema) == DType.FLOAT
    # STRING + STRING -> STRING
    assert _infer_expr_dtype(BinaryOp("+", ColumnRef(None, "s"), ColumnRef(None, "s")), schema) == DType.STRING
    # Comparisons -> BOOL
    assert _infer_expr_dtype(BinaryOp("=", ColumnRef(None, "a"), ColumnRef(None, "b")), schema) == DType.BOOL
    assert _infer_expr_dtype(BinaryOp("<", ColumnRef(None, "a"), ColumnRef(None, "b")), schema) == DType.BOOL


def test_type_inference_column_scoping():
    schema = [
        ("customer.id", DType.STRING),
        ("orders.id", DType.INT),
        ("unqualified_col", DType.FLOAT),
    ]
    # Scoped column reference matches canonical table prefix
    assert _infer_expr_dtype(ColumnRef(table="orders", name="id"), schema) == DType.INT
    assert _infer_expr_dtype(ColumnRef(table="customer", name="id"), schema) == DType.STRING
    # Unqualified resolution
    assert _infer_expr_dtype(ColumnRef(table=None, name="unqualified_col"), schema) == DType.FLOAT
    # Non-existent column falls back to STRING
    assert _infer_expr_dtype(ColumnRef(table="other", name="id"), schema) == DType.STRING


def test_type_inference_nested_and_none():
    schema = [("x", DType.INT), ("y", DType.INT), ("z", DType.FLOAT)]
    # None expression handling
    assert _infer_expr_dtype(None, schema) == DType.STRING
    assert _infer_expr_dtype(Literal(None, dtype=DType.INT), schema) == DType.INT
    # Nested arithmetic (x + y) / z -> FLOAT
    nested = BinaryOp("/", BinaryOp("+", ColumnRef(None, "x"), ColumnRef(None, "y")), ColumnRef(None, "z"))
    assert _infer_expr_dtype(nested, schema) == DType.FLOAT
    # Aggregates
    assert _infer_expr_dtype(AggCall(func="count", arg=None), schema) == DType.INT
    assert _infer_expr_dtype(AggCall(func="avg", arg=ColumnRef(None, "x")), schema) == DType.FLOAT
    assert _infer_expr_dtype(AggCall(func="sum", arg=ColumnRef(None, "x")), schema) == DType.INT

