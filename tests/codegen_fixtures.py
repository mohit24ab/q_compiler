"""Shared tiny tables, plan-building helpers and sample plans for Person C's tests.

SAMPLE_PLANS is the differential corpus: every codegen phase must produce exactly
what the interpreter produces on each of these. Add a plan here whenever a bug is found.
"""
from runtime import Table
from runtime._compat import (
    AggCall, Aggregate, BinaryOp, ColumnRef, DType, Filter, Join, Limit, Literal,
    Project, Scan, Sort, UnaryOp,
)


def col(name, table=None):
    return ColumnRef(table=table, name=name)


def lit(value, dtype=DType.INT):
    return Literal(value=value, dtype=dtype)


def op(o, a, b):
    return BinaryOp(op=o, left=a, right=b)


def scan(table, columns=None, pred=None):
    return Scan(table=table, columns=columns, pushed_predicate=pred)


SALES = Table.from_pydict(
    {
        "id":     [1, 2, 3, 4, 5, 6],
        "region": ["US", "EU", "US", "AP", "EU", None],
        "amount": [10.0, 20.0, 30.0, None, 50.0, 60.0],
        "qty":    [1, 2, 3, 4, 5, 6],
        "day":    ["2024-01-01", "2024-02-01", "2024-03-01",
                   "2024-04-01", "2024-05-01", "2024-06-01"],
    },
    [("id", DType.INT), ("region", DType.STRING), ("amount", DType.FLOAT),
     ("qty", DType.INT), ("day", DType.DATE)],
)

ORDERS = Table.from_pydict(
    {"id": [100, 101, 102, 103], "cust_id": [1, 2, 1, None], "total": [5, 7, 9, 11]},
    [("id", DType.INT), ("cust_id", DType.INT), ("total", DType.INT)],
)

CUSTOMER = Table.from_pydict(
    {"id": [1, 2, 3], "name": ["ann", "bob", "cyd"]},
    [("id", DType.INT), ("name", DType.STRING)],
)

TABLES = {"sales": SALES, "orders": ORDERS, "customer": CUSTOMER}


def orders_join_customer(kind="inner"):
    cond = op("=", col("cust_id", "orders"), col("id", "customer"))
    return Join(left=scan("orders"), right=scan("customer"), condition=cond, kind=kind)


def _canonical_naive_plan():
    """Limit(Sort(Project(Filter(Aggregate(Filter(Join(Scan,Scan))))))) — A3's shape."""
    where = Filter(child=orders_join_customer(), predicate=op(">", col("total", "orders"), lit(5)))
    agg = Aggregate(child=where, group_keys=[col("name", "customer")],
                    aggs=[(AggCall("sum", col("total", "orders")), "spent")])
    having = Filter(child=agg, predicate=op(">", AggCall("sum", col("total", "orders")), lit(0)))
    proj = Project(child=having, exprs=[(col("name", "customer"), "name"), (col("spent"), "spent")])
    return Limit(child=Sort(child=proj, keys=[(col("spent"), True)]), n=1)


# name -> (plan, ordered?)   ordered=True when the query has ORDER BY
SAMPLE_PLANS = {
    "scan_all": (scan("sales"), False),
    "scan_pruned_pushed": (
        scan("sales", columns=["id", "qty"], pred=op("=", col("region"), lit("EU", DType.STRING))),
        False),
    "filter_null_drop": (
        Filter(child=scan("sales"), predicate=op(">", col("amount"), lit(15.0, DType.FLOAT))),
        False),
    "filter_date_kleene": (
        Filter(child=scan("sales"), predicate=op(
            "OR", op(">=", col("day"), lit("2024-05-01", DType.DATE)),
            UnaryOp(op="NOT", operand=op(">", col("amount"), lit(15.0, DType.FLOAT))))),
        False),
    "project_arith": (
        Project(child=scan("sales"), exprs=[
            (op("*", op("+", col("qty"), lit(1)), lit(2)), "a"),
            (op("/", col("amount"), col("qty")), "per_unit"),
            (op("%", op("-", lit(0), col("qty")), lit(4)), "m")]),
        False),
    "inner_join": (orders_join_customer("inner"), False),
    "left_join": (orders_join_customer("left"), False),
    "group_by_all_aggs": (
        Aggregate(child=scan("sales"), group_keys=[col("region")], aggs=[
            (AggCall("sum", col("amount")), "s"), (AggCall("count", None), "n"),
            (AggCall("count", col("amount")), "na"), (AggCall("avg", col("amount")), "a"),
            (AggCall("min", col("qty")), "lo"), (AggCall("max", col("day")), "hi")]),
        False),
    "global_agg_empty": (
        Aggregate(child=Filter(child=scan("sales"), predicate=lit(False, DType.BOOL)),
                  group_keys=[], aggs=[(AggCall("count", None), "n"),
                                       (AggCall("sum", col("qty")), "s")]),
        False),
    "sort_multi_nulls": (
        Sort(child=scan("sales"), keys=[(col("region"), False), (col("amount"), True)]),
        True),
    "top_n": (Limit(child=Sort(child=scan("sales"), keys=[(col("qty"), True)]), n=3), True),
    "canonical_naive": (_canonical_naive_plan(), True),

    # Shapes Person B's column pruning (B2) produces — confirmed with B, keep forever.
    "pruned_group_by_without_aggs": (
        Project(child=Aggregate(child=scan("sales", columns=["region"]),
                                group_keys=[col("region")], aggs=[]),
                exprs=[(col("region"), "region")]),
        False),
    "pruned_scan_columns_cover_pushed_predicate": (
        scan("sales", columns=["id", "amount"], pred=op(">", col("amount"), lit(25.0, DType.FLOAT))),
        False),
    "pruned_scan_explicit_columns": (scan("sales", columns=["amount", "id"]), False),

    # Join conditions pushdown (B3) can leave behind.
    "join_true_inner": (
        Join(left=scan("customer"), right=scan("orders", columns=["total"]),
             condition=lit(True, DType.BOOL), kind="inner"), False),
    "join_true_left_empty_right": (
        Join(left=scan("customer"),
             right=Filter(child=scan("orders"), predicate=lit(False, DType.BOOL)),
             condition=lit(True, DType.BOOL), kind="left"), False),
    "join_false_inner": (
        Join(left=scan("customer"), right=scan("orders"),
             condition=lit(False, DType.BOOL), kind="inner"), False),
    "join_no_condition": (
        Join(left=scan("customer"), right=scan("orders", columns=["id"]),
             condition=None, kind="inner"), False),

    "filter_is_null_and_is_not_null": (
        Filter(child=scan("sales"), predicate=op(
            "OR", UnaryOp(op="IS NULL", operand=col("region")),
            UnaryOp(op="IS NOT NULL", operand=col("amount")))),
        False),
}
