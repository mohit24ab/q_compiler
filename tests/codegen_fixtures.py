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


class FixtureCatalog:
    """The part of Contract §6 codegen needs: schema(table)."""

    def schema(self, table):
        return list(TABLES[table].schema)

    def row_count(self, table):
        return TABLES[table].num_rows


CATALOG = FixtureCatalog()


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


def _s(value):
    return lit(value, DType.STRING)


def _f(value):
    return lit(value, DType.FLOAT)


# Phase C3: single-table queries (Scan / Filter / Project only) — must compile fully.
# name -> (plan, the SQL it stands for)
SINGLE_TABLE_PLANS = {
    "select_star": (scan("sales"), "SELECT * FROM sales"),
    "two_columns": (
        Project(child=scan("sales", columns=["id", "qty"]),
                exprs=[(col("qty"), "qty"), (col("id"), "id")]),
        "SELECT qty, id FROM sales"),
    "where_null_dropped": (
        Filter(child=scan("sales"), predicate=op(">", col("amount"), _f(15.0))),
        "SELECT * FROM sales WHERE amount > 15"),
    "pushed_on_unselected_column": (
        scan("sales", columns=["id"], pred=op("=", col("region"), _s("EU"))),
        "SELECT id FROM sales WHERE region = 'EU'   -- after pushdown + pruning"),
    "arith_precedence": (
        Project(child=scan("sales"), exprs=[
            (op("-", op("-", col("qty"), lit(1)), lit(2)), "left_assoc"),
            (op("-", col("qty"), op("-", lit(1), lit(2))), "right_assoc"),
            (op("*", op("+", col("qty"), lit(1)), lit(2)), "sum_times"),
            (op("+", col("qty"), op("*", lit(1), lit(2))), "plus_product")]),
        "SELECT (qty-1)-2, qty-(1-2), (qty+1)*2, qty+1*2 FROM sales"),
    "division_and_modulo": (
        Project(child=scan("sales"), exprs=[
            (op("/", col("amount"), col("qty")), "per_unit"),
            (op("/", col("qty"), lit(0)), "div_zero"),
            (op("%", op("-", lit(0), col("qty")), lit(4)), "neg_mod"),
            (op("/", col("qty"), op("-", col("qty"), lit(3))), "div_by_col_zero")]),
        "SELECT amount/qty, qty/0, (0-qty)%4, qty/(qty-3) FROM sales"),
    "kleene_and_or_not": (
        Filter(child=scan("sales"), predicate=op(
            "OR", op("AND", op(">", col("amount"), _f(15.0)), op("<>", col("region"), _s("EU"))),
            UnaryOp(op="NOT", operand=op("<", col("qty"), lit(6))))),
        "SELECT * FROM sales WHERE (amount > 15 AND region <> 'EU') OR NOT qty < 6"),
    "is_null_projection": (
        Project(child=scan("sales"), exprs=[
            (UnaryOp(op="IS NULL", operand=col("amount")), "amount_missing"),
            (UnaryOp(op="IS NOT NULL", operand=op("+", col("amount"), col("qty"))), "sum_present")]),
        "SELECT amount IS NULL, (amount + qty) IS NOT NULL FROM sales"),
    "date_and_string_literals": (
        Filter(child=scan("sales"), predicate=op(
            "AND", op(">=", col("day"), _s("2024-03-01")), op("<", col("day"), lit("2024-06-01", DType.DATE)))),
        "SELECT * FROM sales WHERE day >= '2024-03-01' AND day < DATE '2024-06-01'"),
    "like_and_concat": (
        Project(child=Filter(child=scan("sales"), predicate=op("LIKE", col("region"), _s("%U%"))),
                exprs=[(op("||", col("region"), _s("-x")), "tag"), (col("id"), "id")]),
        "SELECT region || '-x', id FROM sales WHERE region LIKE '%U%'"),
    "constants_and_null_literal": (
        Project(child=scan("sales", columns=["id"]), exprs=[
            (col("id"), "id"), (lit(7), "seven"), (_s("k"), "k"),
            (lit(None, DType.INT), "nothing"), (op("+", col("id"), lit(None, DType.INT)), "plus_null")]),
        "SELECT id, 7, 'k', NULL, id + NULL FROM sales"),
    "constant_false_filter": (
        Filter(child=scan("sales"), predicate=lit(False, DType.BOOL)),
        "SELECT * FROM sales WHERE FALSE"),
    "stacked_filter_project": (
        Project(child=Filter(child=Project(child=Filter(
            child=scan("sales"), predicate=op(">", col("qty"), lit(1))),
            exprs=[(col("id"), "id"), (op("*", col("amount"), lit(2)), "dbl")]),
            predicate=op("<", col("dbl"), _f(110.0))),
            exprs=[(op("+", col("dbl"), col("id")), "z")]),
        "SELECT dbl + id FROM (SELECT id, amount*2 AS dbl FROM sales WHERE qty > 1) WHERE dbl < 110"),
    "qualified_columns": (
        Project(child=Filter(child=scan("orders"), predicate=op(">", col("total", "orders"), lit(6))),
                exprs=[(col("id", "orders"), "id"), (col("cust_id", "orders"), "cust")]),
        "SELECT orders.id, orders.cust_id AS cust FROM orders WHERE orders.total > 6"),
}
