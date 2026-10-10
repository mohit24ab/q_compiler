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


# ---------------------------------------------------------------- Phase C4 corpus

PAYMENTS = Table.from_pydict(
    {"order_id": [100, 100, 101, 103, None, 999],
     "amt":      [5, 4, 7, 11, 3, 1],
     "method":   ["card", "cash", "card", None, "cash", "card"]},
    [("order_id", DType.INT), ("amt", DType.INT), ("method", DType.STRING)],
)
TABLES["payments"] = PAYMENTS


def _agg(func, arg=None):
    return AggCall(func, arg)


def _orders_payments(kind, extra=None):
    cond = op("=", col("id", "orders"), col("order_id", "payments"))
    if extra is not None:
        cond = op("AND", cond, extra)
    return Join(left=scan("orders"), right=scan("payments"), condition=cond, kind=kind)


# name -> (plan, ordered?)
JOIN_AGG_PLANS = {
    "three_way_join": (
        Join(left=orders_join_customer("inner"), right=scan("payments"),
             condition=op("=", col("id", "orders"), col("order_id", "payments")), kind="inner"),
        False),
    "three_way_left_joins": (
        Join(left=orders_join_customer("left"), right=scan("payments"),
             condition=op("=", col("id", "orders"), col("order_id", "payments")), kind="left"),
        False),
    "multi_key_join": (
        _orders_payments("inner", op("=", col("total", "orders"), col("amt", "payments"))), False),
    "join_key_expression": (
        Join(left=scan("orders"), right=scan("payments"), kind="inner",
             condition=op("=", op("+", col("id", "orders"), lit(0)), col("order_id", "payments"))),
        False),
    "join_residual_inner": (
        _orders_payments("inner", op("<", col("amt", "payments"), col("total", "orders"))), False),
    "join_residual_left_pads_when_residual_fails": (
        _orders_payments("left", op("<", col("amt", "payments"), col("total", "orders"))), False),
    "left_join_empty_right": (
        Join(left=scan("orders"), right=Filter(child=scan("payments"),
                                               predicate=op(">", col("amt"), lit(100))),
             condition=op("=", col("id", "orders"), col("order_id", "payments")), kind="left"),
        False),
    "inner_join_empty_left": (
        Join(left=Filter(child=scan("orders"), predicate=op(">", col("total"), lit(100))),
             right=scan("payments"), kind="inner",
             condition=op("=", col("id", "orders"), col("order_id", "payments"))),
        False),
    "non_equi_only_join": (
        Join(left=scan("customer"), right=scan("orders"), kind="inner",
             condition=op("<", col("id", "customer"), col("cust_id", "orders"))),
        False),
    "join_limit_without_sort_keeps_interpreter_order": (
        Limit(child=_orders_payments("left"), n=3), True),
    "multi_key_group_by": (
        Aggregate(child=scan("sales"), group_keys=[col("region"), op(">", col("qty"), lit(3))],
                  aggs=[(_agg("count"), "n"), (_agg("sum", col("qty")), "s"),
                        (_agg("avg", col("amount")), "a"), (_agg("min", col("day")), "first_day"),
                        (_agg("max", col("region")), "max_region")]),
        False),
    "group_by_expression_having_order": (
        Sort(child=Filter(
            child=Aggregate(child=scan("sales"), group_keys=[op("%", col("qty"), lit(2))],
                            aggs=[(_agg("sum", col("amount")), "s"), (_agg("count"), "n")]),
            predicate=op(">", _agg("count"), lit(1))),
            keys=[(_agg("sum", col("amount")), True), (op("%", col("qty"), lit(2)), False)]),
        True),
    "aggregate_over_left_join_counts": (
        Aggregate(child=_orders_payments("left"), group_keys=[col("id", "orders")],
                  aggs=[(_agg("count"), "rows"), (_agg("count", col("amt", "payments")), "paid"),
                        (_agg("sum", col("amt", "payments")), "amount"),
                        (_agg("max", col("method", "payments")), "method")]),
        False),
    "global_aggregates_all_kinds": (
        Aggregate(child=scan("sales"), group_keys=[], aggs=[
            (_agg("count"), "n"), (_agg("count", col("amount")), "na"),
            (_agg("sum", col("qty")), "sq"), (_agg("avg", col("qty")), "aq"),
            (_agg("min", col("region")), "lo"), (_agg("max", col("day")), "hi"),
            (_agg("sum", col("amount")), "sa")]),
        False),
    "global_aggregate_all_null_input": (
        Aggregate(child=Filter(child=scan("sales"), predicate=UnaryOp(op="IS NULL", operand=col("amount"))),
                  group_keys=[], aggs=[(_agg("sum", col("amount")), "s"), (_agg("avg", col("amount")), "a"),
                                       (_agg("min", col("amount")), "m"), (_agg("count", col("amount")), "c")]),
        False),
    "global_aggregate_over_false_filter": (
        Aggregate(child=Filter(child=scan("sales"), predicate=lit(False, DType.BOOL)),
                  group_keys=[], aggs=[(_agg("count"), "n"), (_agg("max", col("qty")), "m")]),
        False),
    "sort_mixed_directions_with_nulls": (
        Sort(child=scan("sales"), keys=[(col("region"), True), (col("amount"), False)]), True),
    "sort_on_computed_expression_stable_ties": (
        Sort(child=scan("sales"), keys=[(op("%", col("qty"), lit(3)), False)]), True),
    "sort_strings_and_dates": (
        Sort(child=scan("payments"), keys=[(col("method"), False), (col("amt"), True)]), True),
    "limit_zero_and_overflow": (
        Limit(child=Limit(child=Sort(child=scan("sales"), keys=[(col("id"), True)]), n=100), n=0), True),
    "top_n_per_joined_aggregate": (
        Limit(child=Sort(child=Project(
            child=Aggregate(child=orders_join_customer("inner"), group_keys=[col("name", "customer")],
                            aggs=[(_agg("avg", col("total", "orders")), "avg_total")]),
            exprs=[(col("name"), "name"), (op("*", col("avg_total"), lit(2)), "dbl")]),
            keys=[(col("dbl"), True)]), n=2),
        True),
    # The binder's shape for `SELECT region, SUM(amount) AS total ... ORDER BY SUM(amount)`:
    # the Sort sits above the Project and names the aggregate, not its alias.
    "order_by_aggregate_above_project": (
        Limit(child=Sort(child=Project(
            child=Aggregate(child=scan("sales"), group_keys=[col("region", "sales")],
                            aggs=[(_agg("sum", col("amount", "sales")), "total")]),
            exprs=[(col("region", "sales"), "region"), (col("total"), "total")]),
            keys=[(_agg("sum", col("amount", "sales")), True)]), n=3),
        True),
    # ... and `ORDER BY SUM(amount) + 1 DESC, qty * 2` with both expressions in the select list
    "order_by_select_list_expressions_above_project": (
        Sort(child=Project(
            child=Aggregate(child=scan("sales"), group_keys=[col("qty", "sales")],
                            aggs=[(_agg("sum", col("amount", "sales")), "total")]),
            exprs=[(op("*", col("qty", "sales"), lit(2)), "dbl"), (col("total"), "total")]),
            keys=[(op("+", _agg("sum", col("amount", "sales")), lit(1)), True),
                  (op("*", col("qty", "sales"), lit(2)), False)]),
        True),
}


# ---------------------------------------------------------------- B's review (Oct 2026)

EVENTS = Table.from_pydict(
    {"id":  [1, 2, 3, 4, 5],
     "day": ["2024-01-05", "2024-03-01", None, "2024-02-02", "2024-01-05"],
     "txt": ["2024-01-05", "2024-02-01", "2024-02-02", None, "20240105"]},  # ISO basic form too
    [("id", DType.INT), ("day", DType.DATE), ("txt", DType.STRING)],
)
TABLES["events"] = EVENTS


def _count(child, keys=()):
    return Aggregate(child=child, group_keys=list(keys), aggs=[(_agg("count"), "n")])


def _events_where(predicate):
    return Project(child=Filter(child=scan("events"), predicate=predicate),
                   exprs=[(col("id", "events"), "id")])


REVIEW_PLANS = {
    # Constant GROUP BY / join keys: generated code called .tolist() on a constant.
    "group_by_constant_only": (
        Aggregate(child=scan("sales"), group_keys=[lit("all", DType.STRING)],
                  aggs=[(_agg("count"), "n"), (_agg("sum", col("qty", "sales")), "q")]),
        False),
    "group_by_constant_over_no_rows": (   # no rows: no group (unlike a global aggregate)
        _count(Filter(child=scan("sales"), predicate=op(">", col("qty"), lit(100))),
               [lit("all", DType.STRING)]),
        False),
    "group_by_column_and_constants": (
        _count(scan("sales"), [col("region", "sales"), op("+", lit(1), lit(1)),
                               lit(None, DType.INT)]),
        False),
    "group_by_constant_null_only_known_at_run_time": (   # 1 / (1 - 1)
        _count(scan("sales"), [op("/", lit(1), op("-", lit(1), lit(1))), col("region", "sales")]),
        False),
    "group_by_key_null_on_every_row": (   # region || NULL: an array, NULL everywhere
        _count(scan("sales"), [op("||", col("region", "sales"), lit(None, DType.STRING))]),
        False),
    "join_on_constant_null_key_inner": (   # orders.id / 0 is NULL on every row
        Join(left=scan("orders"), right=scan("payments"),
             condition=op("=", op("/", col("id", "orders"), lit(0)), col("order_id", "payments")),
             kind="inner"),
        False),
    "join_on_constant_null_key_left": (
        Join(left=scan("orders"), right=scan("payments"),
             condition=op("=", op("/", col("id", "orders"), lit(0)), col("order_id", "payments")),
             kind="left"),
        False),
    "join_on_constant_true_key": (   # count(*) is never NULL, so `n IS NOT NULL` is TRUE
        Join(left=_count(scan("orders"), [col("cust_id", "orders")]), right=scan("payments"),
             condition=op("=", UnaryOp(op="IS NOT NULL", operand=col("n")),
                          op(">", col("amt", "payments"), lit(4))),
             kind="inner"),
        False),
    # Relations with rows but no columns: the interpreter counted 0 rows.
    "scan_with_no_columns": (scan("sales", columns=[]), False),
    "count_over_scan_with_no_columns": (_count(scan("sales", columns=[])), False),
    "count_over_pushed_scan_with_no_columns": (
        _count(scan("sales", columns=[], pred=op(">", col("qty"), lit(2)))), False),
    "count_over_filter_over_no_columns": (
        _count(Filter(child=scan("sales", columns=[]), predicate=lit(True, DType.BOOL))), False),
    "count_over_project_with_no_exprs": (_count(Project(child=scan("sales"), exprs=[])), False),
    "count_over_join_of_no_columns": (
        _count(Join(left=scan("sales", columns=[]), right=scan("orders", columns=[]),
                    condition=lit(True, DType.BOOL), kind="inner")),
        False),
    "count_over_limit_of_no_columns": (_count(Limit(child=scan("sales", columns=[]), n=4)), False),
    "group_by_over_no_columns": (_count(scan("sales", columns=[]), [lit(7)]), False),
    # A DATE compared with a STRING: the string is parsed as an ISO date, in both engines.
    "date_column_equals_string_column": (_events_where(op("=", col("day"), col("txt"))), False),
    "string_column_before_date_column": (_events_where(op("<", col("txt"), col("day"))), False),
    "date_column_equals_iso_basic_string": (
        _events_where(op("=", col("day"), lit("20240105", DType.STRING))), False),
    "iso_basic_date_literal_equals_date_column": (
        _events_where(op("=", lit("20240105", DType.DATE), col("day"))), False),
    "string_column_after_date_literal": (
        _events_where(op(">=", col("txt"), lit("2024-02-01", DType.DATE))), False),
    # `||` on a side that isn't a STRING (the binder accepts it since main @ a2ed35c)
    "concat_every_type": (
        Project(child=scan("sales"), exprs=[
            (op("||", col("id"), col("region")), "int_str"),
            (op("||", col("amount"), lit("!", DType.STRING)), "float_str"),
            (op("||", op(">", col("qty"), lit(3)), col("day")), "bool_date"),
            (op("||", lit(7), op("||", lit(2.5, DType.FLOAT), lit(True, DType.BOOL))), "constants"),
            (op("||", lit("2024-01-05", DType.DATE), col("qty")), "date_int")]),
        False),
}
JOIN_AGG_PLANS.update(REVIEW_PLANS)
