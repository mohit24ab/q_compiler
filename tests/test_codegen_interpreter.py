"""Phase C1: each operator on tiny hand-built tables with hand-computed results."""
import datetime

import pyarrow as pa
import pytest

from runtime import InterpreterError, Table, interpret
from runtime._compat import (
    AggCall, Aggregate, DType, Filter, Limit, Project, Scan, Sort, UnaryOp,
)

from codegen_fixtures import TABLES, col, lit, op, orders_join_customer, scan


def run(plan):
    return interpret(plan, TABLES)


# ------------------------------------------------------------------ Scan

def test_scan_returns_whole_table_tagged_with_table_name():
    out = run(scan("sales"))
    assert out.num_rows == 6
    assert out.column_names == ["id", "region", "amount", "qty", "day"]
    assert all(c.table == "sales" for c in out.columns)


def test_scan_column_subset_in_requested_order():
    out = run(scan("sales", columns=["qty", "id"]))
    assert out.to_rows() == [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]


def test_scan_pushed_predicate_may_use_unselected_column():
    out = run(scan("sales", columns=["id"], pred=op("=", col("region"), lit("EU", DType.STRING))))
    assert out.to_pydict() == {"id": [2, 5]}


def test_scan_accepts_pyarrow_tables():
    arrow = pa.table({"x": [3, 1, 2]})
    out = interpret(Sort(child=scan("a"), keys=[(col("x"), False)]), {"a": arrow})
    assert out.to_pydict() == {"x": [1, 2, 3]}


def test_scan_unknown_table_is_a_clear_error():
    with pytest.raises(InterpreterError, match="'nope' not provided"):
        run(scan("nope"))


# ------------------------------------------------------------------ Filter

def test_filter_drops_rows_where_predicate_is_null():
    # amount is NULL for id=4: NULL > 15 is NULL, so the row is dropped
    out = run(Filter(child=scan("sales"), predicate=op(">", col("amount"), lit(15.0, DType.FLOAT))))
    assert out.column("id").to_pylist() == [2, 3, 5, 6]


def test_filter_on_date_against_string_literal():
    pred = op(">=", col("day"), lit("2024-04-01", DType.DATE))
    out = run(Filter(child=scan("sales"), predicate=pred))
    assert out.column("id").to_pylist() == [4, 5, 6]
    assert out.column("day").to_pylist()[0] == datetime.date(2024, 4, 1)


def test_filter_and_or_three_valued_logic():
    # region = 'US' OR amount > 55   -> ids 1,3 (US) and 6 (60.0); id 4 is NULL OR FALSE = NULL
    pred = op("OR", op("=", col("region"), lit("US", DType.STRING)),
              op(">", col("amount"), lit(55.0, DType.FLOAT)))
    assert run(Filter(child=scan("sales"), predicate=pred)).column("id").to_pylist() == [1, 3, 6]
    # NOT (amount > 15)  -> id 1 only; NOT NULL stays NULL so id 4 is dropped
    pred = UnaryOp(op="NOT", operand=op(">", col("amount"), lit(15.0, DType.FLOAT)))
    assert run(Filter(child=scan("sales"), predicate=pred)).column("id").to_pylist() == [1]


def test_is_null():
    pred = UnaryOp(op="IS NULL", operand=col("region"))
    assert run(Filter(child=scan("sales"), predicate=pred)).column("id").to_pylist() == [6]


# ------------------------------------------------------------------ Project

def test_project_arithmetic_respects_tree_structure():
    # (qty + 1) * 2  vs  qty + (1 * 2): parenthesisation lives in the tree
    plan = Project(child=scan("sales"), exprs=[
        (op("*", op("+", col("qty"), lit(1)), lit(2)), "a"),
        (op("+", col("qty"), op("*", lit(1), lit(2))), "b"),
    ])
    out = run(plan)
    assert out.schema == [("a", DType.INT), ("b", DType.INT)]
    assert out.column("a").to_pylist() == [4, 6, 8, 10, 12, 14]
    assert out.column("b").to_pylist() == [3, 4, 5, 6, 7, 8]


def test_project_null_propagation_and_float_division():
    plan = Project(child=scan("sales"), exprs=[
        (op("/", col("amount"), col("qty")), "per_unit"),
        (op("/", col("qty"), lit(2)), "half"),
        (op("/", col("qty"), lit(0)), "div0"),
    ])
    out = run(plan)
    assert out.schema == [("per_unit", DType.FLOAT), ("half", DType.FLOAT), ("div0", DType.FLOAT)]
    assert out.column("per_unit").to_pylist() == [10.0, 10.0, 10.0, None, 10.0, 10.0]
    assert out.column("half").to_pylist() == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert out.column("div0").to_pylist() == [None] * 6


def test_modulo_truncates_like_sql():
    t = {"t": Table.from_pydict({"x": [-7, 7]}, [("x", DType.INT)])}
    out = interpret(Project(child=scan("t"), exprs=[(op("%", col("x"), lit(3)), "m")]), t)
    assert out.column("m").to_pylist() == [-1, 1]


# ------------------------------------------------------------------ Join

def test_inner_join_matches_and_null_key_never_matches():
    out = run(orders_join_customer("inner"))
    assert out.qualified_names() == ["orders.id", "orders.cust_id", "orders.total",
                                     "customer.id", "customer.name"]
    # order 103 has cust_id NULL -> dropped; customer 3 has no orders -> absent
    assert out.to_rows() == [(100, 1, 5, 1, "ann"), (101, 2, 7, 2, "bob"),
                             (102, 1, 9, 1, "ann")]


def test_left_join_pads_unmatched_rows_with_null():
    out = run(orders_join_customer("left"))
    assert out.to_rows()[-1] == (103, None, 11, None, None)
    assert out.num_rows == 4


def test_project_after_join_resolves_qualified_duplicates():
    plan = Project(child=orders_join_customer("inner"), exprs=[
        (col("id", "orders"), "order_id"), (col("name", "customer"), "name")])
    assert run(plan).to_rows() == [(100, "ann"), (101, "bob"), (102, "ann")]


def test_unqualified_ambiguous_column_is_an_error():
    plan = Project(child=orders_join_customer("inner"), exprs=[(col("id"), "id")])
    with pytest.raises(InterpreterError, match="ambiguous"):
        run(plan)


# ------------------------------------------------------------------ Aggregate

def by_region(child=None):
    return Aggregate(
        child=child or scan("sales"),
        group_keys=[col("region")],
        aggs=[(AggCall("sum", col("amount")), "s"),
              (AggCall("count", None), "n_rows"),
              (AggCall("count", col("amount")), "n_amt"),
              (AggCall("avg", col("amount")), "a"),
              (AggCall("min", col("qty")), "lo"),
              (AggCall("max", col("qty")), "hi")])


def test_group_by_all_aggregates_with_null_handling():
    out = run(by_region())
    assert out.schema == [("region", DType.STRING), ("s", DType.FLOAT), ("n_rows", DType.INT),
                          ("n_amt", DType.INT), ("a", DType.FLOAT), ("lo", DType.INT),
                          ("hi", DType.INT)]
    rows = {r[0]: r[1:] for r in out.to_rows()}
    assert rows == {
        "US": (40.0, 2, 2, 20.0, 1, 3),
        "EU": (70.0, 2, 2, 35.0, 2, 5),
        "AP": (None, 1, 0, None, 4, 4),   # only amount is NULL -> sum/avg NULL, count(x)=0
        None: (60.0, 1, 1, 60.0, 6, 6),   # NULL keys form one group
    }


def test_global_aggregate_on_empty_input_returns_one_row():
    empty = Filter(child=scan("sales"), predicate=lit(False, DType.BOOL))
    out = run(Aggregate(child=empty, group_keys=[],
                        aggs=[(AggCall("count", None), "n"), (AggCall("sum", col("qty")), "s")]))
    assert out.to_rows() == [(0, None)]


def test_grouped_aggregate_on_empty_input_returns_zero_rows():
    empty = Filter(child=scan("sales"), predicate=lit(False, DType.BOOL))
    assert run(by_region(empty)).num_rows == 0


def test_multi_key_grouping_over_join():
    plan = Aggregate(
        child=orders_join_customer("inner"),
        group_keys=[col("name", "customer"), op(">", col("total", "orders"), lit(6))],
        aggs=[(AggCall("sum", col("total", "orders")), "t")])
    out = run(plan)
    assert out.column_names == ["name", "(orders.total > 6)", "t"]
    assert sorted(out.to_rows()) == [("ann", False, 5), ("ann", True, 9), ("bob", True, 7)]


def test_having_and_order_by_aggregate_expression_above_aggregate():
    # SELECT region, sum(amount) s FROM sales GROUP BY region
    # HAVING count(*) >= 2 ORDER BY sum(amount) DESC
    agg = Aggregate(child=scan("sales"), group_keys=[col("region")],
                    aggs=[(AggCall("sum", col("amount")), "s"),
                          (AggCall("count", None), "n")])
    having = Filter(child=agg, predicate=op(">=", AggCall("count", None), lit(2)))
    ordered = Sort(child=having, keys=[(AggCall("sum", col("amount")), True)])
    out = run(Project(child=ordered, exprs=[(col("region"), "region"), (col("s"), "s")]))
    assert out.to_rows() == [("EU", 70.0), ("US", 40.0)]


# ------------------------------------------------------------------ Sort / Limit

def test_sort_multi_key_stable_with_nulls_last():
    asc = run(Sort(child=scan("sales"), keys=[(col("region"), False), (col("id"), True)]))
    assert asc.column("id").to_pylist() == [4, 5, 2, 3, 1, 6]
    desc = run(Sort(child=scan("sales"), keys=[(col("amount"), True)]))
    assert desc.column("id").to_pylist() == [6, 5, 3, 2, 1, 4]


def test_limit():
    assert run(Limit(child=scan("sales"), n=2)).column("id").to_pylist() == [1, 2]
    assert run(Limit(child=scan("sales"), n=0)).num_rows == 0
    assert run(Limit(child=scan("sales"), n=99)).num_rows == 6


# ------------------------------------------------------------------ whole pipeline

def test_canonical_naive_plan_shape_end_to_end():
    """Limit(Sort(Project(Filter(Aggregate(Filter(Join(Scan,Scan))))))) — Person A's A3 shape.

    SELECT c.name, sum(o.total) AS spent FROM orders o JOIN customer c ON o.cust_id = c.id
    WHERE o.total > 5 GROUP BY c.name HAVING sum(o.total) > 0 ORDER BY spent DESC LIMIT 1
    """
    join = orders_join_customer("inner")
    where = Filter(child=join, predicate=op(">", col("total", "orders"), lit(5)))
    agg = Aggregate(child=where, group_keys=[col("name", "customer")],
                    aggs=[(AggCall("sum", col("total", "orders")), "spent")])
    having = Filter(child=agg, predicate=op(">", AggCall("sum", col("total", "orders")), lit(0)))
    proj = Project(child=having, exprs=[(col("name", "customer"), "name"), (col("spent"), "spent")])
    plan = Limit(child=Sort(child=proj, keys=[(col("spent"), True)]), n=1)
    # ann: 9 (order 100 has total 5, filtered), bob: 7
    assert run(plan).to_rows() == [("ann", 9)]


def test_dispatch_works_for_scan_subclasses():
    class MockScan(Scan):
        pass
    out = interpret(MockScan(table="customer", columns=["name"], pushed_predicate=None), TABLES)
    assert out.to_pydict() == {"name": ["ann", "bob", "cyd"]}


# ------------------------------------------------------------------ join candidates (Phase C6)
# The interpreter indexes equality conjuncts so bench-scale joins finish. The index may only
# skip pairs that can't match: against the plain all-pairs loop, the output must be the
# same rows in the same order.

def _all_pairs(monkeypatch):
    import runtime.interpreter as interp
    monkeypatch.setattr(interp, "_candidate_finder",
                        lambda condition, left, right, rows: (lambda lrow: range(len(rows))))


def _same_as_all_pairs(plan, monkeypatch, tables=TABLES):
    from runtime import compare_tables
    indexed = interpret(plan, tables)
    with monkeypatch.context() as m:
        _all_pairs(m)
        everything = interpret(plan, tables)
    ok, why = compare_tables(everything, indexed, ordered=True)
    assert ok, why
    return indexed


@pytest.mark.parametrize("seed", range(150))
def test_indexed_join_equals_all_pairs_join(seed, monkeypatch):
    from test_codegen_join_aggregate_sort import _PlanFuzzer
    plan, _ = _PlanFuzzer(seed).pipeline()
    _same_as_all_pairs(plan, monkeypatch)


def test_indexed_join_key_types(monkeypatch):
    from runtime._compat import Join
    left = Table.from_pydict({"k": [1, 2, None, 3], "d": ["2024-01-01", "2023-12-31", None, "2024-01-02"]},
                             [("k", DType.INT), ("d", DType.STRING)], "l")
    right = Table.from_pydict(
        {"f": [1.0, 2.5, None, 3.0, 1.0],
         "day": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), None,
                 datetime.date(2024, 1, 2), datetime.date(2024, 1, 1)]},
        [("f", DType.FLOAT), ("day", DType.DATE)], "r")
    tables = {"l": left, "r": right}
    int_float = Join(left=scan("l"), right=scan("r"), kind="left",
                     condition=op("=", col("k", "l"), col("f", "r")))
    out = _same_as_all_pairs(int_float, monkeypatch, tables)
    assert [(r[0], r[2]) for r in out.to_rows()] == [(1, 1.0), (1, 1.0), (2, None), (None, None), (3, 3.0)]
    # STRING = DATE coerces in SQL but not in Python's ==: it must not be indexed
    str_date = Join(left=scan("l"), right=scan("r"), kind="inner",
                    condition=op("=", col("d", "l"), col("day", "r")))
    out = _same_as_all_pairs(str_date, monkeypatch, tables)
    assert sorted(r[1] for r in out.to_rows()) == ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]
