"""The optimizer's differential query suite: hand-built plans plus small synthetic tables.

These are the plans the frontend would produce for the SQL in each query's
docstring. Every pass is checked against every query here (see
test_opt_differential.py). When Person A's frontend and Person C's runtime
land, the real query suite runs alongside this one.

The data is chosen to exercise edge cases:

* NULLs in ``sales.amount``.
* Orders that reference missing customers, and customers with no orders,
  so inner and left joins differ.
* ``emp``/``dept``, whose column names collide (``id``, ``name``), so
  queries over them need qualified references.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable

import opt_ir  # noqa: F401  (must come before any ir import)
from ir.dtype import DType
from ir.expr import AggCall, BinaryOp, ColumnRef, Literal, UnaryOp
from ir.nodes import Aggregate, Filter, Join, Limit, Project, Scan, Sort

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

SCHEMAS: dict[str, list[tuple[str, DType]]] = {
    "sales": [
        ("sale_id", DType.INT), ("region", DType.STRING), ("product", DType.STRING),
        ("amount", DType.FLOAT), ("qty", DType.INT), ("sale_date", DType.DATE),
    ] + [(f"f{i:02d}", DType.INT) for i in range(6, 20)],  # 20 columns in total
    "orders": [
        ("o_id", DType.INT), ("o_custkey", DType.INT), ("o_total", DType.FLOAT),
        ("o_status", DType.STRING), ("o_date", DType.DATE),
    ],
    "customer": [
        ("c_id", DType.INT), ("c_name", DType.STRING), ("c_nationkey", DType.INT),
        ("c_segment", DType.STRING), ("c_balance", DType.FLOAT),
    ],
    "nation": [("n_id", DType.INT), ("n_name", DType.STRING), ("n_region", DType.STRING)],
    "emp": [("id", DType.INT), ("name", DType.STRING), ("dept_id", DType.INT), ("salary", DType.FLOAT)],
    "dept": [("id", DType.INT), ("name", DType.STRING), ("budget", DType.FLOAT)],
}


def _make_tables() -> dict[str, tuple[list[str], list[tuple]]]:
    rng = random.Random(20261006)
    rows: dict[str, list[tuple]] = {}
    rows["sales"] = [
        (
            i,
            rng.choice(["EU", "US", "AP", "LATAM"]),
            rng.choice(["widget", "gadget", "doohickey"]),
            None if rng.random() < 0.1 else round(rng.uniform(5, 250), 2),
            rng.randint(1, 9),
            f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
            *(rng.randint(0, 999) for _ in range(14)),
        )
        for i in range(1, 61)
    ]
    # Customers 11 and 12 place no orders; customer keys 13 and 14 do not exist.
    rows["orders"] = [
        (
            i,
            rng.choice(list(range(1, 11)) + [13, 14]),
            round(rng.uniform(10, 500), 2),
            rng.choice(["O", "F", "P"]),
            f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        )
        for i in range(1, 41)
    ]
    rows["customer"] = [
        (i, f"cust#{i}", rng.randint(1, 5), rng.choice(["AUTO", "BUILDING", "MACHINERY"]),
         round(rng.uniform(-100, 1000), 2))
        for i in range(1, 13)
    ]
    rows["nation"] = [
        (1, "FRANCE", "EUROPE"), (2, "BRAZIL", "AMERICA"), (3, "JAPAN", "ASIA"),
        (4, "KENYA", "AFRICA"), (5, "CANADA", "AMERICA"),
    ]
    rows["emp"] = [
        (1, "ada", 10, 120.0), (2, "bob", 20, 90.0), (3, "cy", 10, 105.0),
        (4, "dee", 30, 70.0), (5, "eve", 99, 50.0),
    ]
    rows["dept"] = [(10, "eng", 1000.0), (20, "ops", 400.0), (30, "sales", 600.0)]
    return {t: ([name for name, _ in SCHEMAS[t]], rows[t]) for t in SCHEMAS}


TABLES = _make_tables()


@dataclass(frozen=True)
class ColumnStats:
    """The fields of Person A's ``catalog.stats.ColumnStats``."""

    ndv: int
    min: Any
    max: Any
    null_count: int


class SuiteCatalog:
    """Contract §6 over in-memory rows: schema, row counts and column statistics.

    ``stats`` computes exact statistics the way Person A's
    ``compute_column_stats`` does: distinct non-NULL values, min and max of the
    non-NULL values (None for an all-NULL column), and the NULL count.
    """

    def __init__(self, tables: dict[str, tuple[list[str], list[tuple]]] | None = None,
                 schemas: dict[str, list[tuple[str, DType]]] | None = None):
        self.tables = TABLES if tables is None else tables
        self.schemas = SCHEMAS if schemas is None else schemas

    def schema(self, table: str) -> list[tuple[str, DType]]:
        return list(self.schemas[table])

    def row_count(self, table: str) -> int:
        return len(self.tables[table][1])

    def stats(self, table: str, column: str) -> ColumnStats:
        names, rows = self.tables[table]
        if column not in names:
            raise KeyError(f"Column '{column}' not found in table '{table}'.")
        i = names.index(column)
        values = [r[i] for r in rows if r[i] is not None]
        return ColumnStats(
            ndv=len(set(values)),
            min=min(values) if values else None,
            max=max(values) if values else None,
            null_count=len(rows) - len(values),
        )


CATALOG = SuiteCatalog()

# --------------------------------------------------------------------------
# Plan-building shorthand
# --------------------------------------------------------------------------


def col(name: str, table: str | None = None) -> ColumnRef:
    return ColumnRef(table=table, name=name)


def lit(value: Any) -> Literal:
    dtype = {bool: DType.BOOL, int: DType.INT, float: DType.FLOAT, str: DType.STRING}[type(value)]
    return Literal(value=value, dtype=dtype)


def op(symbol: str, left: Any, right: Any) -> BinaryOp:
    return BinaryOp(op=symbol, left=left, right=right)


def scan(table: str, *, bound: bool = False, pushed: Any = None) -> Scan:
    """Build a Scan. ``bound=True`` fills table_schema, as a binder would; otherwise the pass must ask the catalog."""
    schema = list(SCHEMAS[table]) if bound else None
    return Scan(table=table, columns=None, pushed_predicate=pushed, table_schema=schema)


def project(child: Any, *exprs: tuple[Any, str]) -> Project:
    return Project(child=child, exprs=list(exprs))


def keep(*names: str, table: str | None = None) -> list[tuple[Any, str]]:
    """Project expressions that pass named columns straight through."""
    return [(col(n, table), n) for n in names]


def agg(func: str, arg: Any = None) -> AggCall:
    return AggCall(func=func, arg=arg)


def join(left: Any, right: Any, condition: Any, kind: str = "inner") -> Join:
    return Join(left=left, right=right, condition=condition, kind=kind)


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


@dataclass
class Query:
    name: str
    build: Callable[[], Any]
    sql: str = field(default="")

    @property
    def plan(self) -> Any:
        return self.build()


QUERIES: list[Query] = []


def query(fn: Callable[[], Any]) -> Callable[[], Any]:
    QUERIES.append(Query(fn.__name__, fn, (fn.__doc__ or "").strip()))
    return fn


@query
def two_of_twenty():
    """SELECT sale_id, amount FROM sales"""
    return project(scan("sales"), *keep("sale_id", "amount"))


@query
def filter_only_column():
    """SELECT sale_id FROM sales WHERE region = 'EU'"""
    return project(Filter(child=scan("sales"), predicate=op("=", col("region"), lit("EU"))),
                   *keep("sale_id"))


@query
def computed_projection():
    """SELECT qty * amount AS value FROM sales WHERE region = 'EU' AND qty > 2"""
    pred = op("AND", op("=", col("region"), lit("EU")), op(">", col("qty"), lit(2)))
    return project(Filter(child=scan("sales", bound=True), predicate=pred),
                   (op("*", col("qty"), col("amount")), "value"))


@query
def select_star():
    """SELECT * FROM sales WHERE qty > 3"""
    return Filter(child=scan("sales"), predicate=op(">", col("qty"), lit(3)))


@query
def pushed_predicate_column():
    """SELECT sale_id FROM sales WHERE amount > 100   -- predicate already in the scan"""
    return project(scan("sales", pushed=op(">", col("amount"), lit(100.0))), *keep("sale_id"))


@query
def sort_key_only_column():
    """SELECT sale_id, amount FROM sales ORDER BY qty DESC, sale_id LIMIT 5"""
    sort = Sort(child=scan("sales"), keys=[(col("qty"), True), (col("sale_id"), False)])
    return Limit(child=project(sort, *keep("sale_id", "amount")), n=5)


@query
def nested_projects():
    """SELECT a FROM (SELECT sale_id AS a, amount * 2 AS b, region AS c FROM sales)"""
    inner = project(scan("sales"), (col("sale_id"), "a"), (op("*", col("amount"), lit(2.0)), "b"),
                    (col("region"), "c"))
    return project(inner, *keep("a"))


@query
def aggregate_unused_aggs():
    """SELECT region, total FROM (SELECT region, SUM(amount) total, COUNT(*) n, MAX(qty) max_qty FROM sales GROUP BY region)"""
    aggregate = Aggregate(child=scan("sales"), group_keys=[col("region")],
                          aggs=[(agg("sum", col("amount")), "total"), (agg("count"), "n"),
                                (agg("max", col("qty")), "max_qty")])
    return project(aggregate, *keep("region", "total"))


@query
def having_on_aggregate():
    """SELECT region FROM sales GROUP BY region HAVING SUM(amount) > 100   (COUNT(*) computed but unused)"""
    aggregate = Aggregate(child=scan("sales"), group_keys=[col("region")],
                          aggs=[(agg("sum", col("amount")), "total"), (agg("count"), "n")])
    having = Filter(child=aggregate, predicate=op(">", col("total"), lit(1000.0)))
    return project(having, *keep("region"))


@query
def having_names_an_aggregate_the_select_list_drops():
    """SELECT region FROM sales GROUP BY region HAVING COUNT(*) > 12
    -- as the binder builds it: HAVING repeats the call, and COUNT(*) is not selected"""
    count = agg("count")
    grouped = Aggregate(child=scan("sales"), group_keys=[col("region", "sales")], aggs=[(count, "count(*)")])
    return project(Filter(child=grouped, predicate=op(">", count, lit(12))), (col("region", "sales"), "region"))


@query
def order_by_an_aggregate_the_select_list_drops():
    """SELECT region FROM sales GROUP BY region ORDER BY SUM(amount) DESC"""
    total = agg("sum", col("amount"))
    grouped = Aggregate(child=scan("sales"), group_keys=[col("region")], aggs=[(total, "sum(amount)")])
    return project(Sort(child=grouped, keys=[(total, True)]), *keep("region"))


@query
def group_by_without_aggs():
    """SELECT region FROM (SELECT region, COUNT(*) n FROM sales GROUP BY region)"""
    aggregate = Aggregate(child=scan("sales"), group_keys=[col("region")], aggs=[(agg("count"), "n")])
    return project(aggregate, *keep("region"))


@query
def count_star():
    """SELECT COUNT(*) AS n FROM sales"""
    return Aggregate(child=scan("sales", bound=True), group_keys=[], aggs=[(agg("count"), "n")])


@query
def constant_over_global_aggregate():
    """SELECT 1 AS one FROM (SELECT SUM(amount) s, COUNT(*) n FROM sales)"""
    aggregate = Aggregate(child=scan("sales"), group_keys=[],
                          aggs=[(agg("sum", col("amount")), "s"), (agg("count"), "n")])
    return project(aggregate, (lit(1), "one"))


@query
def join_condition_only_columns():
    """SELECT orders.o_id, customer.c_name FROM orders JOIN customer ON orders.o_custkey = customer.c_id"""
    j = join(scan("orders"), scan("customer"), op("=", col("o_custkey", "orders"), col("c_id", "customer")))
    return project(j, (col("o_id", "orders"), "o_id"), (col("c_name", "customer"), "c_name"))


@query
def left_join():
    """SELECT c.c_name, o.o_total FROM customer c LEFT JOIN orders o ON c.c_id = o.o_custkey"""
    j = join(scan("customer"), scan("orders"),
             op("=", col("c_id", "customer"), col("o_custkey", "orders")), kind="left")
    return project(j, (col("c_name", "customer"), "c_name"), (col("o_total", "orders"), "o_total"))


@query
def colliding_names():
    """SELECT emp.name AS emp_name, dept.name AS dept_name FROM emp JOIN dept ON emp.dept_id = dept.id"""
    j = join(scan("emp"), scan("dept"), op("=", col("dept_id", "emp"), col("id", "dept")))
    return project(j, (col("name", "emp"), "emp_name"), (col("name", "dept"), "dept_name"))


@query
def three_way_join_unqualified():
    """SELECT n_name, SUM(o_total) revenue FROM orders JOIN customer ON o_custkey = c_id
    JOIN nation ON c_nationkey = n_id GROUP BY n_name"""
    inner = join(scan("orders"), scan("customer"), op("=", col("o_custkey"), col("c_id")))
    outer = join(inner, scan("nation"), op("=", col("c_nationkey"), col("n_id")))
    return Aggregate(child=outer, group_keys=[col("n_name")], aggs=[(agg("sum", col("o_total")), "revenue")])


@query
def three_way_join_qualified():
    """Same as three_way_join_unqualified, with every reference qualified."""
    inner = join(scan("orders"), scan("customer"),
                 op("=", col("o_custkey", "orders"), col("c_id", "customer")))
    outer = join(inner, scan("nation"), op("=", col("c_nationkey", "customer"), col("n_id", "nation")))
    return Aggregate(child=outer, group_keys=[col("n_name", "nation")],
                     aggs=[(agg("sum", col("o_total", "orders")), "revenue")])


@query
def filter_over_join_unqualified():
    """SELECT o_id FROM orders JOIN customer ON o_custkey = c_id WHERE c_segment = 'AUTO'"""
    j = join(scan("orders"), scan("customer"), op("=", col("o_custkey"), col("c_id")))
    return project(Filter(child=j, predicate=op("=", col("c_segment"), lit("AUTO"))), *keep("o_id"))


@query
def sorted_join():
    """SELECT o_id, c_name FROM orders JOIN customer ON o_custkey = c_id ORDER BY o_id"""
    j = join(scan("orders"), scan("customer"), op("=", col("o_custkey"), col("c_id")))
    return project(Sort(child=j, keys=[(col("o_id"), False)]), *keep("o_id", "c_name"))


# --------------------------------------------------------------------------
# Predicate pushdown (B3)
#
# The "must not push" queries are built so that the illegal push gives a
# different answer on this data. The differential test would then fail.
# test_opt_differential.py's negative controls check this.
# --------------------------------------------------------------------------


def is_null(expr: Any) -> UnaryOp:
    return UnaryOp(op="IS NULL", operand=expr)


@query
def stacked_filters():
    """SELECT * FROM (SELECT * FROM sales WHERE qty > 2) WHERE region = 'EU'"""
    inner = Filter(child=scan("sales"), predicate=op(">", col("qty"), lit(2)))
    return Filter(child=inner, predicate=op("=", col("region"), lit("EU")))


@query
def filter_merges_with_pushed_predicate():
    """SELECT sale_id FROM sales WHERE qty > 5   -- over a scan that already filters amount"""
    base = scan("sales", pushed=op(">", col("amount"), lit(50.0)))
    return project(Filter(child=base, predicate=op(">", col("qty"), lit(5))), *keep("sale_id"))


@query
def filter_through_computed_project():
    """SELECT * FROM (SELECT sale_id, qty * amount AS value FROM sales) WHERE value > 500"""
    inner = project(scan("sales"), (col("sale_id"), "sale_id"), (op("*", col("qty"), col("amount")), "value"))
    return Filter(child=inner, predicate=op(">", col("value"), lit(500.0)))


@query
def filter_through_sort():
    """SELECT * FROM (SELECT sale_id, qty FROM sales ORDER BY qty DESC, sale_id) WHERE qty < 4"""
    inner = Sort(child=project(scan("sales"), *keep("sale_id", "qty")),
                 keys=[(col("qty"), True), (col("sale_id"), False)])
    return Filter(child=inner, predicate=op("<", col("qty"), lit(4)))


@query
def filter_above_limit_must_not_push():
    """SELECT * FROM (SELECT sale_id, qty FROM sales ORDER BY sale_id LIMIT 10) WHERE qty > 5"""
    inner = Limit(child=Sort(child=project(scan("sales"), *keep("sale_id", "qty")),
                             keys=[(col("sale_id"), False)]), n=10)
    return Filter(child=inner, predicate=op(">", col("qty"), lit(5)))


@query
def where_splits_across_inner_join():
    """SELECT o_id, c_name FROM orders JOIN customer ON o_custkey = c_id
    WHERE o_total > 100 AND c_segment = 'AUTO' AND o_custkey + c_nationkey > 4"""
    j = join(scan("orders"), scan("customer"), op("=", col("o_custkey"), col("c_id")))
    pred = op("AND", op("AND", op(">", col("o_total"), lit(100.0)), op("=", col("c_segment"), lit("AUTO"))),
              op(">", op("+", col("o_custkey"), col("c_nationkey")), lit(4)))
    return project(Filter(child=j, predicate=pred), *keep("o_id", "c_name"))


@query
def single_side_on_conjuncts_of_inner_join():
    """SELECT o_id FROM orders JOIN customer ON o_custkey = c_id AND o_status = 'F' AND c_balance > 0"""
    cond = op("AND", op("AND", op("=", col("o_custkey"), col("c_id")), op("=", col("o_status"), lit("F"))),
              op(">", col("c_balance"), lit(0.0)))
    return project(join(scan("orders"), scan("customer"), cond), *keep("o_id"))


@query
def filter_through_three_way_join():
    """SELECT o_id FROM orders JOIN customer ON o_custkey = c_id JOIN nation ON c_nationkey = n_id
    WHERE n_region = 'AMERICA'"""
    inner = join(scan("orders"), scan("customer"), op("=", col("o_custkey"), col("c_id")))
    outer = join(inner, scan("nation"), op("=", col("c_nationkey"), col("n_id")))
    return project(Filter(child=outer, predicate=op("=", col("n_region"), lit("AMERICA"))), *keep("o_id"))


@query
def left_join_where_on_preserved_side():
    """SELECT c_name, o_total FROM customer LEFT JOIN orders ON c_id = o_custkey WHERE c_segment = 'AUTO'"""
    j = join(scan("customer"), scan("orders"), op("=", col("c_id"), col("o_custkey")), kind="left")
    return project(Filter(child=j, predicate=op("=", col("c_segment"), lit("AUTO"))), *keep("c_name", "o_total"))


@query
def left_join_null_rejecting_where_becomes_inner():
    """SELECT c_name, o_total FROM customer LEFT JOIN orders ON c_id = o_custkey WHERE o_total > 200"""
    j = join(scan("customer"), scan("orders"), op("=", col("c_id"), col("o_custkey")), kind="left")
    return project(Filter(child=j, predicate=op(">", col("o_total"), lit(200.0))), *keep("c_name", "o_total"))


@query
def left_join_is_null_must_not_push():
    """SELECT c_name FROM customer LEFT JOIN orders ON c_id = o_custkey WHERE o_id IS NULL   -- anti-join"""
    j = join(scan("customer"), scan("orders"), op("=", col("c_id"), col("o_custkey")), kind="left")
    return project(Filter(child=j, predicate=is_null(col("o_id"))), *keep("c_name"))


@query
def left_join_or_with_preserved_side_must_not_push():
    """SELECT c_name, o_total FROM customer LEFT JOIN orders ON c_id = o_custkey
    WHERE o_total > 200 OR c_name = 'cust#11'   -- customer 11 has no orders, by construction"""
    j = join(scan("customer"), scan("orders"), op("=", col("c_id"), col("o_custkey")), kind="left")
    pred = op("OR", op(">", col("o_total"), lit(200.0)), op("=", col("c_name"), lit("cust#11")))
    return project(Filter(child=j, predicate=pred), *keep("c_name", "o_total"))


@query
def left_join_on_conjuncts():
    """SELECT c_name, o_total FROM customer LEFT JOIN orders
    ON c_id = o_custkey AND o_status = 'F' AND c_segment = 'AUTO'"""
    cond = op("AND", op("AND", op("=", col("c_id"), col("o_custkey")), op("=", col("o_status"), lit("F"))),
              op("=", col("c_segment"), lit("AUTO")))
    j = join(scan("customer"), scan("orders"), cond, kind="left")
    return project(j, *keep("c_name", "o_total"))


@query
def having_splits_on_group_key():
    """SELECT region, total FROM sales GROUP BY region HAVING region <> 'AP' AND SUM(amount) > 1500"""
    aggregate = Aggregate(child=scan("sales"), group_keys=[col("region")],
                          aggs=[(agg("sum", col("amount")), "total")])
    pred = op("AND", op("<>", col("region"), lit("AP")), op(">", col("total"), lit(1500.0)))
    return project(Filter(child=aggregate, predicate=pred), *keep("region", "total"))


@query
def having_on_an_aggregate_of_a_group_key_must_not_push():
    """SELECT region FROM sales GROUP BY region HAVING MAX(region) = 'EU'
    -- MAX(region) reads only the group key, but is a result of the Aggregate"""
    top = agg("max", col("region"))
    grouped = Aggregate(child=scan("sales"), group_keys=[col("region")], aggs=[(top, "max(region)")])
    return project(Filter(child=grouped, predicate=op("=", top, lit("EU"))), *keep("region"))


@query
def having_mixes_an_aggregate_and_a_group_key():
    """SELECT region FROM sales GROUP BY region HAVING SUM(amount) > 1500.0 AND region <> 'AP'
    -- the binder's form: the region conjunct can push, the SUM conjunct cannot"""
    total = agg("sum", col("amount"))
    grouped = Aggregate(child=scan("sales"), group_keys=[col("region")], aggs=[(total, "sum(amount)")])
    pred = op("AND", op(">", total, lit(1500.0)), op("<>", col("region"), lit("AP")))
    return project(Filter(child=grouped, predicate=pred), *keep("region"))


@query
def constant_false_over_global_aggregate_must_not_push():
    """SELECT n FROM (SELECT COUNT(*) n FROM sales) WHERE 1 = 0"""
    aggregate = Aggregate(child=scan("sales"), group_keys=[], aggs=[(agg("count"), "n")])
    return Filter(child=aggregate, predicate=op("=", lit(1), lit(0)))


@query
def filter_through_project_over_join():
    """SELECT * FROM (SELECT orders.o_id AS oid, customer.c_segment AS seg FROM orders JOIN customer
    ON orders.o_custkey = customer.c_id) WHERE seg = 'BUILDING' AND oid > 10"""
    j = join(scan("orders"), scan("customer"), op("=", col("o_custkey", "orders"), col("c_id", "customer")))
    inner = project(j, (col("o_id", "orders"), "oid"), (col("c_segment", "customer"), "seg"))
    pred = op("AND", op("=", col("seg"), lit("BUILDING")), op(">", col("oid"), lit(10)))
    return Filter(child=inner, predicate=pred)


# --------------------------------------------------------------------------
# Constant folding and simplification (B4)
# --------------------------------------------------------------------------


def date(value: str) -> Literal:
    return Literal(value=value, dtype=DType.DATE)


NULL = Literal(value=None, dtype=DType.FLOAT)
WHERE_FALSE = op("=", lit(1), lit(0))


@query
def constant_arithmetic_in_filter():
    """SELECT sale_id FROM sales WHERE qty > 1 + 2"""
    return project(Filter(child=scan("sales"), predicate=op(">", col("qty"), op("+", lit(1), lit(2)))),
                   *keep("sale_id"))


@query
def constant_date_comparison():
    """SELECT sale_id FROM sales WHERE DATE '2024-01-01' < DATE '2024-06-01' AND region = 'EU'"""
    pred = op("AND", op("<", date("2024-01-01"), date("2024-06-01")), op("=", col("region"), lit("EU")))
    return project(Filter(child=scan("sales"), predicate=pred), *keep("sale_id"))


@query
def or_true_removes_the_filter():
    """SELECT sale_id FROM sales WHERE region = 'EU' OR 1 = 1"""
    pred = op("OR", op("=", col("region"), lit("EU")), op("=", lit(1), lit(1)))
    return project(Filter(child=scan("sales"), predicate=pred), *keep("sale_id"))


@query
def double_negation():
    """SELECT sale_id FROM sales WHERE NOT NOT (qty > 4)"""
    pred = UnaryOp(op="NOT", operand=UnaryOp(op="NOT", operand=op(">", col("qty"), lit(4))))
    return project(Filter(child=scan("sales"), predicate=pred), *keep("sale_id"))


@query
def contradiction_on_equalities():
    """SELECT sale_id FROM sales WHERE qty = 1 AND qty = 2"""
    pred = op("AND", op("=", col("qty"), lit(1)), op("=", col("qty"), lit(2)))
    return project(Filter(child=scan("sales"), predicate=pred), *keep("sale_id"))


@query
def contradiction_on_range():
    """SELECT sale_id FROM sales WHERE qty > 5 AND 3 > qty"""
    pred = op("AND", op(">", col("qty"), lit(5)), op(">", lit(3), col("qty")))
    return project(Filter(child=scan("sales"), predicate=pred), *keep("sale_id"))


@query
def satisfiable_point_range():
    """SELECT sale_id FROM sales WHERE qty >= 2 AND qty <= 2   -- not a contradiction: qty = 2"""
    pred = op("AND", op(">=", col("qty"), lit(2)), op("<=", col("qty"), lit(2)))
    return project(Filter(child=scan("sales"), predicate=pred), *keep("sale_id"))


@query
def null_comparison_rejects_every_row():
    """SELECT sale_id FROM sales WHERE region = 'EU' AND amount = NULL"""
    pred = op("AND", op("=", col("region"), lit("EU")), op("=", col("amount"), NULL))
    return project(Filter(child=scan("sales"), predicate=pred), *keep("sale_id"))


@query
def false_filter_over_join_and_sort():
    """SELECT o_id, c_name FROM orders JOIN customer ON o_custkey = c_id WHERE 1 = 0 ORDER BY o_id"""
    j = join(scan("orders"), scan("customer"), op("=", col("o_custkey"), col("c_id")))
    return Sort(child=project(Filter(child=j, predicate=WHERE_FALSE), *keep("o_id", "c_name")),
                keys=[(col("o_id"), False)])


@query
def empty_side_of_inner_join():
    """SELECT o_id FROM orders JOIN (SELECT * FROM customer WHERE 1 = 0) c ON o_custkey = c_id"""
    j = join(scan("orders"), Filter(child=scan("customer"), predicate=WHERE_FALSE),
             op("=", col("o_custkey"), col("c_id")))
    return project(j, *keep("o_id"))


@query
def empty_right_side_of_left_join_must_not_lift():
    """SELECT c_name, o_total FROM customer LEFT JOIN (SELECT * FROM orders WHERE 1 = 0) o ON c_id = o_custkey"""
    j = join(scan("customer"), Filter(child=scan("orders"), predicate=WHERE_FALSE),
             op("=", col("c_id"), col("o_custkey")), kind="left")
    return project(j, *keep("c_name", "o_total"))


@query
def global_aggregate_over_empty_must_not_lift():
    """SELECT COUNT(*) n, SUM(amount) s FROM sales WHERE qty = 1 AND qty = 2   -- one row: (0, NULL)"""
    pred = op("AND", op("=", col("qty"), lit(1)), op("=", col("qty"), lit(2)))
    return Aggregate(child=Filter(child=scan("sales"), predicate=pred), group_keys=[],
                     aggs=[(agg("count"), "n"), (agg("sum", col("amount")), "s")])


@query
def grouped_aggregate_over_empty():
    """SELECT region, COUNT(*) n FROM sales WHERE 1 = 0 GROUP BY region"""
    return Aggregate(child=Filter(child=scan("sales"), predicate=WHERE_FALSE),
                     group_keys=[col("region")], aggs=[(agg("count"), "n")])


@query
def limit_zero():
    """SELECT sale_id FROM sales ORDER BY sale_id LIMIT 0"""
    return Limit(child=Sort(child=project(scan("sales"), *keep("sale_id")), keys=[(col("sale_id"), False)]), n=0)


@query
def folded_select_expressions():
    """SELECT sale_id, 2 * 3 AS six, 1.5 * 2 AS three, 7 / 2 AS seven_halves, -(4) AS neg FROM sales"""
    return project(scan("sales"), (col("sale_id"), "sale_id"), (op("*", lit(2), lit(3)), "six"),
                   (op("*", lit(1.5), lit(2)), "three"), (op("/", lit(7), lit(2)), "seven_halves"),
                   (UnaryOp(op="-", operand=lit(4)), "neg"))


@query
def contradiction_in_select_must_stay():
    """SELECT sale_id, (amount = 10.0 AND amount = 20.0) AS flag FROM sales   -- NULL, not FALSE, for NULL amounts"""
    flag = op("AND", op("=", col("amount"), lit(10.0)), op("=", col("amount"), lit(20.0)))
    return project(scan("sales"), (col("sale_id"), "sale_id"), (flag, "flag"))


@query
def noop_project_over_scan():
    """SELECT id, name, dept_id, salary FROM emp   -- every column, in order"""
    return project(scan("emp"), *keep("id", "name", "dept_id", "salary"))


@query
def constant_conjuncts_in_join_and_scan():
    """SELECT o_id FROM (SELECT * FROM orders WHERE 1 = 1) JOIN customer ON o_custkey = c_id AND 2 > 1"""
    left = scan("orders", pushed=op("=", lit(1), lit(1)))
    j = join(left, scan("customer"), op("AND", op("=", col("o_custkey"), col("c_id")), op(">", lit(2), lit(1))))
    return project(j, *keep("o_id"))


@query
def null_and_in_select_must_stay():
    """SELECT sale_id, (amount > 100 AND NULL) AS flag FROM sales   -- NULL where amount > 100, else FALSE"""
    flag = op("AND", op(">", col("amount"), lit(100.0)), Literal(value=None, dtype=DType.BOOL))
    return project(scan("sales"), (col("sale_id"), "sale_id"), (flag, "flag"))


@query
def reordering_project_must_stay():
    """SELECT name, id, dept_id, salary FROM emp   -- every column, but not in table order"""
    return project(scan("emp"), *keep("name", "id", "dept_id", "salary"))
