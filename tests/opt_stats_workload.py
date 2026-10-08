"""A workload for validating the cardinality estimator, not the rewrites.

The differential suite's tables have at most 60 rows and were built for
edge cases, so its estimation errors mix model errors with small-number
noise. These tables are larger, and each column is generated to either meet
or break one assumption of the estimator (optimizer/stats.py):

  uniformity    ``u`` is uniform; ``z`` follows a Zipf law (value k is
                drawn with probability proportional to 1/k).
  independence  ``c`` is independent of ``a``; ``b`` always equals ``a``.
  containment   every ``users.uid`` appears among ``events.user_id``,
                which also holds ids with no user (dangling keys).

Each query records the assumption it tests and whether the data meets it.
Where it does, the estimate should be close (test_opt_stats.py checks
this); where it doesn't, the error is the point, and it goes into the
report (docs/cardinality_estimates.md).
"""

from __future__ import annotations

import datetime
import random
from dataclasses import dataclass
from typing import Any, Callable

import opt_ir  # noqa: F401  (must come before any ir import)
from ir.dtype import DType
from ir.expr import UnaryOp
from ir.nodes import Aggregate, Filter, Limit
from opt_query_suite import SuiteCatalog, agg, col, date, join, lit, op, project, scan

SCHEMAS: dict[str, list[tuple[str, DType]]] = {
    "events": [
        ("e_id", DType.INT), ("u", DType.INT), ("z", DType.INT), ("a", DType.INT),
        ("b", DType.INT), ("c", DType.INT), ("d", DType.DATE), ("cat", DType.STRING),
        ("amt", DType.FLOAT), ("user_id", DType.INT),
    ],
    "users": [("uid", DType.INT), ("tier", DType.STRING)],
    "clicks": [("k_id", DType.INT), ("kz", DType.INT)],
}

EVENTS, USERS, CLICKS = 1000, 100, 300
ZIPF_N = 50


def _zipf(rng: random.Random) -> int:
    return rng.choices(range(1, ZIPF_N + 1), weights=[1 / k for k in range(1, ZIPF_N + 1)])[0]


def _make_tables() -> dict[str, tuple[list[str], list[tuple]]]:
    rng = random.Random(5)
    day0 = datetime.date(2024, 1, 1)
    events = []
    for i in range(1, EVENTS + 1):
        a = rng.randint(0, 19)
        events.append((
            i,
            rng.randint(0, 99),                                          # u: uniform
            _zipf(rng),                                                  # z: skewed
            a, a,                                                        # b = a: correlated
            rng.randint(0, 19),                                          # c: independent of a
            (day0 + datetime.timedelta(days=rng.randint(0, 365))).isoformat(),
            f"cat_{rng.randint(0, 19):02d}",
            None if rng.random() < 0.25 else round(rng.uniform(0, 100), 2),
            rng.randint(1, 120),                                         # ids 101..120 have no user
        ))
    users = [(i, rng.choice(["bronze", "silver", "gold", "platinum"])) for i in range(1, USERS + 1)]
    clicks = [(i, _zipf(rng)) for i in range(1, CLICKS + 1)]
    rows = {"events": events, "users": users, "clicks": clicks}
    return {t: ([n for n, _ in SCHEMAS[t]], rows[t]) for t in SCHEMAS}


TABLES = _make_tables()
CATALOG = SuiteCatalog(TABLES, SCHEMAS)


@dataclass
class WorkloadQuery:
    name: str
    tests: str        # the estimator rule or assumption this query exercises
    assumption_holds: bool
    build: Callable[[], Any]

    @property
    def plan(self) -> Any:
        return self.build()


QUERIES: list[WorkloadQuery] = []


def query(tests: str, holds: bool = True):
    def register(fn):
        QUERIES.append(WorkloadQuery(fn.__name__, tests, holds, fn))
        return fn
    return register


def where(table: str, predicate: Any) -> Any:
    return Filter(child=scan(table), predicate=predicate)


@query("equality on a uniform column: 1/ndv")
def uniform_equality():
    return where("events", op("=", col("u"), lit(42)))


@query("range on a uniform column: interpolation over [min, max]")
def uniform_range():
    return where("events", op("<", col("u"), lit(25)))


@query("two bounds on one column: one interval, not a product")
def range_pair():
    return where("events", op("AND", op(">=", col("u"), lit(20)), op("<", col("u"), lit(30))))


@query("range on a DATE column: interpolation over day numbers")
def date_range():
    return where("events", op(">=", col("d"), date("2024-10-01")))


@query("IS NULL: the catalog's null_count")
def null_fraction():
    return where("events", UnaryOp(op="IS NULL", operand=col("amt")))


@query("AND of independent columns: product")
def independent_and():
    return where("events", op("AND", op("<", col("u"), lit(50)), op("=", col("c"), lit(3))))


@query("OR: inclusion-exclusion")
def disjunction():
    return where("events", op("OR", op("<", col("u"), lit(10)), op("<", col("c"), lit(2))))


@query("NOT pushed into the comparison: NOT u < 90 is u >= 90")
def negated_range():
    return where("events", UnaryOp(op="NOT", operand=op("<", col("u"), lit(90))))


@query("equality on a skewed column, most frequent value: uniformity", holds=False)
def skewed_equality_frequent():
    return where("events", op("=", col("z"), lit(1)))


@query("equality on a skewed column, rarest value: uniformity", holds=False)
def skewed_equality_rare():
    return where("events", op("=", col("z"), lit(ZIPF_N)))


@query("AND of correlated columns (b = a): independence", holds=False)
def correlated_and():
    return where("events", op("AND", op("=", col("a"), lit(7)), op("=", col("b"), lit(7))))


@query("range on a STRING column: no interpolation, System R's 1/3", holds=False)
def string_range():
    return where("events", op("<", col("cat"), lit("cat_05")))


@query("foreign-key join with dangling keys: |L||R| / max(ndv)")
def fk_join():
    return join(scan("events"), scan("users"), op("=", col("user_id"), col("uid")))


@query("join with a filtered dimension: ndv shrinks with the filter")
def filtered_dimension_join():
    users = where("users", op("=", col("tier"), lit("gold")))
    return join(scan("events"), users, op("=", col("user_id"), col("uid")))


@query("LEFT join: every left row at least once")
def left_join_users():
    return join(scan("users"), scan("events"), op("=", col("uid"), col("user_id")), kind="left")


@query("join on skewed keys on both sides: uniformity", holds=False)
def skewed_join():
    return join(scan("events"), scan("clicks"), op("=", col("z"), col("kz")))


@query("GROUP BY one column: ndv")
def group_by_one():
    return Aggregate(child=scan("events"), group_keys=[col("a")], aggs=[(agg("count"), "n")])


@query("GROUP BY two correlated columns: product of ndv", holds=False)
def group_by_correlated():
    return Aggregate(child=scan("events"), group_keys=[col("a"), col("b")], aggs=[(agg("count"), "n")])


@query("GROUP BY over a filter: urn-model ndv after selection")
def group_by_after_filter():
    child = where("events", op("<", col("u"), lit(10)))
    return Aggregate(child=child, group_keys=[col("c")], aggs=[(agg("sum", col("amt")), "total")])


@query("HAVING on an aggregate: no statistics, System R's 1/3", holds=False)
def having():
    grouped = Aggregate(child=scan("events"), group_keys=[col("cat")], aggs=[(agg("count"), "n")])
    return project(Filter(child=grouped, predicate=op(">", col("n"), lit(55))), (col("cat"), "cat"))


@query("LIMIT: min(n, rows)")
def limit():
    return Limit(child=where("events", op("<", col("u"), lit(50))), n=10)
