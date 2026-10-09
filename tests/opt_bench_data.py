"""Larger tables with the suite's schemas, for measuring runtime (the ablation study).

The suite's own tables have 3 to 80 rows: enough to check answers, too few to
time. These have the same columns and value domains, so every suite query
runs on them unchanged (``cust#11`` exists, ``k9_id = 2`` exists, the same
regions and segments), but each table is about 50 times larger, and
``scale`` multiplies that. Generated deterministically.

At scale 1 the largest cross product any configuration builds (the comma
join run with no passes) is 2,000 x 5 x 200 = 2 million pairs.
"""

from __future__ import annotations

import random

from opt_query_suite import SCHEMAS, SuiteCatalog

BASE = {"sales": 2000, "orders": 2000, "customer": 200, "lineitem": 4000, "emp": 200, "k": 200}


def make_tables(scale: int = 1) -> dict[str, tuple[list[str], list[tuple]]]:
    rng = random.Random(1234)
    n = {k: v * scale for k, v in BASE.items()}

    def date():
        return f"2024-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"

    rows: dict[str, list[tuple]] = {}
    rows["sales"] = [
        (i, rng.choice(["EU", "US", "AP", "LATAM"]), rng.choice(["widget", "gadget", "doohickey"]),
         None if rng.random() < 0.1 else round(rng.uniform(5, 250), 2), rng.randint(1, 9), date(),
         *(rng.randint(0, 999) for _ in range(14)))
        for i in range(1, n["sales"] + 1)
    ]
    # 5% of orders reference customers that do not exist.
    rows["orders"] = [
        (i, rng.randint(n["customer"] + 1, n["customer"] + 20) if rng.random() < 0.05
         else rng.randint(1, n["customer"]),
         round(rng.uniform(10, 500), 2), rng.choice(["O", "F", "P"]), date())
        for i in range(1, n["orders"] + 1)
    ]
    rows["customer"] = [
        (i, f"cust#{i}", rng.randint(1, 5), rng.choice(["AUTO", "BUILDING", "MACHINERY"]),
         round(rng.uniform(-100, 1000), 2))
        for i in range(1, n["customer"] + 1)
    ]
    rows["nation"] = [
        (1, "FRANCE", "EUROPE"), (2, "BRAZIL", "AMERICA"), (3, "JAPAN", "ASIA"),
        (4, "KENYA", "AFRICA"), (5, "CANADA", "AMERICA"),
    ]
    rows["region"] = [("EUROPE", "old world"), ("AMERICA", "new world"), ("ASIA", "far east"),
                      ("AFRICA", "the cradle"), ("OCEANIA", "islands")]
    rows["emp"] = [(i, f"emp{i}", rng.choice([10, 20, 30, 99]), round(rng.uniform(40, 150), 1))
                   for i in range(1, n["emp"] + 1)]
    rows["dept"] = [(10, "eng", 1000.0), (20, "ops", 400.0), (30, "sales", 600.0)]
    rows["lineitem"] = [
        (i, rng.randint(1, n["orders"] + 20), rng.randint(1, 50), round(rng.uniform(1, 100), 2))
        for i in range(1, n["lineitem"] + 1)
    ]
    for t in range(1, 10):
        rows[f"k{t}"] = [(j, rng.randint(1, n["k"])) for j in range(1, n["k"] + 1)]
    return {t: ([name for name, _ in SCHEMAS[t]], rows[t]) for t in SCHEMAS}


def catalog(tables) -> SuiteCatalog:
    return SuiteCatalog(tables, SCHEMAS)
