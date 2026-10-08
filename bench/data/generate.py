from __future__ import annotations

import datetime
import random
from typing import Literal

import pyarrow as pa

from catalog.catalog import Catalog


NATIONS = [
    "ALGERIA",
    "ARGENTINA",
    "BRAZIL",
    "CANADA",
    "EGYPT",
    "ETHIOPIA",
    "FRANCE",
    "GERMANY",
    "INDIA",
    "INDONESIA",
    "IRAN",
    "IRAQ",
    "JAPAN",
    "JORDAN",
    "KENYA",
    "MOROCCO",
    "MOZAMBIQUE",
    "PERU",
    "CHINA",
    "ROMANIA",
    "SAUDI ARABIA",
    "VIETNAM",
    "RUSSIA",
    "UNITED KINGDOM",
    "UNITED STATES",
]

MKTSEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]

ORDER_PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
ORDER_STATUSES = ["O", "F", "P"]  # Open, Fulfilled, Pending
RETURN_FLAGS = ["R", "A", "N"]
LINE_STATUSES = ["O", "F"]

MANUFACTURERS = [f"Manufacturer#{i}" for i in range(1, 6)]
BRANDS = [f"Brand#{i}{j}" for i in range(1, 6) for j in range(1, 6)]
PART_TYPES = [
    "STANDARD ANODIZED TIN",
    "STANDARD BRUSHED COPPER",
    "STANDARD POLISHED STEEL",
    "SMALL ANODIZED BRASS",
    "SMALL BRUSHED NICKEL",
    "MEDIUM POLISHED TIN",
    "LARGE BRUSHED STEEL",
    "ECONOMY BURNISHED BRASS",
    "PROMO PLATED STEEL",
]


def generate_customer(n: int, rng: random.Random) -> pa.Table:
    """Generates customer dimension table."""
    ids = list(range(1, n + 1))
    names = [f"Customer#{i:08d}" for i in ids]
    nations = [rng.choice(NATIONS) for _ in range(n)]
    phones = [f"{rng.randint(10, 99)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}" for _ in range(n)]
    acctbals = [round(rng.uniform(-999.0, 9999.0), 2) for _ in range(n)]
    mktsegments = [rng.choice(MKTSEGMENTS) for _ in range(n)]

    return pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "name": pa.array(names, type=pa.string()),
            "nation": pa.array(nations, type=pa.string()),
            "phone": pa.array(phones, type=pa.string()),
            "acctbal": pa.array(acctbals, type=pa.float64()),
            "mktsegment": pa.array(mktsegments, type=pa.string()),
        }
    )


def generate_part(n: int, rng: random.Random) -> pa.Table:
    """Generates part dimension table."""
    ids = list(range(1, n + 1))
    names = [f"Part#{i:08d}" for i in ids]
    mfgrs = [rng.choice(MANUFACTURERS) for _ in range(n)]
    brands = [rng.choice(BRANDS) for _ in range(n)]
    types = [rng.choice(PART_TYPES) for _ in range(n)]
    sizes = [rng.randint(1, 50) for _ in range(n)]
    retail_prices = [round(rng.uniform(900.0, 2100.0), 2) for _ in range(n)]

    return pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "name": pa.array(names, type=pa.string()),
            "mfgr": pa.array(mfgrs, type=pa.string()),
            "brand": pa.array(brands, type=pa.string()),
            "type": pa.array(types, type=pa.string()),
            "size": pa.array(sizes, type=pa.int64()),
            "retail_price": pa.array(retail_prices, type=pa.float64()),
        }
    )


def generate_orders(n: int, num_customers: int, rng: random.Random) -> pa.Table:
    """Generates orders dimension table."""
    start_date = datetime.date(1992, 1, 1)
    ids = list(range(1, n + 1))
    cust_ids = [rng.randint(1, num_customers) for _ in range(n)]
    order_statuses = [rng.choice(ORDER_STATUSES) for _ in range(n)]
    total_prices = [round(rng.uniform(100.0, 50000.0), 2) for _ in range(n)]
    order_dates = [start_date + datetime.timedelta(days=rng.randint(0, 2400)) for _ in range(n)]
    order_priorities = [rng.choice(ORDER_PRIORITIES) for _ in range(n)]

    return pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "cust_id": pa.array(cust_ids, type=pa.int64()),
            "order_status": pa.array(order_statuses, type=pa.string()),
            "total_price": pa.array(total_prices, type=pa.float64()),
            "order_date": pa.array(order_dates, type=pa.date32()),
            "order_priority": pa.array(order_priorities, type=pa.string()),
        }
    )


def generate_lineitem(n: int, num_orders: int, num_parts: int, rng: random.Random) -> pa.Table:
    """Generates lineitem fact table."""
    base_date = datetime.date(1992, 1, 1)
    ids = list(range(1, n + 1))
    order_ids = [rng.randint(1, num_orders) for _ in range(n)]
    part_ids = [rng.randint(1, num_parts) for _ in range(n)]
    quantities = [rng.randint(1, 50) for _ in range(n)]
    extended_prices = [round(quantities[i] * rng.uniform(20.0, 500.0), 2) for i in range(n)]
    discounts = [round(rng.choice([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]), 2) for _ in range(n)]
    taxes = [round(rng.choice([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]), 2) for _ in range(n)]
    ship_dates = [base_date + datetime.timedelta(days=rng.randint(1, 2500)) for _ in range(n)]
    return_flags = [rng.choice(RETURN_FLAGS) for _ in range(n)]
    statuses = [rng.choice(LINE_STATUSES) for _ in range(n)]

    return pa.table(
        {
            "id": pa.array(ids, type=pa.int64()),
            "order_id": pa.array(order_ids, type=pa.int64()),
            "part_id": pa.array(part_ids, type=pa.int64()),
            "quantity": pa.array(quantities, type=pa.int64()),
            "extended_price": pa.array(extended_prices, type=pa.float64()),
            "discount": pa.array(discounts, type=pa.float64()),
            "tax": pa.array(taxes, type=pa.float64()),
            "ship_date": pa.array(ship_dates, type=pa.date32()),
            "return_flag": pa.array(return_flags, type=pa.string()),
            "status": pa.array(statuses, type=pa.string()),
        }
    )


def generate_dataset(
    scale: Literal["tiny", "bench"] | str = "tiny",
    seed: int = 42,
) -> dict[str, pa.Table]:
    """Generates all 4 star-schema tables at tiny or benchmark scale deterministically."""
    rng = random.Random(seed)

    if scale == "tiny":
        num_cust = 100
        num_parts = 100
        num_orders = 250
        num_lineitems = 1000
    elif scale == "bench":
        num_cust = 5000
        num_parts = 5000
        num_orders = 25000
        num_lineitems = 100000
    else:
        raise ValueError(f"Unknown scale: {scale}. Expected 'tiny' or 'bench'.")

    customer_table = generate_customer(num_cust, rng)
    part_table = generate_part(num_parts, rng)
    orders_table = generate_orders(num_orders, num_cust, rng)
    lineitem_table = generate_lineitem(num_lineitems, num_orders, num_parts, rng)

    return {
        "customer": customer_table,
        "part": part_table,
        "orders": orders_table,
        "lineitem": lineitem_table,
    }


def create_test_catalog(scale: str = "tiny", seed: int = 42) -> Catalog:
    """Builds in-memory PyArrow tables and registers them in a Catalog instance."""
    tables = generate_dataset(scale=scale, seed=seed)
    catalog = Catalog()
    for name, table in tables.items():
        catalog.register_table(name, table)
    return catalog


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import pyarrow.parquet as pq

    parser = argparse.ArgumentParser(description="Generate star-schema dataset")
    parser.add_argument("--scale", choices=["tiny", "bench"], default="tiny")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    ds = generate_dataset(scale=args.scale, seed=args.seed)
    print(f"Generated {args.scale} dataset with {len(ds)} tables:")
    for tbl_name, tbl in ds.items():
        print(f"  {tbl_name}: {tbl.num_rows} rows")
        if args.out_dir:
            out_p = Path(args.out_dir)
            out_p.mkdir(parents=True, exist_ok=True)
            pq.write_table(tbl, out_p / f"{tbl_name}.parquet")
            print(f"  Saved to {out_p / f'{tbl_name}.parquet'}")

