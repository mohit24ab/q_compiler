import datetime

import numpy as np
import pyarrow as pa
import pytest

from runtime import Table, compare_tables
from runtime._compat import DType

SCHEMA = [("id", DType.INT), ("name", DType.STRING), ("score", DType.FLOAT)]


def make():
    return Table.from_pydict(
        {"id": [1, 2, 3], "name": ["a", None, "c"], "score": [1.5, 2.5, None]},
        SCHEMA, table="t")


def test_physical_types_and_null_masks():
    t = make()
    assert t.num_rows == 3
    assert t.schema == SCHEMA
    assert t.column("id").values.dtype == np.int64
    assert t.column("name").values.dtype == object
    assert t.column("id").valid is None                      # no nulls -> no mask
    assert t.column("name").valid.tolist() == [True, False, True]
    assert t.to_pydict() == {"id": [1, 2, 3], "name": ["a", None, "c"],
                             "score": [1.5, 2.5, None]}


def test_select_slice_take_filter_concat():
    t = make()
    assert t.select(["score", "id"]).column_names == ["score", "id"]
    assert t.slice(1).to_rows() == [(2, None, 2.5), (3, "c", None)]
    assert t.slice(0, 99).num_rows == 3
    assert t.take([2, 0]).to_rows() == [(3, "c", None), (1, "a", 1.5)]
    assert t.filter(np.array([True, False, True])).column("id").to_pylist() == [1, 3]
    both = Table.concat([t, t.slice(0, 1)])
    assert both.num_rows == 4
    assert both.column("name").to_pylist() == ["a", None, "c", "a"]


def test_qualified_lookup_and_ambiguity():
    a = Table.from_pydict({"id": [1]}, [("id", DType.INT)], table="orders")
    b = Table.from_pydict({"id": [9]}, [("id", DType.INT)], table="customer")
    joined = Table(a.columns + b.columns)
    assert joined.column("id", "customer").to_pylist() == [9]
    with pytest.raises(KeyError, match="ambiguous"):
        joined.column("id")


def test_from_arrow_with_nulls_and_dates():
    arrow = pa.table({
        "n": pa.array([1, None, 3], type=pa.int32()),
        "d": pa.array([datetime.date(2024, 1, 1), None, datetime.date(2024, 3, 1)]),
        "s": pa.array(["x", "y", None]),
        "b": pa.array([True, False, True]),
    })
    t = Table.from_arrow(arrow, table="x")
    assert t.schema == [("n", DType.INT), ("d", DType.DATE),
                        ("s", DType.STRING), ("b", DType.BOOL)]
    assert t.to_rows() == [
        (1, datetime.date(2024, 1, 1), "x", True),
        (None, None, "y", False),
        (3, datetime.date(2024, 3, 1), None, True),
    ]


def test_empty_table_round_trip():
    t = Table.from_pydict({"id": [], "name": []}, SCHEMA[:2])
    assert t.num_rows == 0 and t.to_rows() == []


def test_compare_tables_order_and_float_tolerance():
    t = make()
    shuffled = t.take([2, 0, 1])
    assert compare_tables(t, shuffled)[0]
    assert not compare_tables(t, shuffled, ordered=True)[0]
    nudged = Table.from_pydict(
        {"id": [1, 2, 3], "name": ["a", None, "c"], "score": [1.5 + 1e-12, 2.5, None]},
        SCHEMA)
    assert compare_tables(t, nudged, ordered=True)[0]
    ok, why = compare_tables(t, t.slice(0, 2))
    assert not ok and "row counts" in why
