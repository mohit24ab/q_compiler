"""Key analysis (optimizer/keys.py) and the guard it feeds.

Plan tests: the keys the analysis reports for each node kind, and in
particular the one distinction the guard turns on -- a fact table's key
survives its dimension joins, a dimension's key does not survive the fan-out.
Correctness: the shapes join reordering must still refuse, and the Contract §7
differential on a query whose ORDER BY ties.
"""

import random

import pytest

import opt_ir  # noqa: F401
from ir.dtype import DType
from ir.nodes import Limit, Sort
from opt_query_suite import SuiteCatalog, agg, col, join, keep, lit, op, project, scan
from opt_reference_eval import assert_equivalent
from optimizer.join_reordering import JoinReordering
from optimizer.keys import ordering_is_total, unique_keys

# --------------------------------------------------------------------------
# A star whose grain is `fact`: every dimension row matches many fact rows.
# --------------------------------------------------------------------------

SCHEMAS = {
    "fact": [("f_id", DType.INT), ("f_d1", DType.INT), ("f_d2", DType.INT),
             ("f_amount", DType.FLOAT)],
    "d1": [("d1_id", DType.INT), ("d1_name", DType.STRING)],
    "d2": [("d2_id", DType.INT), ("d2_name", DType.STRING)],
}
_rng = random.Random(7)
TABLES = {
    "fact": (["f_id", "f_d1", "f_d2", "f_amount"],
             [(i, _rng.randint(1, 10), _rng.randint(1, 20), float(i)) for i in range(200)]),
    "d1": (["d1_id", "d1_name"], [(i, f"one#{i}") for i in range(1, 11)]),
    "d2": (["d2_id", "d2_name"], [(i, f"two#{i}") for i in range(1, 21)]),
}
STAR = SuiteCatalog(TABLES, SCHEMAS)

FACT_D1 = op("=", col("f_d1", "fact"), col("d1_id", "d1"))
FACT_D2 = op("=", col("f_d2", "fact"), col("d2_id", "d2"))


def key_names(keys):
    """{frozenset of bare column names} -- qualifiers vary and are not the point here."""
    return {frozenset(name for _table, name in k) for k in keys}


# --------------------------------------------------------------------------
# Scans
# --------------------------------------------------------------------------


def test_a_column_with_one_value_per_row_is_a_key():
    keys = unique_keys(scan("fact"), STAR)
    assert frozenset({"f_id"}) in key_names(keys)


def test_a_repeating_column_is_not_a_key():
    keys = key_names(unique_keys(scan("fact"), STAR))
    assert frozenset({"f_d1"}) not in keys


def test_an_unknown_table_defeats_the_analysis():
    assert unique_keys(scan("fact"), None) is None


# --------------------------------------------------------------------------
# Row-preserving nodes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wrap", [
    lambda c: Sort(child=c, keys=[(col("f_amount", "fact"), False)]),
    lambda c: Limit(child=c, n=5),
    lambda c: scan("fact", pushed=op(">", col("f_amount", "fact"), lit(1.0))),
])
def test_keeping_a_subset_of_rows_keeps_them_distinct(wrap):
    """Filtering, sorting and limiting cannot make two rows equal."""
    node = wrap(scan("fact"))
    assert frozenset({"f_id"}) in key_names(unique_keys(node, STAR))


# --------------------------------------------------------------------------
# Project
# --------------------------------------------------------------------------


def test_a_key_survives_projection_under_its_alias():
    p = project(scan("fact"), (col("f_id", "fact"), "renamed"))
    assert frozenset({"renamed"}) in key_names(unique_keys(p, STAR))


def test_a_pass_through_key_keeps_its_own_name_too():
    """`SELECT fact.f_id AS f_id ... ORDER BY fact.f_id` is legal, so both names
    have to name the key -- this is what q02 needs and q10 got by aliasing."""
    p = project(scan("fact"), *keep("f_id", table="fact"))
    assert frozenset({"f_id"}) in key_names(unique_keys(p, STAR))


def test_a_key_that_is_not_projected_does_not_survive():
    p = project(scan("fact"), *keep("f_d1", table="fact"))
    assert key_names(unique_keys(p, STAR)) == set()


def test_a_computed_column_is_not_tracked_as_a_key():
    """f_id * 2 happens to be unique, but the analysis must not assume that."""
    p = project(scan("fact"), (op("*", col("f_id", "fact"), lit(2)), "doubled"))
    assert key_names(unique_keys(p, STAR)) == set()


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------


def test_group_keys_are_a_key_of_the_aggregate():
    from ir.nodes import Aggregate
    a = Aggregate(child=scan("fact"), group_keys=[col("f_d1", "fact")],
                  aggs=[(agg("count"), "n")])
    assert frozenset({"f_d1"}) in key_names(unique_keys(a, STAR))


def test_a_global_aggregate_returns_one_row():
    from ir.nodes import Aggregate
    a = Aggregate(child=scan("fact"), group_keys=[], aggs=[(agg("count"), "n")])
    keys = unique_keys(a, STAR)
    assert frozenset() in keys, "one row means the empty set identifies it"


# --------------------------------------------------------------------------
# Joins -- the distinction the guard rests on
# --------------------------------------------------------------------------


def test_a_fact_key_survives_a_join_to_a_dimension():
    """Each fact row matches one d1 row, so f_id still identifies a result row."""
    j = join(scan("fact"), scan("d1"), FACT_D1)
    assert frozenset({"f_id"}) in key_names(unique_keys(j, STAR))


def test_a_dimension_key_does_not_survive_the_fan_out():
    """d1_name is unique in d1, but many fact rows share each d1 row, so it
    repeats in the join's output. This is the case that must stay guarded."""
    j = join(scan("fact"), scan("d1"), FACT_D1)
    assert frozenset({"d1_name"}) not in key_names(unique_keys(j, STAR))
    assert frozenset({"d1_id"}) not in key_names(unique_keys(j, STAR))


def test_an_outer_join_defeats_the_analysis():
    j = join(scan("fact"), scan("d1"), FACT_D1, kind="left")
    assert unique_keys(j, STAR) is None


# --------------------------------------------------------------------------
# ordering_is_total
# --------------------------------------------------------------------------


def test_an_ordering_on_a_surviving_key_is_total():
    j = join(scan("fact"), scan("d1"), FACT_D1)
    s = Sort(child=j, keys=[(col("f_amount", "fact"), True), (col("f_id", "fact"), False)])
    assert ordering_is_total(s, STAR) is True


def test_an_ordering_on_a_fanned_out_column_is_not_total():
    j = join(scan("fact"), scan("d1"), FACT_D1)
    s = Sort(child=j, keys=[(col("d1_name", "d1"), False)])
    assert ordering_is_total(s, STAR) is False


def test_an_ordering_with_no_unique_key_is_not_total():
    s = Sort(child=scan("fact"), keys=[(col("f_d1", "fact"), False)])
    assert ordering_is_total(s, STAR) is False


# --------------------------------------------------------------------------
# Correctness: what the guard does with those verdicts
# --------------------------------------------------------------------------


def _three_table_region():
    """A region with enough leaves that reordering is on the table at all."""
    j = join(scan("fact"), scan("d1"), FACT_D1)
    return join(j, scan("d2", pushed=op("<=", col("d2_id", "d2"), lit(15))), FACT_D2)


def test_a_tying_ordering_is_still_left_alone():
    """The §7 hazard: under ORDER BY d1_name LIMIT, reordering changes which of
    the tied rows come back. The guard must refuse, total ordering or not."""
    region = _three_table_region()
    plan = project(Limit(child=Sort(child=region, keys=[(col("d1_name", "d1"), False)]), n=5),
                   *keep("d1_name", table="d1"))
    assert JoinReordering().apply(plan, STAR) is plan


def test_a_bare_limit_is_still_left_alone():
    """No ORDER BY at all: which rows a LIMIT keeps depends on the join order."""
    region = _three_table_region()
    plan = project(Limit(child=region, n=5), *keep("f_id", table="fact"))
    assert JoinReordering().apply(plan, STAR) is plan


def test_optimizing_a_tying_top_n_preserves_the_rows(star_tables=TABLES):
    """Contract §7 end to end on the shape that forced the guard to exist."""
    region = _three_table_region()
    plan = project(Limit(child=Sort(child=region, keys=[(col("d1_name", "d1"), False)]), n=5),
                   *keep("d1_name", table="d1"))
    assert_equivalent(plan, JoinReordering().apply(plan, STAR), star_tables)
