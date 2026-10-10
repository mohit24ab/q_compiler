"""Join reordering (optimizer/join_reordering.py).

Plan tests: a hand-built 4-table star whose best order is known; an
exhaustive search as the oracle for the dynamic program on random join
graphs; the greedy fallback past the cap; conjunct placement; and every case
the pass must leave alone. Correctness: the star with real rows, and (in
test_opt_differential.py) the whole suite through the full pipeline.
"""

import itertools
import random
import warnings

import pytest

import opt_ir  # noqa: F401
import opt_query_suite as S
import optimizer
from ir.dtype import DType
from ir.nodes import Aggregate, Filter, Join, Limit, Scan, Sort
from opt_query_suite import SuiteCatalog, agg, col, join, keep, lit, op, project, scan
from opt_reference_eval import assert_equivalent
from optimizer.columns import column_refs
from optimizer.cost import CostModel
from optimizer.expressions import TRUE, conjoin, split_conjuncts
from optimizer.join_reordering import DP_LIMIT, JoinReordering, build_graph, plan_joins
from optimizer.predicate_pushdown import PredicatePushdown

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def joins(plan):
    """Every Join node, in pre-order."""
    out = [plan] if isinstance(plan, Join) else []
    for c in plan.children:
        out += joins(c)
    return out


def leaves_of(plan):
    if isinstance(plan, Join) and plan.kind == "inner":
        return leaves_of(plan.left) + leaves_of(plan.right)
    return [plan]


def _nodes(plan):
    yield plan
    for c in plan.children:
        yield from _nodes(c)


def tables_under(plan):
    if isinstance(plan, Scan):
        return {plan.table}
    return set().union(*(tables_under(c) for c in plan.children))


def region_conjuncts(plan):
    if isinstance(plan, Join) and plan.kind == "inner":
        own = [p for p in split_conjuncts(plan.condition) if p != TRUE]
        return own + region_conjuncts(plan.left) + region_conjuncts(plan.right)
    return []


def all_trees(graph, model):
    """Every join tree without a cross product, by brute force: the oracle for the DP."""
    n = len(graph.leaves)

    def trees(mask):
        members = [i for i in range(n) if mask >> i & 1]
        if len(members) == 1:
            own = [p for m, p in graph.conjuncts if m == mask]
            leaf = graph.leaves[members[0]]
            yield Filter(child=leaf, predicate=conjoin(own)) if own else leaf
            return
        for r in range(1, len(members)):
            for left_members in itertools.combinations(members, r):
                left = sum(1 << i for i in left_members)
                right = mask ^ left
                if not graph.linked(left, right):
                    continue
                conds = [p for m, p in graph.conjuncts
                         if m & (m - 1) and not m & ~mask and m & ~left and m & ~right]
                for lt in trees(left):
                    for rt in trees(right):
                        yield Join(left=lt, right=rt, condition=conjoin(conds) or TRUE, kind="inner")

    return list(trees((1 << n) - 1))


# --------------------------------------------------------------------------
# A 4-table star with a known best order
# --------------------------------------------------------------------------

# fact: 400 rows referencing d1 (200 rows), d2 (40) and d3 (10). The query keeps
# one row of d1, so joining fact to d1 first shrinks every later join to ~2 rows.
STAR_SCHEMAS = {
    "fact": [("f_id", DType.INT), ("f_d1", DType.INT), ("f_d2", DType.INT), ("f_d3", DType.INT)],
    "d1": [("d1_id", DType.INT), ("d1_name", DType.STRING)],
    "d2": [("d2_id", DType.INT), ("d2_name", DType.STRING)],
    "d3": [("d3_id", DType.INT), ("d3_name", DType.STRING)],
}
_rng = random.Random(4)
STAR_TABLES = {
    "fact": (["f_id", "f_d1", "f_d2", "f_d3"],
             [(i, _rng.randint(1, 200), _rng.randint(1, 40), _rng.randint(1, 10)) for i in range(400)]),
    "d1": (["d1_id", "d1_name"], [(i, f"one#{i}") for i in range(1, 201)]),
    "d2": (["d2_id", "d2_name"], [(i, f"two#{i}") for i in range(1, 41)]),
    "d3": (["d3_id", "d3_name"], [(i, f"three#{i}") for i in range(1, 11)]),
}
STAR = SuiteCatalog(STAR_TABLES, STAR_SCHEMAS)


def star_query():
    """SELECT f_id, d2_name, d3_name FROM fact JOIN d2 ON f_d2 = d2_id JOIN d3 ON f_d3 = d3_id
    JOIN d1 ON f_d1 = d1_id WHERE d1_name = 'one#7'   -- written with the selective join last"""
    d1 = scan("d1", pushed=op("=", col("d1_name"), lit("one#7")))
    j = join(scan("fact"), scan("d2"), op("=", col("f_d2"), col("d2_id")))
    j = join(j, scan("d3"), op("=", col("f_d3"), col("d3_id")))
    j = join(j, d1, op("=", col("f_d1"), col("d1_id")))
    return project(j, *keep("f_id", "d2_name", "d3_name"))


def test_star_joins_the_selective_dimension_first():
    plan = JoinReordering().apply(star_query(), STAR)
    bottom = joins(plan)[-1]
    assert tables_under(bottom) == {"fact", "d1"}
    assert plan != star_query()


def test_star_dp_matches_exhaustive_search():
    region = star_query().child
    graph = build_graph(leaves_of(region), region_conjuncts(region), STAR)
    model = CostModel(STAR)
    best = plan_joins(graph, model)
    trees = all_trees(graph, model)
    assert len(trees) > 1
    assert best.cost == pytest.approx(min(model.cost(t).total for t in trees))
    assert best.algorithm == "dp"


def test_star_returns_the_same_rows():
    plan = star_query()
    assert_equivalent(plan, JoinReordering().apply(plan, STAR), STAR_TABLES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        final, _ = optimizer.optimize(plan, STAR)
    assert_equivalent(plan, final, STAR_TABLES)


# --------------------------------------------------------------------------
# The DP against exhaustive search, on random join graphs
# --------------------------------------------------------------------------


class _StatsCatalog:
    """Statistics only: n tables t0..t<n-1>, each with a key column per table."""

    def __init__(self, n, rng):
        self.n = n
        self.rows = {f"t{i}": rng.choice([5, 50, 500, 5000]) for i in range(n)}
        self.ndv = {(f"t{i}", f"c{i}_{j}"): rng.randint(1, self.rows[f"t{i}"])
                    for i in range(n) for j in range(n)}

    def schema(self, table):
        i = int(table[1:])
        return [(f"c{i}_{j}", DType.INT) for j in range(self.n)]

    def row_count(self, table):
        return self.rows[table]

    def stats(self, table, column):
        return S.ColumnStats(ndv=self.ndv[(table, column)], min=0, max=10_000, null_count=0)


def _random_graph(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 5)
    catalog = _StatsCatalog(n, rng)
    shape = rng.choice(["chain", "star", "cycle", "clique", "random"])
    if shape == "chain":
        edges = [(i, i + 1) for i in range(n - 1)]
    elif shape == "star":
        edges = [(0, i) for i in range(1, n)]
    elif shape == "cycle":
        edges = [(i, (i + 1) % n) for i in range(n)]
    elif shape == "clique":
        edges = list(itertools.combinations(range(n), 2))
    else:
        edges = [(i, rng.randrange(i)) for i in range(1, n)]  # a random tree
    conds = [op("=", col(f"c{a}_{b}"), col(f"c{b}_{a}")) for a, b in edges]
    leaves = [Scan(table=f"t{i}", columns=None, pushed_predicate=None) for i in range(n)]
    return build_graph(leaves, conds, catalog), catalog, shape


@pytest.mark.parametrize("seed", range(40))
def test_dp_finds_the_cheapest_tree(seed):
    graph, catalog, _ = _random_graph(seed)
    model = CostModel(catalog)
    best = plan_joins(graph, model)
    assert best.cost == pytest.approx(min(model.cost(t).total for t in all_trees(graph, model)))
    assert not any(j.condition == TRUE for j in joins(best.plan))  # connected: no cross product


@pytest.mark.parametrize("seed", range(40))
def test_estimates_do_not_depend_on_join_order(seed):
    """The DP is only exact if a set of relations has one estimate, however it was joined."""
    graph, catalog, _ = _random_graph(seed)
    model = CostModel(catalog)
    by_set = {}

    def walk(node):
        if isinstance(node, Join):
            key = frozenset(leaf.table for leaf in leaves_of(node))
            by_set.setdefault(key, []).append(model.estimator.rows(node))
            for child in node.children:
                walk(child)

    for tree in all_trees(graph, model):
        walk(tree)
    for estimates in by_set.values():
        assert max(estimates) == pytest.approx(min(estimates))


@pytest.mark.parametrize("seed", range(40))
def test_greedy_is_valid_and_never_beats_the_dp(seed):
    graph, catalog, _ = _random_graph(seed)
    model = CostModel(catalog)
    greedy = plan_joins(graph, model, dp_limit=1)
    assert greedy.algorithm == "greedy"
    assert sorted(map(repr, region_conjuncts(greedy.plan))) == sorted(repr(p) for _, p in graph.conjuncts)
    assert greedy.cost >= plan_joins(graph, model).cost - 1e-6


# --------------------------------------------------------------------------
# The greedy fallback past the cap
# --------------------------------------------------------------------------


def _chain_graph(n):
    query = next(q for q in S.QUERIES if q.name == "nine_way_chain").plan
    region = query.child.child
    leaves = leaves_of(region)[:n]
    conds = [op("=", col(f"k{i - 1}_next"), col(f"k{i}_id")) for i in range(2, n + 1)]
    return build_graph(leaves, conds, S.CATALOG)


def test_dp_up_to_the_cap():
    assert DP_LIMIT == 8
    assert plan_joins(_chain_graph(8), CostModel(S.CATALOG)).algorithm == "dp"


def test_greedy_past_the_cap():
    result = plan_joins(_chain_graph(9), CostModel(S.CATALOG))
    assert result.algorithm == "greedy"
    assert len(leaves_of(result.plan)) == 9
    assert not any(j.condition == TRUE for j in joins(result.plan))


def test_greedy_joins_the_smallest_result_first():
    # in the star, fact JOIN d1 has the smallest output (d1 keeps one row), so
    # greedy makes it first: it is the bottom join of the tree it builds
    region = star_query().child
    graph = build_graph(leaves_of(region), region_conjuncts(region), STAR)
    greedy = plan_joins(graph, CostModel(STAR), dp_limit=1)
    assert greedy.algorithm == "greedy"
    assert tables_under(joins(greedy.plan)[-1]) == {"fact", "d1"}


class _TwoTinyDimensions:
    """fact (100,000 rows) references d1 and d2, which have one row each."""

    def schema(self, table):
        return {"fact": [("f_d1", DType.INT), ("f_d2", DType.INT)],
                "d1": [("d1_id", DType.INT)], "d2": [("d2_id", DType.INT)]}[table]

    def row_count(self, table):
        return {"fact": 100_000, "d1": 1, "d2": 1}[table]

    def stats(self, table, column):
        ndv = 10 if table == "fact" else 1
        return S.ColumnStats(ndv=ndv, min=0, max=10, null_count=0)


def test_no_cross_product_even_when_one_would_be_cheaper():
    # The classic star-query exception: d1 x d2 is a single row, and probing
    # the fact table once with it beats two joins. The pass still refuses
    # cross products the graph does not force (one bad estimate on a cross
    # product multiplies, and it is the rule the phase asks for).
    cat = _TwoTinyDimensions()
    fact, d1, d2 = (Scan(table=t, columns=None, pushed_predicate=None) for t in ("fact", "d1", "d2"))
    k1, k2 = op("=", col("f_d1"), col("d1_id")), op("=", col("f_d2"), col("d2_id"))
    model = CostModel(cat)
    crossed = Join(left=fact, right=Join(left=d1, right=d2, condition=TRUE, kind="inner"),
                   condition=op("AND", k1, k2), kind="inner")
    best = plan_joins(build_graph([fact, d1, d2], [k1, k2], cat), model)
    assert model.cost(crossed).total < best.cost  # the premise: crossing really is cheaper here
    assert not any(j.condition == TRUE for j in joins(best.plan))


# --------------------------------------------------------------------------
# Conjunct placement and cross products
# --------------------------------------------------------------------------


def _optimized(name):
    q = next(q for q in S.QUERIES if q.name == name)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return optimizer.optimize(q.plan, S.CATALOG)[0]


def test_comma_join_loses_its_cross_product():
    before = next(q for q in S.QUERIES if q.name == "comma_join_with_a_cross_product").plan
    assert any(j.condition == TRUE for j in joins(before))
    assert not any(j.condition == TRUE for j in joins(_optimized("comma_join_with_a_cross_product")))


def test_a_disconnected_graph_gets_exactly_one_cross_product():
    plan = _optimized("forced_cross_product")
    crosses = [j for j in joins(plan) if j.condition == TRUE]
    assert len(crosses) == 1
    assert "nation" in tables_under(crosses[0].left) | tables_under(crosses[0].right)


TABLE_OF = {name: t for t in ("orders", "customer", "nation") for name, _ in S.SCHEMAS[t]}


def test_conjuncts_go_to_the_lowest_join_that_sees_their_tables():
    plan = JoinReordering().apply(S.three_relation_conjunct(), S.CATALOG)
    for j in joins(plan):
        below = tables_under(j)
        for p in split_conjuncts(j.condition):
            refs = {TABLE_OF[name] for _, name in column_refs(p)}
            assert refs <= below
            # not placeable lower: neither child sees all of them
            assert not any(refs <= tables_under(c) for c in j.children)


def test_single_table_conjuncts_become_filters_and_constants_stay_on_top():
    a = Scan(table="orders", columns=None, pushed_predicate=None)
    b = Scan(table="customer", columns=None, pushed_predicate=None)
    c = Scan(table="nation", columns=None, pushed_predicate=None)
    odd = op("=", col("o_status"), lit("F"))
    constant = op("=", lit(1), lit(1))
    region = join(join(a, b, op("AND", op("=", col("o_custkey"), col("c_id")), odd)), c,
                  op("AND", op("=", col("c_nationkey"), col("n_id")), constant))
    graph = build_graph(leaves_of(region), region_conjuncts(region), S.CATALOG)
    plan = plan_joins(graph, CostModel(S.CATALOG)).plan
    assert constant in split_conjuncts(plan.condition)
    filters = [leaf for leaf in leaves_of(plan) if isinstance(leaf, Filter)]
    assert [f.predicate for f in filters] == [odd]


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_every_conjunct_survives_exactly_once(query):
    pushed = PredicatePushdown().apply(query.plan, S.CATALOG)
    after = JoinReordering().apply(pushed, S.CATALOG)

    def everything(node):
        own = []
        if isinstance(node, Join):
            own = split_conjuncts(node.condition)
        elif isinstance(node, Filter):
            own = split_conjuncts(node.predicate)
        own = [p for p in own if p != TRUE]
        return own + [p for c in node.children for p in everything(c)]

    assert sorted(map(repr, everything(after))) == sorted(map(repr, everything(pushed)))


# --------------------------------------------------------------------------
# Regions the pass must leave alone
# --------------------------------------------------------------------------


def test_two_relations_are_left_alone():
    plan = S.colliding_names()
    assert JoinReordering().apply(plan, S.CATALOG) is plan


def test_visible_column_order_is_left_alone():
    plan = next(q for q in S.QUERIES if q.name == "join_output_order_is_the_result").plan
    assert JoinReordering().apply(plan, S.CATALOG) is plan
    # the same join under a Project may be reordered
    wrapped = project(plan, *keep("l_id"))
    assert JoinReordering().apply(wrapped, S.CATALOG) is not wrapped


def test_regions_under_a_sort_or_limit_are_left_alone():
    # Reordering changes the order of a join's output rows: under a Sort, the order of
    # rows whose keys tie; under a Limit, which rows are kept. Both are legal SQL, but
    # Contract §7 compares the results of ORDER BY queries row for row.
    region = star_query()
    by_d3 = Sort(child=region, keys=[(col("d3_name"), False)])
    counted = Aggregate(child=region.child, group_keys=[col("d3_name")], aggs=[(agg("count"), "n")])
    for plan in (by_d3, Limit(child=by_d3, n=5), Limit(child=region, n=5),
                 Sort(child=counted, keys=[(col("n"), True)])):
        assert JoinReordering().apply(plan, STAR) is plan
        assert JoinReordering(keep_row_order=False).apply(plan, STAR) is not plan


def _top_five_by_d3_name():
    """SELECT f_id, d2_name, d3_name FROM fact JOIN d2 ... JOIN d3 ... JOIN d1 ...
    WHERE d1_id <= 10 ORDER BY d3_name LIMIT 5   -- many rows tie on d3_name"""
    j = join(scan("fact"), scan("d2"), op("=", col("f_d2"), col("d2_id")))
    j = join(j, scan("d3"), op("=", col("f_d3"), col("d3_id")))
    j = join(j, scan("d1", pushed=op("<=", col("d1_id"), lit(10))), op("=", col("f_d1"), col("d1_id")))
    ordered = Sort(child=project(j, *keep("f_id", "d2_name", "d3_name")), keys=[(col("d3_name"), False)])
    return Limit(child=ordered, n=5)


def test_reordering_under_a_limit_would_keep_other_rows():
    # the negative control for the test above: with keep_row_order off, the LIMIT keeps
    # a different five of the tied rows, and the §7 check fails
    plan = _top_five_by_d3_name()
    reordered = JoinReordering(keep_row_order=False).apply(plan, STAR)
    assert reordered is not plan
    with pytest.raises(AssertionError, match="ordered rows differ"):
        assert_equivalent(plan, reordered, STAR_TABLES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        final, _ = optimizer.optimize(plan, STAR)
    assert_equivalent(plan, final, STAR_TABLES)


def test_a_conjunct_over_three_tables_that_is_their_only_link():
    # d1_id + d2_id = d3_id reads all three tables and no conjunct links two of them, so
    # every plan needs a cross product. The DP found no split into linked halves and
    # raised KeyError; it now plans the region again with cross products allowed.
    region = join(join(scan("d2"), scan("d1", pushed=op("<=", col("d1_id"), lit(5))), TRUE), scan("d3"),
                  op("=", op("+", col("d1_id"), col("d2_id")), col("d3_id")))
    plan = project(region, *keep("d1_name", "d2_name", "d3_name"))
    graph = build_graph(leaves_of(region), region_conjuncts(region), STAR)
    best = plan_joins(graph, CostModel(STAR))
    assert best.algorithm == "dp"
    assert [j.condition == TRUE for j in joins(best.plan)].count(True) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        final, _ = optimizer.optimize(plan, STAR)
    assert_equivalent(plan, final, STAR_TABLES)


def test_ambiguous_references_are_left_alone():
    # `id` names a column of both emp and dept
    region = join(join(scan("emp"), scan("dept"), op("=", col("dept_id"), col("id"))), scan("nation"),
                  op("=", col("id"), col("n_id")))
    plan = project(region, *keep("n_name"))
    assert JoinReordering().apply(plan, S.CATALOG) is plan
    assert build_graph(leaves_of(region), region_conjuncts(region), S.CATALOG) is None


def test_left_joins_are_leaves_not_part_of_the_region():
    plan = _optimized("join_region_inside_left_join")
    left = [j for j in joins(plan) if j.kind == "left"]
    assert len(left) == 1 and isinstance(left[0].left, Scan) and left[0].left.table == "region"
    assert tables_under(left[0].right) == {"orders", "customer", "nation"}


def test_join_trees_inside_leaves_are_reordered_too():
    plan = _optimized("nested_join_regions")
    grouped = next(n for n in _nodes(plan) if isinstance(n, Aggregate))
    inner = joins(grouped)
    bottom = [j for j in inner if len(joins(j)) == 1]  # joins with no join below them
    # the filtered customer table joins orders first, not lineitem with orders
    assert [tables_under(j) for j in bottom] == [{"orders", "customer"}]
    outer_bottom = [j for j in joins(plan) if not any(j is k for k in inner) and len(joins(j)) == 1]
    assert [tables_under(j) for j in outer_bottom] == [{"nation", "region"}]


def test_equal_cost_alternatives_do_not_change_the_plan():
    plan = star_query()
    once = JoinReordering().apply(plan, STAR)
    assert JoinReordering().apply(once, STAR) is once
    lazy = JoinReordering(min_gain=1.0)  # demands a gain no plan can make
    assert lazy.apply(plan, STAR) is plan


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_idempotent_after_the_full_pipeline(query):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final, _ = optimizer.optimize(query.plan, S.CATALOG)
    assert JoinReordering().apply(final, S.CATALOG) is final


@pytest.mark.parametrize("query", S.QUERIES, ids=lambda q: q.name)
def test_never_more_expensive(query):
    model = CostModel(S.CATALOG)
    plan = query.plan
    assert model.cost(JoinReordering().apply(plan, S.CATALOG)).total <= model.cost(plan).total * (1 + 1e-9)


def test_pass_does_not_mutate_its_input():
    plan = star_query()
    before = repr(plan)
    JoinReordering().apply(plan, STAR)
    assert repr(plan) == before


# --------------------------------------------------------------------------
# Negative controls: the differential harness catches broken reorderings
# --------------------------------------------------------------------------


def test_harness_catches_a_dropped_conjunct():
    plan = star_query()
    good = JoinReordering().apply(plan, STAR)
    bottom = joins(good)[-1]
    broken_bottom = Join(left=bottom.left, right=bottom.right, condition=TRUE, kind="inner")

    def swap(node):
        if node is bottom:
            return broken_bottom
        return node if not node.children else node.replace_children(tuple(swap(c) for c in node.children))

    with pytest.raises(AssertionError):
        assert_equivalent(plan, swap(good), STAR_TABLES)


def test_harness_catches_reordering_through_a_left_join():
    # region LEFT JOIN (orders JOIN customer JOIN nation) is not
    # (region LEFT JOIN nation) JOIN (orders JOIN customer): that loses OCEANIA
    plan = S.join_region_inside_left_join()
    region, inner = plan.child.left, plan.child.right
    orders_customer, nation = inner.left, inner.right
    wrong = Join(left=Join(left=region, right=nation, condition=plan.child.condition, kind="left"),
                 right=orders_customer, condition=inner.condition, kind="inner")
    with pytest.raises(AssertionError):
        assert_equivalent(plan, project(wrong, *keep("r_name", "c_name", "o_id")), S.TABLES)
