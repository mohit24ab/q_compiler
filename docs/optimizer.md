# The optimizer

`optimizer.optimize(plan, catalog)` takes an unoptimized plan from the frontend and returns
`(optimized_plan, traces)` (Contract §5). It runs four rewrite passes to a fixed point, guided by
a cardinality estimator and a cost model built only on the catalog's statistics (Contract §6).
Every pass application is recorded as a `PassTrace`, and `optimizer.render_traces` prints them as
before/after plan diffs.

## What it buys, measured

Person C's generated code ran all 65 queries of the optimizer's differential suite on tables
about 50 times the suite's size (2,000 orders, 4,000 lineitems, 200 customers, and so on), once
per pipeline configuration. The total is the sum over all queries of the fastest of 5 runs, not
counting optimization, code generation or compilation (`docs/ablation/codegen.md`):

| configuration | total runtime | slower than all passes | without the comma join | values read from tables | wrong results |
|---|--:|--:|--:|--:|--:|
| all passes | 32.0 ms | — | — | 342,357 | 0 |
| without predicate pushdown | 303.7 ms | 9.5x | 1.35x | 342,357 | 0 |
| without join reordering | 47.7 ms | 1.5x | 1.44x | 342,357 | 0 |
| without column pruning | 41.6 ms | 1.3x | 1.33x | 1,376,033 | 0 |
| without constant folding | 33.5 ms | 1.05x | 1.07x | 380,957 | 0 |
| no passes (the plans as written) | 513.2 ms | 16.0x | 2.24x | 1,688,033 | 0 |

One query dominates the first column. `comma_join_with_a_cross_product` lists its tables in an
order that makes a cross product, and the unoptimized plan runs it as one: it takes 87% of the
unoptimized total. The "without the comma join" column is the same study over the other 64
queries. There the passes make the suite 2.24 times faster, and each pass is worth between 1.07x
and 1.44x.

The raw numbers, one row per query and configuration, are in `docs/ablation/codegen.csv` for
the report's charts. `docs/ablation/reference.md` repeats the study on the tests' reference
evaluator and the suite's own tables. It is a naive row-at-a-time interpreter, so the absolute
numbers mean less, but the ranking of the passes is the same.

## The pipeline

### The fixed-point loop (`manager.py`)

`OptimizerManager` applies its passes in order, then again, until an iteration changes nothing.
It compares each iteration's result with every earlier iteration's: if the plan returns to a
state it has been in, two passes are undoing each other, and it stops with an
`OscillationWarning` naming them, rather than looping forever. It also stops at an iteration
cap (10) with an `IterationCapWarning`. Passes must return a new plan and never change their
input. The manager fingerprints the input before each pass and raises `PlanMutationError` if a
pass changed it.

### Pass order

The default order is **predicate pushdown → join reordering → column pruning → constant
folding**. It was chosen by measurement: `optimizer.ablation.pass_order_study` runs the suite
through all 24 orders (`docs/ablation/pass_order.md`). The top of that table:

| order | final plans' estimated cost | pass applications | plans differing from the default |
|---|--:|--:|--:|
| **push → reorder → prune → fold (default)** | 1.47 ms | 540 | 0 |
| push → reorder → fold → prune | 1.47 ms | 604 | 1 |
| fold → push → reorder → prune | 1.47 ms | 604 | 1 |
| reorder → push → prune → fold | 1.48 ms | 540 | 3 |
| *every order with prune before reorder* | 1.92 ms | 544 to 716 | 9 to 15 |

No order oscillates or hits the cap. What the study shows about each interaction:

* **Reordering before pruning matters most.** Pruning narrows join outputs by inserting a
  Project between joins, and a Project ends the tree of inner joins that reordering works on.
  Every order that prunes first ends 30% more expensive.
* **Pushdown before reordering.** Reordering places a conjunct only if it is a join condition,
  and pushdown is what turns a WHERE clause above a chain of joins into join conditions.
  Orders that reorder first still reach good plans, because the loop runs reordering again on
  the next iteration, but on that iteration pruning's Projects have already split some trees
  (3 to 4 plans differ). The extreme case: without pushdown at all, the comma join below stays
  a cross product and runs 183 times slower.
* **Folding last, not first.** The textbook advice is to fold before pushing down, so that
  pushdown sees simplified predicates. In a fixed-point loop that only changes which iteration
  the folding happens in: the final plans have the same cost either way. Folding last has one
  real advantage. It removes the no-op Projects that pruning leaves behind (a Project over a
  Scan narrowed to exactly its columns), so the loop doesn't need an extra iteration to clean
  them up. That saves 64 pass applications over the suite.

## Predicate pushdown (`predicate_pushdown.py`)

**Rule.** Split every predicate into AND-conjuncts and push each one as far down as its column
references allow. A conjunct that reaches a Scan becomes part of `Scan.pushed_predicate`, which
codegen evaluates while reading.

**Preconditions.**

* **Sort:** always crossed.
* **Limit:** never crossed (filtering first selects different rows).
* **Project:** crossed when the conjunct's columns are uniquely named outputs; the reference is
  rewritten as the expression behind it.
* **Aggregate:** crossed only by a conjunct on group keys alone (not a HAVING on an aggregate),
  and never through a global aggregate, which returns a row even for empty input.
* **Inner join:** ON and WHERE conjuncts are pooled, and each goes to the side whose columns it
  reads.
* **LEFT join:** a WHERE conjunct may go only to the preserved (left) side, and an ON conjunct
  only to the null-producing (right) side. If a WHERE conjunct can never be TRUE for a
  NULL-extended row, the LEFT join is converted to INNER first.
* **Constants:** a conjunct that reads no columns is never moved; folding handles those.
* **Ambiguity:** a conjunct whose columns can't be resolved unambiguously stays where it is.

**Example** (`left_join_null_rejecting_where_becomes_inner`):

```
  Project[c_name, o_total]
-   Filter[o_total > 200.0]
-     Join[kind=left, cond=c_id = o_custkey]
-       Scan[customer]
-       Scan[orders]
+   Join[kind=inner, cond=c_id = o_custkey]
+     Scan[customer]
+     Scan[orders, pushed=o_total > 200.0]
```

**Measured contribution.** Turning it off makes the suite 9.5 times slower, almost all of it in
the comma join (183x: the WHERE clause stays above the cross product). Over the other 64 queries
it is worth 1.35x, and up to 3.1x on the nine-way chain and 2.2x on the five-way join. Rows
filtered at the scan never reach a join. It doesn't change how many values are read, because the
scan reads every row of its columns anyway; it changes how much work follows the read.

## Join reordering (`join_reordering.py`)

**Rule.** For each maximal tree of inner joins, choose the cheapest join tree over the same
leaves. The join graph has a vertex per leaf, and an edge per conjunct that reads two or more
leaves.

* **Up to 8 leaves:** dynamic programming over connected subsets (Selinger's method, extended
  to bushy trees) finds the cheapest tree under the cost model.
* **More than 8 leaves:** a greedy heuristic repeatedly joins the pair with the smallest
  estimated output.
* **Cross products:** considered only when the graph is disconnected.
* **Placement:** every conjunct goes into the lowest join that sees all the tables it reads.

**Preconditions.** A region is left alone when:

* it has fewer than 3 leaves;
* a column reference matches more than one leaf, or none;
* nothing above it names its outputs (no Project or Aggregate), so its column order is part of
  the query's result;
* the best tree isn't at least 1% cheaper than the current one.

A LEFT join is never part of a region: it is a leaf, reordered around but never through.

**Example** (`comma_join_with_a_cross_product`: `FROM orders, nation, customer WHERE ...`,
whose FROM order crosses orders with nation):

```
  Project[o_id, n_name]
-   Join[kind=inner, cond=(true AND (o_custkey = c_id)) AND (c_nationkey = n_id)]
-     Join[kind=inner, cond=true]
-       Scan[orders]
+   Join[kind=inner, cond=o_custkey = c_id]
+     Scan[orders]
+     Join[kind=inner, cond=c_nationkey = n_id]
+       Scan[customer]
        Scan[nation]
-     Scan[customer]
```

**Measured contribution.** 1.5x over the whole suite, and 1.44x without the comma join, the
largest of any single pass there. On individual queries: the five-way join written fact-table-first
runs 3.8x faster, the nine-way chain (planned greedily) 2.8x, the comma join 2.6x, and the
three-table conjunct 2.3x. Most suite queries join one or two tables, where there is nothing to
reorder.

**How it is checked.** On 40 random join graphs (chains, stars, cycles, cliques, random trees)
the dynamic program's plan costs exactly as much as the cheapest of every possible tree. A
4-table star query has a known best order, and the DP finds it.

That check exposed a flaw in the estimator. Join outputs capped each column's distinct-value
count by the row count, so the same set of tables got estimates from 5 to 55 rows depending
on join order, and the DP missed the cheapest tree. Joins now keep the full distinct-value
count, and a permanent test checks that estimates do not depend on join order.

## Column pruning (`column_pruning.py`)

**Rule.** Walk the plan top-down, collecting the columns each node's parent needs, and narrow
every `Scan.columns` to those. Unused Project expressions and unused aggregates are dropped,
and a narrowing Project is inserted above a join whose output is wider than what is used.

**Preconditions.**

* **What counts as needed:** a column read only by a join condition, a filter, a sort key or a
  pushed predicate is still needed.
* **Zero columns:** a node that would be left with none keeps its narrowest column (or
  `count(*)`), so the row count survives.
* **Qualified references:** no Project is inserted where it would hide a column that a
  qualified reference above still names.
* **Unknown nodes:** a node kind the pass doesn't know gets all its children's columns.

**Example** (`two_of_twenty`: two of the 20 columns of `sales`):

```
  Project[sale_id, amount]
-   Scan[sales]
+   Scan[sales, columns=[sale_id, amount]]
```

**Measured contribution.** Without it, the generated code reads 4.0 times as many values
(1.38 million instead of 342,000) and the suite runs 1.3 times slower. Queries over the 20-column
`sales` table gain the most: up to 14x.

## Constant folding (`constant_folding.py`)

**Rule.**

* **Literal arithmetic:** `2 * 3` becomes `6`.
* **Literal comparisons**, dates included, become TRUE or FALSE.
* **Boolean identities:** `x AND TRUE` becomes `x`, `x OR TRUE` becomes `TRUE`, and `NOT NOT x`
  becomes `x`.
* **Contradictions:** `x = 1 AND x = 2` and `x > 5 AND x < 3` become FALSE.
* **Plan simplifications:** a TRUE filter is removed, `LIMIT 0` returns nothing, and a no-op
  Project is removed.
* **Empty results:** a FALSE predicate becomes an empty-result marker (`Filter[false]`), lifted
  as high as the result stays empty. Codegen then runs nothing below it.

**Preconditions.**

* **Exact under SQL's three-valued logic everywhere:** `NULL AND x` is not FALSE in a SELECT
  list.
* **Predicate-only rules:** rules that treat NULL like FALSE apply only in WHERE, ON and pushed
  predicates.
* **Left unfolded:** integer division, division by zero, `%` and overflow, whose runtime meaning
  isn't pinned down.
* **Where the empty marker stops:** at a global aggregate (`COUNT(*)` of nothing is one row)
  and at the right side of a LEFT join (its left rows survive).

**Example** (`false_filter_over_join_and_sort`: `... WHERE 1 = 0 ORDER BY o_id`):

```
- Sort[keys=o_id ASC]
-   Project[o_id, c_name]
-     Filter[1 = 0]
+ Filter[false]
+   Sort[keys=o_id ASC]
+     Project[o_id, c_name]
        ...the join below is never run
```

**Measured contribution.** Small in total (1.05x), because few queries have constants to fold.
Where it applies, it beats every other pass: the empty-result queries run 15 to 130 times faster,
because their joins and scans never run. Without folding, the suite reads 20,400 more rows.

## Cost model and statistics

**Cardinality estimation** (`stats.py`) is textbook System R:

* **Equality:** 1/ndv.
* **Ranges:** interpolated over [min, max].
* **AND** multiplies, **OR** uses inclusion–exclusion, and **NOT** is pushed into the predicate.
* **Joins:** |L|·|R| / max(ndv).
* **Statistics flow up the plan**, so estimates above filters and joins use narrowed values.

`docs/cardinality_estimates.md` compares estimated and actual rows for every node of every
query:

* **Where the data meets the estimator's assumptions:** the median error is 1.03x and the worst
  1.43x.
* **Where it breaks one:** skewed values are off by up to 13x and correlated columns by up to
  20x. The catalog has no histograms or multi-column statistics, so these are known limits, and
  the report shows them.

**The cost model** (`cost.py`) costs scan I/O, predicate evaluation, hash-join build and probe,
join output, aggregation and sorting separately.

* **Units:** each weight is a measured time per row for the numpy and Python code that Person
  C's code generator emits (`calibration.py`), so a cost reads as a predicted runtime.
* **Prediction:** over the 390 ablation runs, estimated cost and measured runtime have a rank
  correlation of 0.97.
* **Choosing between configurations:** for two configurations of the same query whose runtimes
  differ by more than 10%, the model picks the faster one 92% of the time.

## How correctness is checked

An optimizer that returns wrong answers quickly is worth nothing (Contract §7), so every rewrite
is checked against execution, not only against expected plan shapes.

* **The differential suite** (`tests/opt_query_suite.py`, 65 queries) runs through the full
  pipeline, with oscillation and the iteration cap counted as failures, and through each pass on
  its own. Every result is compared with the reference evaluator.
* **The tables are built to catch mistakes:**
  * NULLs in measured columns;
  * orders whose customers don't exist;
  * customers with no orders;
  * column names shared by two tables;
  * a region with no nations.
* **"Must not rewrite" queries** pin each illegal rewrite: pushing through a LIMIT, pushing into
  the null-producing side of a LEFT join, folding `NULL AND x` in a SELECT list, and so on.
* **Negative controls** show the harness fails when a rewrite is broken on purpose.
* **End to end** (`tests/test_opt_end_to_end.py`): where Person C's code is present, every suite
  query's optimized plan is compiled and run, and its result is compared with Person C's
  interpreter running the original plan. All 65 pass. The ablation study's 390 runs on the
  larger tables returned no wrong results.
* **Mutation testing:** in phases B4 to B6, likely bugs were put back into the passes one at a
  time (8 in folding, 20 in the estimator and cost model, 14 in join reordering). Every one is
  caught. Several were caught only after the first run showed a gap in the tests, and those gaps
  were closed.

## Known gaps

* **Estimation.** There are no histograms or most-common-values lists, so skew is invisible; no
  multi-column statistics, so correlation is invisible; and HAVING on aggregates gets a default
  selectivity.
* **Folding.** No arithmetic identities (`x + 0`, `x * 1`), because they can change a value's
  type, and `x * 0` is wrong for NULL. No contradiction detection between two columns or over IN
  lists. No folding of integer division.
* **Join reordering.** Reordering never moves a join across a LEFT join. No cross product is
  ever introduced, even when one would be cheaper (two tiny dimensions crossed first in a star
  query); that is the policy this phase asked for.
* **Column pruning.** It doesn't insert a Project where that would hide a column from a
  qualified reference above, because how qualifiers survive a Project isn't settled between the
  frontend and codegen.
* **Cost model.** The weights fit Person C's current operators. In particular, the hash join's
  Python loops make it slower than a vectorized nested loop on very small inputs, and the cost
  model reflects that. It ignores fixed per-operator overhead, so the smallest queries are
  dominated by noise.

## Files

| file | contents |
|---|---|
| `optimizer/__init__.py` | `optimize`, the default pipeline and why it is in this order |
| `optimizer/manager.py`, `pass_base.py`, `trace.py` | the fixed-point loop, the pass protocol, trace diffs |
| `optimizer/predicate_pushdown.py` | predicate pushdown |
| `optimizer/join_reordering.py` | join reordering |
| `optimizer/column_pruning.py` | column pruning |
| `optimizer/constant_folding.py` | constant folding and the empty-result marker |
| `optimizer/columns.py`, `expressions.py` | column references, name resolution, conjuncts, NULL rejection |
| `optimizer/stats.py` | cardinality estimation |
| `optimizer/cost.py`, `calibration.py` | the cost model, and the benchmarks behind its weights |
| `optimizer/cardinality_report.py` | estimated vs actual rows, for `docs/cardinality_estimates.md` |
| `optimizer/ablation.py` | the ablation runner and the pass order study |
| `tests/opt_ablation_run.py` | regenerates `docs/ablation/` |
| `tests/opt_cardinality_table.py` | regenerates `docs/cardinality_estimates.md` |
