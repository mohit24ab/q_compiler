# Code generator and runtime (Person C)

```python
from codegen import generate, compile_and_run
from runtime import interpret

source = generate(plan, catalog)          # Python source text (Contract §5)
result = compile_and_run(source, tables)  # runtime.Table
assert result == interpret(plan, tables)  # in spirit: compare_tables(...) does it properly
```

`generate(plan, catalog=None, mode="auto", fuse=True)`

| argument | meaning |
|---|---|
| `mode="auto"` | compile to numpy (every plan node compiles since C4) |
| `mode="compiled"` | same, but raise instead of falling back |
| `mode="passthrough"` | emit a module that rebuilds the plan and calls the interpreter (a baseline) |
| `fuse=True` | fuse `[Project] <- Filter* <- Scan` chains into one pass (C5, below) |
| `fuse=False` | one section per operator, so the benchmark can isolate what fusion is worth |

`catalog` only needs `schema(table)`, and only for a `Scan` whose `table_schema` is `None`.

## Reading generated code

Every module looks the same: a docstring holding the plan it came from, imports, and one
function `run(tables)` that returns a `runtime.Table`. It needs nothing else; you can
`exec` it in an empty namespace.

Every column is **two arrays**: `x` (values) and `x_ok` (`True` where the value is not
NULL). NULL slots hold a typed filler value (`0`, `''`, ...) that is never read without its
mask. When a column is known to be non-NULL its `ok` is the constant `True`, and the
mask algebra folds it away at generation time.

The semantics are copied from `runtime/expr_eval.py`, the interpreter, which is the spec:

* every binary operation is parenthesised, unconditionally;
* `/` is always FLOAT; `/` and `%` by zero give NULL; `%` truncates toward zero (`np.fmod`);
* `AND`/`OR` are Kleene logic: a known FALSE decides AND, a known TRUE decides OR;
* a WHERE keeps a row only when the predicate is TRUE (FALSE and NULL are both dropped);
* aggregates skip NULLs; `count(*)` counts rows; `avg` is `sum / count` at the end;
  a global aggregate over no rows still returns one row;
* NULL group keys form one group; groups come out in first-seen order;
* sorting is stable with NULLs last in both directions;
* join output is left-row order, then right-row order.

When generated code fails, `compile_and_run` raises `GeneratedCodeError` with the numbered
source and a `-->` on the failing line (the innermost generated frame, even when the
exception happened inside a numpy call).

## Operators

| operator | emitted code |
|---|---|
| `Scan` | reads columns lazily, only when something uses them. A pushed predicate reads its own columns first, builds a mask, then reads every other column already filtered (`read_column(t, c, rows=mask)`). A pushed predicate may read columns the Scan doesn't output. |
| `Filter` | one mask, applied to every column it was given (unfused). |
| `Project` | a plain column reference is free (it reuses the arrays under a new name); a computed one is one numpy expression. |
| `Join` | equality conjuncts with one side per input become hash keys: a dict is built on the smaller input and probed with the larger. NULL keys never match. Everything else is a residual, checked on the candidate pairs. A condition with no keys becomes all pairs (`np.repeat`/`np.tile`). LEFT JOIN pads unmatched left rows with NULLs. A constant-FALSE inner join runs neither input. |
| `Aggregate` | hash aggregation: every key tuple gets a dense group id (first-seen order), then vectorised per-group accumulators (`np.bincount`, `np.add.at`, `np.minimum.at`). Without GROUP BY it is a separate, simpler path of plain reductions. |
| `Sort` | each key becomes a rank (`np.unique` inverse, negated for DESC, NULLs pushed past the end), then one stable `np.lexsort`. |
| `Limit` | a slice. |

A `Filter` or pushed predicate that is literally `FALSE` or `NULL` emits an empty table
with the right layout, and the operators below it never run.

## Operator fusion (Phase C5)

### What fuses

A chain fuses when it is **`[Project] <- Filter* <- Scan`** and has at least two operators.
A pushed predicate on the Scan counts as one more filter.

| chain | fused? | what the fused pass does |
|---|---|---|
| `Project <- Filter <- Scan` | yes | one mask, gather only the columns the Project reads, compute |
| `Project <- Filter <- Filter <- ... <- Scan` | yes | every Filter ANDed into the same single mask |
| `Filter* <- Scan[pushed=p]` | yes | `p` joins the same mask |
| `Filter <- Scan` | yes | one mask; columns the predicate didn't read are read straight from the table with `rows=mask` |
| `Project <- Scan` | yes | reads only the columns the Project references (unfused reads every Scan column) |
| `Scan` alone (even with a pushed predicate) | no | it already is a single pass |
| `Filter <- Project <- ...` | no | the Filter reads the Project's output names; fusing would need expression substitution. The `Project <- Scan` underneath still fuses on its own. |
| `Project <- Project <- ...` | no | same reason |
| anything above a `Join`, `Aggregate`, `Sort` or `Limit` (e.g. HAVING) | no | those materialise their output anyway; their own inputs still fuse |

Fusion is applied wherever such a chain appears, including as the input of a Join or an
Aggregate (`codegen.operators.fusable_chain` decides).

### The fused pass

1. Read only the columns some predicate needs, at full length.
2. AND every predicate (the pushed one, then each Filter bottom-up) into **one** mask.
3. Apply that mask **once**, and only to the columns the output needs. A column a
   predicate already read is gathered (`x[keep]`); any other column is read from the table
   already filtered (`rows=keep`), so no unfiltered copy of it ever exists.
4. Compute the Project's expressions over the filtered columns.

Evaluating every predicate on every row gives the same answer as stacking them: a row
passes `WHERE a` then `WHERE b` iff it passes `a AND b` (the mask keeps only TRUE, so
NULL is dropped either way, and evaluating an expression has no side effects here:
division by zero is NULL, not an exception).

**Exception, LIKE.** `LIKE` is the one predicate that compiles to a Python loop per row
(`runtime.vec.like`), not a numpy operation. Evaluated on every row it made one fused
chain 11x *slower* than unfused, where it only ever saw the rows that survived the Filter
below it. So predicates containing LIKE go last and run only on the rows the vectorised
predicates kept: `alive = np.flatnonzero(keep)`, then `keep[alive] = <LIKE over those rows>`.
It is still one mask and one gather. If `keep` would otherwise be a column's own array
(e.g. `x IS NOT NULL` compiles to `x_ok`), it is copied first so the in-place update can
never write into an input table.

### Example: `SELECT id, amount * 2 FROM sales WHERE amount > 15 AND region = 'US' AND qty < 6`

The plan: one predicate pushed into the Scan, two stacked Filters, a Project.

**Fused** (`generate(plan, catalog)`), one pass:

```python
def run(tables):
    # Fused pipeline: 4 operators, one pass over sales
    #   Project[id, amount * 2 AS double_amount]
    #     Filter[qty < 6]
    #       Filter[region = 'US']
    #         Scan[sales, pushed=amount > 15]
    sales_1 = as_table(tables['sales'])
    # every predicate, over just the columns it reads, ANDed into one mask
    # pushed into Scan[sales]: (amount > 15)
    amount_1, amount_1_ok = read_column(sales_1, 'amount')
    # Filter: (region = 'US')
    region_1, region_1_ok = read_column(sales_1, 'region')
    # Filter: (qty < 6)
    qty_1, qty_1_ok = read_column(sales_1, 'qty')
    keep_1 = ((amount_1 > 15) & amount_1_ok) & ((region_1 == 'US') & region_1_ok) & ((qty_1 < 6) & qty_1_ok)
    n_1 = int(np.count_nonzero(keep_1))
    # apply the mask once, only to the columns the output needs
    id_1, id_1_ok = read_column(sales_1, 'id', rows=keep_1)
    amount_2, amount_2_ok = amount_1[keep_1], amount_1_ok[keep_1]
    # the projection, over the filtered columns
    double_amount_1 = (amount_2 * 2)

    return build_table([
        ('id', DType.INT, 'sales', id_1, id_1_ok),
        ('double_amount', DType.FLOAT, None, double_amount_1, amount_2_ok),
    ])
```

**Unfused** (`generate(plan, catalog, fuse=False)`), four sections, each materialising
its whole output. Note `day`: nothing ever uses it, yet it is read and then copied by
each Filter.

```python
def run(tables):
    # Scan[sales, pushed=amount > 15]
    sales_1 = as_table(tables['sales'])
    amount_1, amount_1_ok = read_column(sales_1, 'amount')
    pushed_1 = ((amount_1 > 15) & amount_1_ok)
    id_1, id_1_ok = read_column(sales_1, 'id', rows=pushed_1)
    region_1, region_1_ok = read_column(sales_1, 'region', rows=pushed_1)
    amount_2, amount_2_ok = amount_1[pushed_1], amount_1_ok[pushed_1]
    qty_1, qty_1_ok = read_column(sales_1, 'qty', rows=pushed_1)
    day_1, day_1_ok = read_column(sales_1, 'day', rows=pushed_1)
    n_1 = int(np.count_nonzero(pushed_1))

    # Filter[region = 'US']
    keep_1 = ((region_1 == 'US') & region_1_ok)
    n_2 = int(np.count_nonzero(keep_1))
    id_2, id_2_ok = id_1[keep_1], id_1_ok[keep_1]
    region_2, region_2_ok = region_1[keep_1], region_1_ok[keep_1]
    amount_3, amount_3_ok = amount_2[keep_1], amount_2_ok[keep_1]
    qty_2, qty_2_ok = qty_1[keep_1], qty_1_ok[keep_1]
    day_2, day_2_ok = day_1[keep_1], day_1_ok[keep_1]

    # Filter[qty < 6]
    keep_2 = ((qty_2 < 6) & qty_2_ok)
    n_3 = int(np.count_nonzero(keep_2))
    id_3, id_3_ok = id_2[keep_2], id_2_ok[keep_2]
    region_3, region_3_ok = region_2[keep_2], region_2_ok[keep_2]
    amount_4, amount_4_ok = amount_3[keep_2], amount_3_ok[keep_2]
    qty_3, qty_3_ok = qty_2[keep_2], qty_2_ok[keep_2]
    day_3, day_3_ok = day_2[keep_2], day_2_ok[keep_2]

    # Project[id, amount * 2 AS double_amount]
    double_amount_1 = (amount_4 * 2)

    return build_table([
        ('id', DType.INT, 'sales', id_3, id_3_ok),
        ('double_amount', DType.FLOAT, None, double_amount_1, amount_4_ok),
    ])
```

| | unfused | fused |
|---|---|---|
| operator passes (sections) | 4 | 1 |
| masks built | 3 | 1 |
| columns read from the table | 5 | 4 |
| row gathers (`[mask]` or `rows=mask`) | 26 | 3 |

### Example: LIKE runs last, on the survivors

`SELECT id FROM sales WHERE region LIKE '%S' AND qty < 4` with the LIKE as the lower Filter:

```python
def run(tables):
    # Fused pipeline: 4 operators, one pass over sales
    #   Project[id]
    #     Filter[qty < 4]
    #       Filter[region LIKE '%S']
    #         Scan[sales]
    sales_1 = as_table(tables['sales'])
    # every predicate, over just the columns it reads, ANDed into one mask
    # Filter: (qty < 4)
    qty_1, qty_1_ok = read_column(sales_1, 'qty')
    keep_1 = ((qty_1 < 4) & qty_1_ok)
    # Filter: (region LIKE '%S')
    #   a Python loop per row: run it only on the rows still alive
    alive_1 = np.flatnonzero(keep_1)
    region_1, region_1_ok = read_column(sales_1, 'region', rows=alive_1)
    keep_1[alive_1] = (like(region_1, '%S') & region_1_ok)
    n_1 = int(np.count_nonzero(keep_1))
    # apply the mask once, only to the columns the output needs
    id_1, id_1_ok = read_column(sales_1, 'id', rows=keep_1)

    return build_table([
        ('id', DType.INT, 'sales', id_1, id_1_ok),
    ])
```

### Measurements

200,000 rows, 8 INT columns and one STRING column, median of 6 runs after a warm-up
(Python 3.13, numpy 2, a laptop; the C6 benchmark runner measures the real query suite):

| chain | unfused ms | fused ms | speedup |
|---|---|---|---|
| `Project <- Filter <- Filter <- Scan`, numeric, ~45% of rows kept | 24.46 | 4.57 | 5.4x |
| LIKE above a 1%-selective filter | 2.45 | 0.76 | 3.2x (was 0.09x before LIKE went last) |
| a filter that keeps every row | 7.06 | 1.40 | 5.0x |

Peak memory (`tracemalloc`) for the first chain at 50,000 rows: 746 KB fused vs 4,020 KB
unfused. `tests/test_codegen_fusion.py` asserts fused stays under 60% of unfused.

### How it is tested (`tests/test_codegen_fusion.py`)

* Every plan in the corpus, fused and unfused, against the interpreter, including the
  output layout (column names, types, qualifiers).
* 300 random fusable chains (pruned and full scans, pushed predicates that read
  non-output columns, 0 to 3 stacked filters, LIKE / NOT LIKE, constant FALSE/NULL
  predicates) and 150 random join/aggregate/sort pipelines whose inputs fuse.
* Shape: the canonical `Project <- Filter <- Scan` is 3 sections unfused and 1 fused; the
  chain above builds 3 masks unfused and 1 fused, and the fused one gathers 3 times.
* LIKE runs after the vectorised predicates, only on survivors; the in-place update
  never writes into an input table.
* Fused peak memory is lower, and fused LIKE is no slower than unfused.

## Benchmark runner (Phase C6)

```bash
python -m bench.runner --scale bench            # 20 golden queries, ~20 minutes
python -m bench.runner --scale tiny --queries q01 q07 --runs 3
python -m bench.runner --summarize bench/results/runner_bench.csv
```

Every golden query (`tests/fixtures/queries/`, data from `bench/data/generate.py`) runs
under six configurations:

| configuration | plan | engine |
|---|---|---|
| `interpreted_unoptimized` | as bound | `runtime.interpret` (the naive baseline) |
| `interpreted_optimized` | `optimizer.optimize` | `runtime.interpret` |
| `compiled_unoptimized` | as bound | generated code, fusion on |
| `compiled_optimized` | `optimizer.optimize` | generated code, fusion on |
| `compiled_unoptimized_unfused` | as bound | generated code, fusion off |
| `compiled_optimized_unfused` | `optimizer.optimize` | generated code, fusion off |

The first four are the 2x2 from the brief. It separates the optimizer's contribution
(rows 1 vs 2, 3 vs 4) from compilation's (2 vs 4). The last two isolate fusion (C5).

How it measures, and why:

* **Correctness first.** Every configuration's answer is compared with
  `interpret(unoptimized plan)`. A wrong answer is still timed but flagged
  (`matches_reference`), and the runner exits 1.
* **Execution only.** Binding, optimizing and code generation happen once, before timing,
  as for a prepared statement. `compile_ms` reports generate() + compile() separately.
  Input tables are converted from Arrow once, up front, so no configuration pays for that.
* **Blocks, taking turns.** A sample is a block of back-to-back calls lasting at least
  50 ms (one call for anything slower), with the garbage collector off, as `timeit` does.
  Each round times one block of every configuration in turn, and the result is the
  median of 5 rounds after a warm-up. Both choices came from measuring the measurement:
  timing configurations one after another put a slow stretch of the laptop on one of
  them (identical code measured 27.0 and 17.8 ms), and timing single calls in turn put
  an interpreted run's garbage inside the next short compiled call (0.19 ms measured
  as 1.6 ms).
* **Memory apart.** `peak_memory_kb` is the `tracemalloc` peak of one extra, untimed call,
  because tracing slows everything down. numpy reports its buffers to tracemalloc.
* `rows_scanned` uses Person A's definition (base rows read by Scans). `cells_scanned`
  is rows x columns read, so column pruning shows up. `fused_pipelines` counts the
  chains fusion fused. Where it is 0, fused and unfused are the same code, and the
  summary leaves that ratio blank instead of reporting noise.

The CSV starts with Person A's columns (`bench/report.py: CSV_COLUMNS`), so
`bench.report.generate_charts("bench/results/runner_bench.csv")` draws it unchanged. A
Markdown summary is written next to it.

Two things the runner needed from outside codegen, both temporary:

* `bench/aliases.py`. The binder emits alias qualifiers (`o.cust_id`) under
  `Scan[orders]`, which nothing downstream can resolve. That affects 9 of the 20
  queries (every join). `resolve_aliases` maps each alias to the one scanned table whose schema has
  every column used with it, and refuses to guess otherwise. It is a no-op once the
  binder emits table names.
* The interpreter's joins. The C1 interpreter tried every left x right pair, which is
  2.5 billion pairs for lineitem x orders at bench scale. Equality conjuncts now build a
  hash index that only supplies candidate rows. The full condition is still evaluated
  on every candidate, and output order is unchanged (tests compare it with the all-pairs
  loop row for row). Without this, "interpreted vs compiled" would have measured
  nested loops against hash joins rather than interpretation against compilation.

### Results at bench scale

100,000 lineitem / 25,000 orders / 5,000 customer and part rows, with Person B's optimizer
(`b/phase-7-ablation`, not on main yet). All 120 measurements give the reference answer.
Full table: [`bench/results/runner_bench.md`](../bench/results/runner_bench.md); raw
numbers: `bench/results/runner_bench.csv`.

| speedup (geometric mean over 20 queries) | |
|---|---|
| optimizer, interpreted (`interpreted_unoptimized` / `interpreted_optimized`) | **1.9x** |
| optimizer, compiled (`compiled_unoptimized` / `compiled_optimized`) | **1.8x** |
| compilation (`interpreted_optimized` / `compiled_optimized`) | **42.7x** |
| fusion, unoptimized plans (the 7 queries where a chain fused) | **1.14x** |
| fusion, optimized plans (the 2 queries where a chain fused) | **1.03x** |
| total (`interpreted_unoptimized` / `compiled_optimized`) | **82.4x** |

What the numbers say:

* **Compilation is the big lever:** 8x to 304x. It is largest on single-table scans and
  filters (q03-q06: 229-304x), where the interpreter evaluates each expression row by
  row and the generated code makes one numpy call per operator. It is smallest where the
  generated code still loops in Python: building and probing the join hash table and
  assigning group ids (q13-q16, q19: 8-18x). Those loops are the next thing to
  vectorise (e.g. grouping with `np.unique`, a sort-merge join with `np.searchsorted`).
  Even counting the one-off generate + compile (0.6-12.8 ms), a query's first run is at
  least 7.2x faster than the interpreter.
* **The optimizer matters for joins:** 2.8-15x on q07-q12 and q19-q20, where predicates
  pushed below the joins shrink both join inputs. It gives at most 1.35x on single-table
  queries, where the scan already is the whole query. It contributes about the same
  factor to both engines, so its gain and compilation's multiply.
* **Fusion and the optimizer remove the same intermediates.** On bound plans fusion is
  worth up to 1.52x (q02) and 1.34x (q05). After B's pushdown, pruning and folding,
  almost every `Project <- Filter <- Scan` chain has become a single pushed, pruned
  Scan, which already is one pass. Only 2 optimized plans still contain a fusable chain,
  and there it is worth 1.03x. Fusion pays when the optimizer can't simplify the plan
  (or isn't run).
* **Memory follows the data touched.** Peak memory falls from 0.8-93 MB (naive) to
  0.1-18 MB (compiled, optimized). Pruning cuts cells scanned by 17-70%, which is most
  visible in compiled joins (q10: 61 MB unoptimized, 5.8 MB optimized).
* **Noise.** On the largest joins, repeated runs on this laptop vary by up to about 30%:
  q11 fuses nothing, so its fused and unfused code is identical, yet it measured 342 and
  259 ms. Treat differences under about 1.3x on those queries as noise. The ratios above
  2x are stable across the three full runs made while building the runner.
