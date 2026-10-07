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
