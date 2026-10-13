# q_compiler

**Compiler-Driven Data Analytics Pipeline Generation and Query Optimization Framework**

A SQL-like query goes in. An optimized relational plan comes out. Executable Python is
generated from that plan and run. Performance is measured against an unoptimized baseline.

The shared contract (`00-SHARED-CONTRACT.md`, circulated with the brief and not checked in)
is the authority for IR shapes, module interfaces and directory ownership; section numbers
such as §7 below refer to it. This README covers getting the repo running and finding your
way around.

## Setup

Contract §2 pins **Python 3.11** so all three machines behave identically.

```bash
py -3.11 -m venv .venv          # Windows;  python3.11 -m venv .venv elsewhere
.venv\Scripts\activate          # source .venv/bin/activate elsewhere
pip install -e ".[dev]"
```

Run the suite:

```bash
python -m pytest -q
```

CI runs exactly that on every push and pull request (`.github/workflows/ci.yml`) — the
Contract §8 "main stays green" requirement.

See the end-to-end pipeline on one query -- it prints the plan at each stage, the
generated Python, the result, the §7 differential check against the interpreter, and
the timings:

```bash
python demo.py --query q10     # --list shows all twenty; a bare SQL string works too
```

## Layout and ownership

Contract §3. **You may READ any directory. You may WRITE only your own.** Cross-directory
changes go through the owner.

| Directory | Owner | Contents |
|---|---|---|
| `ir/` | A | plan nodes, expressions, printer, visitor utilities |
| `catalog/` | A | table metadata, schemas, statistics |
| `frontend/` | A | sqlglot AST -> our IR, name resolution, type checking |
| `optimizer/` | B | pass framework, rewrite rules, cost model |
| `codegen/` | C | plan -> Python source, emitted operator templates |
| `runtime/` | C | naive reference executor, runtime helpers |
| `bench/` | A + C | benchmark suite, harness, results |
| `tests/` | all three | namespaced `test_ir_*`, `test_opt_*`, `test_codegen_*` |

## Module interfaces

Contract §5:

```python
parse_and_bind(sql, catalog) -> PlanNode                    # frontend  (A)
optimize(plan, catalog) -> tuple[PlanNode, list[PassTrace]] # optimizer (B)
generate(plan, catalog) -> str                              # codegen   (C)
compile_and_run(source, tables) -> Table                    # codegen   (C)
interpret(plan, tables) -> Table                            # runtime   (C)
```

## The optimizer pipeline

`optimizer.default_passes()` runs to a fixed point, in this order:

1. **Predicate pushdown** — filters reach the scans, so reordering costs plans from
   filtered estimates.
2. **Join reordering** — Selinger-style DP over connected subsets.
3. **Column pruning** — after reordering, because pruning inserts Projects between joins
   and those split the join trees reordering works on.
4. **Constant folding** — last; it also removes the no-op Projects pruning leaves behind.

The order is measured, not assumed — `optimizer.ablation.pass_order_study` runs the suite
through all 24 orderings. See `docs/optimizer.md`.

## Testing

Contract §7 defines correct as: `interpret(plan)` and
`compile_and_run(generate(optimize(plan)))` return identical rows, order-insensitive unless
the query has `ORDER BY`. That differential check is `tests/test_differential.py` and
`tests/test_opt_differential.py`, and it runs over the whole golden suite.

| Path | What it holds |
|---|---|
| `tests/fixtures/queries/q01–q20.sql` | the golden query suite |
| `tests/fixtures/golden/*.txt` | expected results, byte-for-byte |
| `tests/test_golden_plans.py` | expected optimized plans |
| `tests/test_duckdb_oracle.py` | DuckDB cross-check (Contract §2 allows it in tests only) |

**ORDER BY ties:** every golden query with a non-unique `ORDER BY` carries a unique
tie-breaker (commit `a8b273a`), so result ordering is total and the differential harness
can compare strictly rather than treating tied rows as a set.

## Documentation

| File | Contents |
|---|---|
| `docs/optimizer.md` | one section per pass: rule, preconditions, example, measured contribution |
| `docs/codegen.md` | emitted operator templates and the demo |
| `docs/ir-reference.md` | IR node reference |
| `docs/cardinality_estimates.md` | estimated vs. actual row counts per node |
| `docs/ablation/`, `bench/results/` | ablation study and benchmark numbers |
