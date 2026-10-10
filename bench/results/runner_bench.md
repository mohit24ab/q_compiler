# Benchmark results: scale `bench`

Produced by `python -m bench.runner --scale bench`; raw numbers in `runner_bench.csv`. Median of 5 timed runs per configuration (after a warm-up), configurations interleaved round-robin. Times are execution only: binding, optimizing and code generation happen once, beforehand.

* data: 1,000,000 lineitem / 250,000 orders / 50,000 customer / 50,000 part rows (`bench/data/generate.py`, seed 42)
* optimizer: optimizer.optimize from Person B's `b/phase-7-ablation` @ 454744a, run on main @ 0993653 + `c/order-by-aggregate` (B's latest is not on main yet)
* python: 3.13.7
* numpy: 2.5.3
* platform: Windows-11-10.0.26200-SP0
* processor: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
* every answer matches the reference interpreter: yes
* since this run (checked 10 Oct; the table is not re-measured, see below):
  * B's `e8ca91e` keeps join order under Sort/Limit (`keep_row_order`). Only q10's
    optimized plan changes: `(lineitem ⋈ orders) ⋈ customer` instead of
    `lineitem ⋈ (orders ⋈ customer)`. Timed alternately in one process, it is **1.6x
    slower** in both engines (compiled 448 → 715 ms, interpreted 6.2 → 9.9 s), with
    the same answer.
  * A's q11 gained a tie-break key (`ORDER BY quantity DESC, l.id`). Timed the same
    way, no measurable difference.
  * Re-measuring q10 and q11 on 10 Oct swung up to 3x with the laptop's CPU state,
    between rounds of the same code. Those rows are not spliced into a table measured
    in one 2-hour session.

Ratios are speedups (higher is better). `optimizer` compares unoptimized with optimized plans on the same engine, `compilation` compares the interpreter with generated code on the optimized plan, `fusion` compares generated code with fusion off and on, `total` compares the naive baseline with the full pipeline.

| query | interpreted_unoptimized ms | interpreted_optimized ms | compiled_unoptimized ms | compiled_optimized ms | compiled_unoptimized_unfused ms | compiled_optimized_unfused ms | optimizer, interpreted x | optimizer, compiled x | compilation x | fusion, unoptimized x | fusion, optimized x | total x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q01 | 152.49 | 133.05 | 2.52 | 2.53 | 3.41 | 2.50 | 1.15 | 1.00 | 52.59 | 1.35 |  | 60.27 |
| q02 | 10113.30 | 10053.29 | 83.84 | 82.38 | 117.68 | 82.78 | 1.01 | 1.02 | 122.04 | 1.40 | 1.00 | 122.77 |
| q03 | 2458.91 | 2310.60 | 22.11 | 21.98 | 26.03 | 22.13 | 1.06 | 1.01 | 105.13 | 1.18 |  | 111.88 |
| q04 | 443.13 | 444.45 | 2.56 | 2.51 | 3.05 | 2.52 | 1.00 | 1.02 | 176.93 | 1.19 | 1.00 | 176.41 |
| q05 | 31832.42 | 30829.69 | 103.01 | 104.55 | 121.24 | 95.14 | 1.03 | 0.99 | 294.89 | 1.18 |  | 304.49 |
| q06 | 424.26 | 405.51 | 1.61 | 1.60 | 2.07 | 1.80 | 1.05 | 1.00 | 253.44 | 1.29 |  | 265.16 |
| q07 | 4999.89 | 1423.15 | 230.91 | 44.05 | 229.41 | 45.37 | 3.51 | 5.24 | 32.31 |  |  | 113.52 |
| q08 | 27162.59 | 6464.06 | 1228.22 | 182.50 | 1154.40 | 173.19 | 4.20 | 6.73 | 35.42 |  |  | 148.84 |
| q09 | 22572.97 | 7410.14 | 1405.34 | 429.20 | 1345.32 | 430.85 | 3.05 | 3.27 | 17.26 |  |  | 52.59 |
| q10 | 72180.55 | 4871.80 | 3366.48 | 335.25 | 3356.93 | 355.05 | 14.82 | 10.04 | 14.53 |  |  | 215.30 |
| q11 | 51241.16 | 6000.93 | 1596.85 | 462.00 | 1615.53 | 464.74 | 8.54 | 3.46 | 12.99 |  |  | 110.91 |
| q12 | 67847.52 | 12903.20 | 2393.73 | 374.93 | 2303.19 | 376.21 | 5.26 | 6.38 | 34.41 |  |  | 180.96 |
| q13 | 9790.50 | 9392.69 | 666.60 | 645.66 | 666.89 | 655.93 | 1.04 | 1.03 | 14.55 |  |  | 15.16 |
| q14 | 307.57 | 303.64 | 17.42 | 17.41 | 16.82 | 17.17 | 1.01 | 1.00 | 17.45 |  |  | 17.67 |
| q15 | 1355.98 | 1313.47 | 158.85 | 158.79 | 156.24 | 156.37 | 1.03 | 1.00 | 8.27 |  |  | 8.54 |
| q16 | 362.30 | 336.78 | 32.19 | 30.23 | 33.36 | 30.46 | 1.08 | 1.06 | 11.14 |  |  | 11.98 |
| q17 | 10097.44 | 10070.41 | 403.81 | 252.90 | 390.94 | 240.76 | 1.00 | 1.60 | 39.82 | 0.97 |  | 39.93 |
| q18 | 9427.37 | 5088.28 | 446.45 | 234.68 | 474.94 | 237.21 | 1.85 | 1.90 | 21.68 |  |  | 40.17 |
| q19 | 94245.08 | 17484.52 | 4193.90 | 1277.76 | 4150.82 | 1267.30 | 5.39 | 3.28 | 13.68 |  |  | 73.76 |
| q20 | 73177.21 | 16918.12 | 2584.43 | 521.13 | 2550.83 | 533.74 | 4.33 | 4.96 | 32.46 |  |  | 140.42 |
| geomean |  |  |  |  |  |  | 2.07 | 2.03 | 35.53 | 1.22 | 1.00 | 73.40 |

A blank fusion ratio means no chain fused: both configurations ran the same code.
