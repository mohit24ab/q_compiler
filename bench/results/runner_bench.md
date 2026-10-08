# Benchmark results: scale `bench`

Produced by `python -m bench.runner --scale bench`; raw numbers in `runner_bench.csv`. Median of 5 timed runs per configuration (after a warm-up), configurations interleaved round-robin. Times are execution only: binding, optimizing and code generation happen once, beforehand.

* optimizer: optimizer.optimize from Person B's `b/phase-7-ablation` @ dfadf15, run on main @ d06ae64 + this branch (B's optimizer is not on main yet)
* python: 3.13.7
* numpy: 2.5.3
* platform: Windows-11-10.0.26200-SP0
* processor: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
* every answer matches the reference interpreter: yes

Ratios are speedups (higher is better). `optimizer` compares unoptimized with optimized plans on the same engine, `compilation` compares the interpreter with generated code on the optimized plan, `fusion` compares generated code with fusion off and on, `total` compares the naive baseline with the full pipeline.

| query | interpreted_unoptimized ms | interpreted_optimized ms | compiled_unoptimized ms | compiled_optimized ms | compiled_unoptimized_unfused ms | compiled_optimized_unfused ms | optimizer, interpreted x | optimizer, compiled x | compilation x | fusion, unoptimized x | fusion, optimized x | total x |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| q01 | 16.06 | 13.58 | 0.24 | 0.23 | 0.28 | 0.24 | 1.18 | 1.02 | 58.03 | 1.20 |  | 68.62 |
| q02 | 1061.53 | 1035.74 | 7.55 | 7.61 | 11.46 | 7.53 | 1.02 | 0.99 | 136.12 | 1.52 | 0.99 | 139.51 |
| q03 | 772.82 | 574.47 | 2.43 | 2.20 | 2.38 | 2.02 | 1.35 | 1.10 | 260.89 | 0.98 |  | 350.96 |
| q04 | 41.75 | 41.27 | 0.19 | 0.18 | 0.21 | 0.19 | 1.01 | 1.04 | 229.29 | 1.11 | 1.07 | 231.97 |
| q05 | 2341.66 | 2208.32 | 7.48 | 7.33 | 10.05 | 7.26 | 1.06 | 1.02 | 301.19 | 1.34 |  | 319.37 |
| q06 | 33.36 | 31.94 | 0.11 | 0.10 | 0.11 | 0.10 | 1.04 | 1.05 | 304.21 | 0.99 |  | 317.74 |
| q07 | 328.68 | 109.19 | 10.30 | 2.95 | 10.33 | 2.90 | 3.01 | 3.50 | 37.06 |  |  | 111.57 |
| q08 | 1538.01 | 377.59 | 42.02 | 10.81 | 43.27 | 11.03 | 4.07 | 3.89 | 34.92 |  |  | 142.25 |
| q09 | 1777.60 | 593.43 | 63.44 | 22.85 | 63.21 | 23.34 | 3.00 | 2.78 | 25.97 |  |  | 77.78 |
| q10 | 9179.11 | 619.97 | 261.72 | 34.28 | 291.28 | 37.09 | 14.81 | 7.64 | 18.09 |  |  | 267.80 |
| q11 | 9033.98 | 1748.63 | 341.99 | 83.58 | 258.51 | 70.09 | 5.17 | 4.09 | 20.92 |  |  | 108.09 |
| q12 | 6315.05 | 1520.92 | 146.30 | 37.72 | 187.53 | 37.23 | 4.15 | 3.88 | 40.33 |  |  | 167.44 |
| q13 | 1169.45 | 1106.61 | 71.24 | 72.97 | 73.59 | 72.78 | 1.06 | 0.98 | 15.17 |  |  | 16.03 |
| q14 | 12.46 | 12.00 | 0.69 | 0.68 | 0.70 | 0.71 | 1.04 | 1.02 | 17.63 |  |  | 18.30 |
| q15 | 149.06 | 128.88 | 13.93 | 15.51 | 13.40 | 17.07 | 1.16 | 0.90 | 8.31 |  |  | 9.61 |
| q16 | 51.99 | 50.88 | 3.96 | 4.07 | 4.30 | 4.09 | 1.02 | 0.97 | 12.50 |  |  | 12.77 |
| q17 | 1383.91 | 1341.87 | 52.82 | 33.37 | 50.23 | 32.05 | 1.03 | 1.58 | 40.21 | 0.95 |  | 41.47 |
| q18 | 1083.19 | 632.45 | 42.44 | 23.56 | 42.72 | 22.01 | 1.71 | 1.80 | 26.84 |  |  | 45.97 |
| q19 | 6026.40 | 2021.58 | 371.03 | 129.12 | 345.56 | 131.61 | 2.98 | 2.87 | 15.66 |  |  | 46.67 |
| q20 | 8680.05 | 2461.48 | 242.63 | 62.95 | 248.40 | 65.80 | 3.53 | 3.85 | 39.10 |  |  | 137.88 |
| geomean |  |  |  |  |  |  | 1.93 | 1.82 | 42.71 | 1.14 | 1.03 | 82.42 |

A blank fusion ratio means no chain fused: both configurations ran the same code.
