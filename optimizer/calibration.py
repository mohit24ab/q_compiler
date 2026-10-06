"""Measure the cost model's constants: ``python -m optimizer.calibration``.

Each weight in ``optimizer.cost.CostWeights`` is the time of one unit of
work, in nanoseconds. Each benchmark here runs the same code that Person
C's code generator emits for that unit of work (codegen/operators.py and
runtime/table.py on the C4 branch), on 200,000 rows, and keeps the fastest
of several runs. The numbers vary by machine. What the optimizer needs is
their *ratios*, which hold up much better: a hash-table insert in a Python
loop costs ~100x a vectorized comparison on any machine.

The optimizer never imports this module. It needs numpy and pyarrow (both
in the project stack), and its results are copied by hand into
``DEFAULT_WEIGHTS`` in optimizer/cost.py, with the machine they came from.
"""

from __future__ import annotations

import math
import time
from typing import Callable


def _best(fn: Callable[[], object], repeat: int) -> float:
    """Fastest of ``repeat`` runs, in nanoseconds."""
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter_ns()
        fn()
        best = min(best, time.perf_counter_ns() - start)
    return best


def measure(n: int = 200_000, repeat: int = 5, seed: int = 7) -> dict[str, float]:
    """Return nanoseconds per unit for every ``CostWeights`` field."""
    import numpy as np
    import pyarrow as pa

    rng = np.random.default_rng(seed)
    ints = rng.integers(0, n // 4, n)
    floats = rng.random(n) * 1000
    strings = pa.array([f"value#{i % 5000}" for i in range(n)])
    nullable = pa.array([None if i % 10 == 0 else int(v) for i, v in enumerate(ints)])
    mask = ints < n // 8  # keeps ~half the rows
    ok = np.ones(n, dtype=bool)

    def convert(arr):  # runtime/table.py Table.from_arrow, one column
        valid = arr.is_valid().to_numpy(zero_copy_only=False) if arr.null_count else None
        if arr.null_count:
            arr = arr.fill_null(0)
        return arr.to_numpy(zero_copy_only=False), valid

    out: dict[str, float] = {}
    out["read_fixed"] = _best(lambda: convert(nullable), repeat) / n
    out["read_string"] = _best(lambda: convert(strings), repeat) / n
    out["predicate"] = _best(lambda: (ints > 7) & ok, repeat) / (2 * n)  # two vectorized operators
    out["gather_mask"] = _best(lambda: (floats[mask], ok[mask]), repeat) / n  # per input row
    idx = rng.integers(0, n, n)
    out["gather_index"] = _best(lambda: (floats[idx], ok[idx]), repeat) / n  # per output row

    keys, oks = ints.tolist(), ok.tolist()

    def build():  # codegen _hash_join: the build loop
        table: dict = {}
        for i, (key, present) in enumerate(zip(ints.tolist(), ok.tolist())):
            if present:
                table.setdefault(key, []).append(i)
        return table

    out["hash_build"] = _best(build, repeat) / n
    table = build()
    misses = (ints + n).tolist()  # no key matches: the probe's cost per row alone

    def probe(probe_keys):  # codegen _hash_join: the probe loop
        bi, pi = [], []
        for j, (key, present) in enumerate(zip(probe_keys, oks)):
            if present:
                for match in table.get(key, ()):
                    bi.append(match)
                    pi.append(j)
        return bi, pi

    out["hash_probe"] = _best(lambda: probe(misses), repeat) / n
    pairs = len(probe(keys)[0])

    def emit():  # every matching pair: append, convert, then the reference-order lexsort
        bi, pi = probe(keys)
        li, ri = np.array(bi, dtype=np.int64), np.array(pi, dtype=np.int64)
        order = np.lexsort((ri, li))
        return li[order], ri[order]

    out["join_emit"] = max(0.0, _best(emit, repeat) - out["hash_probe"] * n) / pairs

    side = int(math.isqrt(n))

    def nested_loop():  # codegen _join without equality keys
        return np.repeat(np.arange(side), side), np.tile(np.arange(side), side)

    out["pair"] = _best(nested_loop, repeat) / (side * side)

    def group():  # codegen _group_ids
        groups: dict = {}
        gid = np.fromiter((groups.setdefault(key, len(groups)) for key in keys), dtype=np.int64, count=n)
        first = np.zeros(len(groups), dtype=np.int64)
        first[gid[::-1]] = np.arange(n)[::-1]
        return gid

    out["group"] = _best(lambda: group(), repeat) / n
    gid = group()
    ng = int(gid.max()) + 1

    def accumulate():  # codegen _accumulate, grouped sum
        count = np.bincount(gid, minlength=ng)
        total = np.zeros(ng)
        np.add.at(total, gid, floats)
        return count, total

    out["aggregate"] = _best(accumulate, repeat) / n
    out["reduce"] = _best(lambda: (floats[ok].sum(), len(floats)), repeat) / n  # global aggregate

    def sort():  # codegen _sort, one key: rank, then a stable lexsort
        rank = np.unique(floats, return_inverse=True)[1].reshape(-1)
        return np.lexsort((rank,))

    out["sort"] = _best(sort, repeat) / (n * math.log2(n))
    return out


def main() -> None:
    import platform

    from optimizer.cost import DEFAULT_WEIGHTS

    measured = measure()
    print(f"machine: {platform.processor() or platform.machine()}, python {platform.python_version()}")
    print(f"{'weight':<14}{'measured ns':>12}{'in DEFAULT_WEIGHTS':>20}")
    for name, value in measured.items():
        print(f"{name:<14}{value:>12.3f}{getattr(DEFAULT_WEIGHTS, name):>20.3f}")


if __name__ == "__main__":
    main()
