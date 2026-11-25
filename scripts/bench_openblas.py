#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measures OpenBLAS sgemm through NumPy for the shapes of bench/micro/bench_cpu.cpp (BM_Linear), so ii-cpu can be
compared with a mature BLAS on the same machine.

    python scripts/bench_openblas.py [--threads N]   # N = 1 or 0 (all CPUs); runs itself in a fresh process per N

Prints one line per shape: "M N K threads time_us GFLOP/s". NumPy wheels bundle OpenBLAS (check numpy.show_config()).
"""

import argparse
import os
import subprocess
import sys
import time

SHAPES = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (1, 4096, 4096),
          (64, 4096, 4096), (512, 4096, 4096)]


def measure(threads: int) -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    for m, n, k in SHAPES:
        x = rng.uniform(-1, 1, (m, k)).astype(np.float32)
        w = rng.uniform(-1, 1, (n, k)).astype(np.float32)
        wt = w.T  # linear layer: x @ w^T (the transposed view, as BLAS sees it)
        x @ wt  # warm up
        best = float("inf")
        deadline = time.perf_counter() + 1.0
        runs = 0
        while runs < 3 or time.perf_counter() < deadline:
            start = time.perf_counter()
            x @ wt
            best = min(best, time.perf_counter() - start)
            runs += 1
        gflops = 2.0 * m * n * k / best / 1e9
        print(f"{m:5d} {n:5d} {k:5d} {threads:3d} {best * 1e6:12.1f} {gflops:9.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, action="append", help="1, or 0 for all CPUs (repeatable)")
    parser.add_argument("--child", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is not None:
        measure(args.child)
        return
    print("    M     N     K thr      time_us   GFLOP/s")
    for threads in args.threads or [1, 0]:
        env = dict(os.environ)
        count = str(threads if threads > 0 else os.cpu_count())
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = count
        subprocess.run([sys.executable, __file__, "--child", str(threads)], env=env, check=True)


if __name__ == "__main__":
    main()
