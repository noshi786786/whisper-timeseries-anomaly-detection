"""
benchmark_inference.py

Benchmarks naive per-sample inference vs batched inference.
This demonstrates the idea behind inference-speed optimizations (batching/caching).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def main() -> None:
    out_dir = Path("results")
    X_val = np.load(out_dir / "X_val.npy")

    import joblib  # type: ignore
    model = joblib.load(out_dir / "baseline_model.joblib")

    # ---- Naive loop: predict one sample at a time ----
    t0 = time.perf_counter()
    for i in range(len(X_val)):
        _ = model.predict_proba(X_val[i : i + 1])
    t1 = time.perf_counter()
    naive_time = t1 - t0

    # ---- Batched inference: predict everything at once ----
    t2 = time.perf_counter()
    _ = model.predict_proba(X_val)
    t3 = time.perf_counter()
    batch_time = t3 - t2

    speedup = naive_time / max(batch_time, 1e-12)

    print(f"Naive loop time: {naive_time:.6f} s")
    print(f"Batched time:    {batch_time:.6f} s")
    print(f"Speedup:         {speedup:.2f}x")

    out_dir.mkdir(exist_ok=True)
    (out_dir / "benchmark.txt").write_text(
        f"naive_time={naive_time}\n"
        f"batch_time={batch_time}\n"
        f"speedup={speedup}\n"
    )


if __name__ == "__main__":
    main()
