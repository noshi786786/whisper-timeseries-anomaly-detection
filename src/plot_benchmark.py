"""
plot_benchmark.py

Reads results/benchmark.txt and produces a simple bar chart.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    out_dir = Path("results")
    txt = (out_dir / "benchmark.txt").read_text().strip().splitlines()
    vals = {line.split("=")[0]: float(line.split("=")[1]) for line in txt}

    naive = vals["naive_time"]
    batch = vals["batch_time"]
    speedup = vals["speedup"]

    plt.figure()
    plt.bar(["Naive loop", "Batched"], [naive, batch])
    plt.ylabel("Seconds")
    plt.title(f"Inference Benchmark (speedup = {speedup:.2f}x)")
    plt.tight_layout()
    plt.savefig(out_dir / "inference_benchmark.png", dpi=200)
    print("Saved plot to results/inference_benchmark.png")


if __name__ == "__main__":
    main()
