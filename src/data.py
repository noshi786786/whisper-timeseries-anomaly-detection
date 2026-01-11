"""
data.py

Demo-scale data utilities for the representative Whisper time-series anomaly detection project.

This repo intentionally uses ONLY public-safe data:
- synthetic time-series signals (e.g., sinusoids + noise)
- optional cached embeddings (if generated locally)

No proprietary datasets are included.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DatasetConfig:
    n_samples: int = 2000
    seq_len: int = 2048
    anomaly_fraction: float = 0.10
    noise_std: float = 0.5
    seed: int = 42


def generate_synthetic_timeseries(cfg: DatasetConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a simple synthetic dataset:
      - Normal samples: sine waves + Gaussian noise
      - Anomalies: sine waves with injected spikes/perturbations

    Returns
    -------
    X : np.ndarray of shape (n_samples, seq_len)
    y : np.ndarray of shape (n_samples,) with labels {0, 1}
    """
    rng = np.random.default_rng(cfg.seed)

    n_anom = int(cfg.n_samples * cfg.anomaly_fraction)
    n_norm = cfg.n_samples - n_anom

    t = np.linspace(0, 8 * np.pi, cfg.seq_len)

    def make_base(n: int) -> np.ndarray:
        freqs = rng.uniform(0.8, 1.2, size=(n, 1))
        phases = rng.uniform(0, 2 * np.pi, size=(n, 1))
        base = np.sin(freqs * t + phases)
        noise = rng.normal(0.0, cfg.noise_std, size=base.shape)
        return base + noise

    X_norm = make_base(n_norm)
    X_anom = make_base(n_anom)

    # Inject simple anomaly: random spikes
    spike_positions = rng.integers(0, cfg.seq_len, size=(n_anom, 10))
    for i in range(n_anom):
        X_anom[i, spike_positions[i]] += rng.normal(5.0, 1.0, size=(10,))

    X = np.vstack([X_norm, X_anom]).astype(np.float32)
    y = np.array([0] * n_norm + [1] * n_anom, dtype=np.int64)

    # Shuffle
    idx = rng.permutation(cfg.n_samples)
    return X[idx], y[idx]
