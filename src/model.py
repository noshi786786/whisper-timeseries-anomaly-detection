"""
model.py

Representative baseline models for time-series anomaly detection.

Note:
- The full project adapts a pretrained Whisper encoder.
- This public repo starts with a lightweight baseline to keep the demo runnable.
- Optional extensions can load cached "embeddings" produced by Whisper locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelConfig:
    max_iter: int = 1000
    C: float = 1.0
    random_state: int = 42


def build_baseline_classifier(cfg: Optional[ModelConfig] = None) -> Pipeline:
    """
    Baseline anomaly classifier using a simple sklearn pipeline.

    Input expected:
    - X: (n_samples, n_features)
      For raw sequences, you should first featurize (e.g., summary stats / FFT bins).
    """
    cfg = cfg or ModelConfig()
    clf = LogisticRegression(
        max_iter=cfg.max_iter,
        C=cfg.C,
        random_state=cfg.random_state,
        n_jobs=None,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def simple_featurize(X: np.ndarray) -> np.ndarray:
    """
    Turn a raw time-series sequence into a small feature vector.
    This keeps the demo lightweight and runnable.

    Features:
    - mean, std, max, min
    - energy (mean squared value)
    """
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mx = X.max(axis=1)
    mn = X.min(axis=1)
    energy = (X ** 2).mean(axis=1)
    return np.vstack([mean, std, mx, mn, energy]).T
