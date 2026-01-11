"""
train.py

Demo-scale training script.
Trains a baseline anomaly detector on synthetic time-series data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data import DatasetConfig, generate_synthetic_timeseries
from model import build_baseline_classifier, simple_featurize


def main() -> None:
    cfg = DatasetConfig()
    X, y = generate_synthetic_timeseries(cfg)

    X_feat = simple_featurize(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_feat, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )

    model = build_baseline_classifier()
    model.fit(X_train, y_train)

    # Save model using numpy-friendly approach (simple pickle is fine for demo)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # We'll save validation split so evaluate.py can run consistently
    np.save(out_dir / "X_val.npy", X_val)
    np.save(out_dir / "y_val.npy", y_val)

    # Save a tiny metadata file
    meta = {"n_samples": int(cfg.n_samples), "seq_len": int(cfg.seq_len), "anomaly_fraction": float(cfg.anomaly_fraction)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # For demo simplicity, store model via joblib
    try:
        import joblib  # type: ignore
        joblib.dump(model, out_dir / "baseline_model.joblib")
        print("Saved model to results/baseline_model.joblib")
    except Exception:
        print("joblib not installed; skipping model save. (Install joblib for persistence.)")

    print("Training complete.")


if __name__ == "__main__":
    main()
