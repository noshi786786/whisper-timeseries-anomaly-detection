"""
evaluate.py

Demo-scale evaluation script.
Loads the trained baseline model and prints simple metrics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def main() -> None:
    out_dir = Path("results")
    X_val = np.load(out_dir / "X_val.npy")
    y_val = np.load(out_dir / "y_val.npy")

    # Load model
    try:
        import joblib  # type: ignore
        model = joblib.load(out_dir / "baseline_model.joblib")
    except Exception as e:
        raise RuntimeError("Model file not found. Run `python src/train.py` first (and install joblib).") from e

    # Probabilities for metrics
    probs = model.predict_proba(X_val)[:, 1]

    auroc = roc_auc_score(y_val, probs)
    ap = average_precision_score(y_val, probs)

    print(f"Validation AUROC: {auroc:.4f}")
    print(f"Validation Average Precision: {ap:.4f}")


if __name__ == "__main__":
    main()
