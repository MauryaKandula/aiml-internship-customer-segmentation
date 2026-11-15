"""Assign new wholesale customer spend to a saved cluster model."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np

FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=root / "artifacts" / "customer_segments.joblib")
    parser.add_argument("--spend", type=float, nargs=6, required=True,
                        metavar=("FRESH", "MILK", "GROCERY", "FROZEN", "DETERGENTS_PAPER", "DELICASSEN"))
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError("Run python train.py before requesting a prediction.")
    values = np.asarray(args.spend, dtype=float)
    if np.any(values < 0):
        raise ValueError("Spending values must be non-negative.")
    bundle = joblib.load(args.model)
    scaled = bundle["scaler"].transform(np.log1p(values.reshape(1, -1)))
    cluster = int(bundle["model"].predict(scaled)[0])
    print(f"Assigned cluster: {cluster}")
    if hasattr(bundle["model"], "predict_proba"):
        membership = float(bundle["model"].predict_proba(scaled)[0, cluster])
        print(f"Gaussian mixture membership probability: {membership:.1%}")
    print("Input feature order: " + ", ".join(FEATURES))


if __name__ == "__main__":
    main()
