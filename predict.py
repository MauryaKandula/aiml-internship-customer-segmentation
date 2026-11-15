"""Assign a new wholesale customer to a saved cluster."""
from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=project / "artifacts" / "customer_segments.joblib")
    parser.add_argument("--spend", type=float, nargs=6, required=True,
                        metavar=("FRESH", "MILK", "GROCERY", "FROZEN", "DETERGENTS_PAPER", "DELICASSEN"))
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found at {args.model}; run train.py first.")
    values = np.asarray(args.spend, dtype=float)
    if np.any(values < 0):
        raise ValueError("Spending values must be non-negative.")
    bundle = joblib.load(args.model)
    scaled = bundle["scaler"].transform(np.log1p(values.reshape(1, -1)))
    print(f"Assigned cluster: {int(bundle['kmeans'].predict(scaled)[0])}")


if __name__ == "__main__":
    main()
