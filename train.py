"""Segment wholesale customers with log-scaled K-Means."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = pd.read_csv(path)
    if records.columns[0].lower().startswith("unnamed"):
        records = records.drop(columns=records.columns[0])
    missing = [name for name in FEATURES if name not in records.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    spending = records[FEATURES].apply(pd.to_numeric, errors="coerce")
    if spending.isna().any().any() or (spending < 0).any().any():
        raise ValueError("Spending features must be numeric, present, and non-negative.")
    return records, spending


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project / "data" / "Wholesale customers data.csv")
    parser.add_argument("--output-dir", type=Path, default=project / "artifacts")
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if args.clusters < 2:
        raise ValueError("--clusters must be at least 2.")
    records, spending = load_data(args.data)
    if args.clusters >= len(spending):
        raise ValueError("--clusters must be smaller than the customer count.")

    scaler = StandardScaler()
    logged_values = np.log1p(spending.to_numpy(dtype=float))
    scaled_values = scaler.fit_transform(logged_values)
    model = KMeans(n_clusters=args.clusters, n_init=20, random_state=args.random_state)
    labels = model.fit_predict(scaled_values)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "kmeans": model, "features": FEATURES,
                 "transform": "log1p"}, args.output_dir / "customer_segments.joblib")

    results = records.copy()
    results["cluster"] = labels
    coordinates = PCA(n_components=2, random_state=args.random_state).fit_transform(scaled_values)
    results["pca_1"], results["pca_2"] = coordinates[:, 0], coordinates[:, 1]
    results.to_csv(args.output_dir / "segmented_customers.csv", index=False)
    counts = results.groupby("cluster").size().rename("customer_count")
    profiles = results.groupby("cluster")[FEATURES].mean().round(2)
    profiles.insert(0, "customer_count", counts)
    profiles.to_csv(args.output_dir / "cluster_profiles.csv")

    scores = []
    for k in range(2, min(8, len(spending) - 1) + 1):
        candidate = KMeans(n_clusters=k, n_init=20, random_state=args.random_state)
        candidate_labels = candidate.fit_predict(scaled_values)
        scores.append({"clusters": k, "inertia": float(candidate.inertia_),
                       "silhouette_score": float(silhouette_score(scaled_values, candidate_labels))})
    pd.DataFrame(scores).to_csv(args.output_dir / "cluster_selection.csv", index=False)

    metrics = {
        "project": "Wholesale Customer Segmentation with K-Means",
        "dataset": "UCI Wholesale customers",
        "customers": int(len(spending)),
        "features": FEATURES,
        "clusters": int(args.clusters),
        "silhouette_score": float(silhouette_score(scaled_values, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(scaled_values, labels)),
        "inertia": float(model.inertia_),
        "random_state": args.random_state,
        "preprocessing": ["log1p", "StandardScaler"],
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(9, 6))
    points = axis.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10",
                          alpha=0.8, edgecolor="white", linewidth=0.25)
    axis.set(title="Wholesale Customer Clusters (PCA projection)",
             xlabel="Principal component 1", ylabel="Principal component 2")
    axis.grid(alpha=0.2)
    fig.colorbar(points, ax=axis, label="Cluster")
    fig.tight_layout()
    fig.savefig(args.output_dir / "customer_segments_pca.png", dpi=160)
    plt.close(fig)

    print(f"Customers: {len(spending)}")
    print(f"Clusters: {args.clusters}")
    print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
    print(f"Outputs saved in {args.output_dir}")


if __name__ == "__main__":
    main()
