"""Fit K-Means segments, compare counts, and report cross-seed stability."""
from __future__ import annotations
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
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def load_records(path=ROOT / "data" / "Wholesale customers data.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = pd.read_csv(path)
    if records.columns[0].lower().startswith("unnamed"):
        records = records.drop(columns=records.columns[0])
    missing = [name for name in FEATURES if name not in records.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    spending = records[FEATURES].apply(pd.to_numeric, errors="coerce")
    if spending.isna().any().any():
        raise ValueError("Spending data contains missing or non-numeric values.")
    if (spending < 0).any().any():
        raise ValueError("Spending values must be non-negative.")
    if len(spending) < 9:
        raise ValueError("At least nine customer rows are required.")
    return records, spending


def make_clusterer(algorithm, clusters, random_state):
    if algorithm == "K-Means":
        return KMeans(n_clusters=clusters, n_init=30, random_state=random_state)
    if algorithm == "Gaussian Mixture":
        return GaussianMixture(
            n_components=clusters,
            covariance_type="full",
            n_init=3,
            reg_covar=1e-5,
            max_iter=500,
            random_state=random_state,
        )
    raise ValueError("algorithm must be 'K-Means' or 'Gaussian Mixture'.")


def stability_summary(scaled, reference_labels, algorithm, clusters, random_state):
    repeat_scores = []
    for offset in range(1, 6):
        repeat = make_clusterer(algorithm, clusters, random_state + offset).fit_predict(scaled)
        repeat_scores.append(float(adjusted_rand_score(reference_labels, repeat)))
    return {
        "stability_ari_mean": float(np.mean(repeat_scores)),
        "stability_ari_std": float(np.std(repeat_scores)),
    }


def fit_segmentation(spending, clusters=2, random_state=42, algorithm="K-Means"):
    if not 2 <= clusters <= 8:
        raise ValueError("clusters must be between 2 and 8.")
    if clusters >= len(spending):
        raise ValueError("clusters must be smaller than the customer count.")
    logged = np.log1p(spending.to_numpy(dtype=float))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(logged)
    model = make_clusterer(algorithm, clusters, random_state)
    labels = model.fit_predict(scaled)
    metrics = {
        "algorithm": algorithm,
        "clusters": int(clusters),
        "silhouette_score": float(silhouette_score(scaled, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(scaled, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(scaled, labels)),
    }
    if algorithm == "K-Means":
        metrics["inertia"] = float(model.inertia_)
    else:
        metrics["aic"] = float(model.aic(scaled))
        metrics["bic"] = float(model.bic(scaled))
        metrics["converged"] = bool(model.converged_)
        metrics["iterations"] = int(model.n_iter_)
    metrics.update(stability_summary(scaled, labels, algorithm, clusters, random_state))
    return {"scaler": scaler, "model": model, "scaled": scaled,
            "labels": labels, "metrics": metrics}


def score_cluster_counts(spending, random_state=42):
    logged = np.log1p(spending.to_numpy(dtype=float))
    scaled = StandardScaler().fit_transform(logged)
    rows = []
    upper = min(8, len(spending) - 1)
    for algorithm in ("K-Means", "Gaussian Mixture"):
        for clusters in range(2, upper + 1):
            model = make_clusterer(algorithm, clusters, random_state)
            labels = model.fit_predict(scaled)
            stability = stability_summary(scaled, labels, algorithm, clusters, random_state)
            row = {
                "algorithm": algorithm,
                "clusters": clusters,
                "silhouette_score": float(silhouette_score(scaled, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(scaled, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(scaled, labels)),
                **stability,
            }
            if algorithm == "K-Means":
                row["inertia"] = float(model.inertia_)
            else:
                row["aic"] = float(model.aic(scaled))
                row["bic"] = float(model.bic(scaled))
                row["converged"] = bool(model.converged_)
            rows.append(row)
    return pd.DataFrame(rows)


def spending_profiles(spending, labels):
    clustered = spending.copy()
    clustered["cluster"] = labels
    means = clustered.groupby("cluster")[FEATURES].mean()
    counts = clustered.groupby("cluster").size().rename("customer_count")
    profiles = means.join(counts)
    profiles.insert(1, "share_pct", (counts / len(spending) * 100).round(1))
    overall = spending.mean()
    signatures = {}
    for cluster_id, row in means.iterrows():
        lift = row / overall
        above = lift[lift >= 1].sort_values(ascending=False).head(2)
        signatures[cluster_id] = (
            "Higher relative spend: " + ", ".join(above.index.tolist())
            if len(above) else "Below overall average across categories"
        )
    profiles["spending_signature"] = pd.Series(signatures)
    return profiles.reset_index()


def train_project(clusters=2, random_state=42, algorithm="K-Means",
                  data_path=ROOT / "data" / "Wholesale customers data.csv"):
    records, spending = load_records(data_path)
    fitted = fit_segmentation(spending, clusters, random_state, algorithm)
    labels = fitted["labels"]
    profiles = spending_profiles(spending, labels)
    selection = score_cluster_counts(spending, random_state)
    coordinates = PCA(n_components=2, random_state=random_state).fit_transform(fitted["scaled"])

    results_dir = ROOT / "results"
    artifacts_dir = ROOT / "artifacts"
    results_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    assigned = records.copy()
    assigned["cluster"] = labels
    if hasattr(fitted["model"], "predict_proba"):
        assigned["membership_probability"] = fitted["model"].predict_proba(fitted["scaled"]).max(axis=1)
    assigned["pca_1"], assigned["pca_2"] = coordinates[:, 0], coordinates[:, 1]
    assigned.to_csv(results_dir / "customer_assignments.csv", index=False)
    profiles.to_csv(results_dir / "cluster_profiles.csv", index=False)
    selection.to_csv(results_dir / "cluster_model_selection.csv", index=False)

    summary = {
        "project": "Wholesale Customer Segmentation: Interactive Analytics",
        "dataset": "UCI Wholesale customers",
        "customer_count": int(len(spending)),
        "features": FEATURES,
        "preprocessing": ["log1p", "StandardScaler"],
        "random_state": int(random_state),
        "metrics": fitted["metrics"],
        "interpretation_note": "Cluster IDs are arbitrary; inspect spending profiles.",
        "stability_note": "ARI compares assignments from five alternate seeds for the selected algorithm.",
    }
    (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    joblib.dump({
        "scaler": fitted["scaler"],
        "model": fitted["model"],
        "features": FEATURES,
        "transform": "log1p",
        "metrics": fitted["metrics"],
    }, artifacts_dir / "customer_segments.joblib")

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels,
                         cmap="tab10", alpha=0.82, edgecolor="white", linewidth=0.3)
    ax.set_title(f"Wholesale customers: {clusters} {algorithm} segments (PCA view)")
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Cluster ID")
    fig.tight_layout()
    fig.savefig(results_dir / "segment_map.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for algorithm_name, group in selection.groupby("algorithm"):
        group = group.sort_values("clusters")
        axes[0, 0].plot(group["clusters"], group["silhouette_score"], marker="o", label=algorithm_name)
        axes[0, 1].plot(group["clusters"], group["davies_bouldin_score"], marker="o", label=algorithm_name)
        axes[1, 0].plot(group["clusters"], group["stability_ari_mean"], marker="o", label=algorithm_name)
    kmeans_rows = selection[selection["algorithm"] == "K-Means"].sort_values("clusters")
    gmm_rows = selection[selection["algorithm"] == "Gaussian Mixture"].sort_values("clusters")
    axes[0, 2].plot(kmeans_rows["clusters"], kmeans_rows["calinski_harabasz_score"], marker="o", label="K-Means")
    axes[0, 2].plot(gmm_rows["clusters"], gmm_rows["calinski_harabasz_score"], marker="s", label="Gaussian Mixture")
    axes[1, 1].plot(kmeans_rows["clusters"], kmeans_rows["inertia"], marker="o", color="#3b78a1")
    axes[1, 2].plot(gmm_rows["clusters"], gmm_rows["aic"], marker="o", label="AIC")
    axes[1, 2].plot(gmm_rows["clusters"], gmm_rows["bic"], marker="s", label="BIC")
    axes[0, 0].set_title("Silhouette (higher is better)")
    axes[0, 1].set_title("Davies-Bouldin (lower is better)")
    axes[1, 0].set_title("Seed stability ARI (higher is more repeatable)")
    axes[0, 2].set_title("Calinski-Harabasz (higher is better)")
    axes[1, 1].set_title("K-Means inertia (elbow heuristic)")
    axes[1, 2].set_title("Gaussian Mixture criteria (lower is better)")
    for ax in axes.flat:
        ax.set_xlabel("Number of clusters")
        ax.grid(alpha=0.22)
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[0, 2].legend()
    axes[1, 2].legend()
    fig.suptitle("Wholesale customer segmentation diagnostics", fontsize=15)
    fig.tight_layout()
    fig.savefig(results_dir / "model_comparison.png", dpi=170)
    plt.close(fig)
    return summary, profiles, selection


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--algorithm", choices=["K-Means", "Gaussian Mixture"], default="K-Means")
    args = parser.parse_args()
    summary, profiles, selection = train_project(args.clusters, args.random_state, args.algorithm)
    print(profiles.to_string(index=False))
    print(json.dumps(summary["metrics"], indent=2))
    print(f"Cluster comparisons saved to {ROOT / 'results' / 'cluster_model_selection.csv'}")


if __name__ == "__main__":
    main()
