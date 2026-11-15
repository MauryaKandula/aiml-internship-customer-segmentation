"""Interactive Streamlit dashboard for customer segmentation."""
from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from train import FEATURES, fit_segmentation, load_records, score_cluster_counts, spending_profiles

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Wholesale customers data.csv"


@st.cache_data
def load_default_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def cached_score_cluster_counts(spending, random_state):
    return score_cluster_counts(spending, random_state)


def read_upload(upload):
    records = pd.read_csv(upload)
    missing = [name for name in FEATURES if name not in records.columns]
    if missing:
        raise ValueError("Uploaded CSV is missing: " + ", ".join(missing))
    spending = records[FEATURES].apply(pd.to_numeric, errors="coerce")
    if spending.isna().any().any() or (spending < 0).any().any():
        raise ValueError("Spending columns must be numeric, complete, and non-negative.")
    if len(spending) < 9:
        raise ValueError("Upload at least nine customer rows.")
    return records, spending


def main():
    st.set_page_config(page_title="Customer Segmentation Studio", page_icon="C", layout="wide")
    st.title("Wholesale Customer Segmentation Studio")
    st.caption("Compare hard and probabilistic clustering, inspect customer profiles, and assign new records.")
    st.sidebar.header("Analysis settings")
    upload = st.sidebar.file_uploader("Optional: use a compatible customer CSV", type=["csv"])
    algorithm = st.sidebar.selectbox("Clustering method", ["K-Means", "Gaussian Mixture"])
    clusters = st.sidebar.slider("Number of clusters", 2, 8, 2)
    random_state = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

    try:
        if upload is None:
            records, spending = load_records(DATA_PATH)
            source = "Bundled UCI dataset"
        else:
            records, spending = read_upload(upload)
            source = upload.name
    except (ValueError, OSError, pd.errors.ParserError) as error:
        st.error(str(error))
        st.stop()

    with st.spinner("Building segments and calculating quality metrics..."):
        fitted = fit_segmentation(spending, clusters, int(random_state), algorithm)
        labels = fitted["labels"]
        profiles = spending_profiles(spending, labels)
        scores = cached_score_cluster_counts(spending, int(random_state))
        coordinates = PCA(n_components=2, random_state=int(random_state)).fit_transform(fitted["scaled"])

    st.sidebar.subheader("Assign a new customer")
    with st.sidebar.form("new_customer_form"):
        values = [
            st.number_input(name, min_value=0.0, value=float(spending[name].median()), step=100.0)
            for name in FEATURES
        ]
        submitted = st.form_submit_button("Assign segment")
    if submitted:
        new_scaled = fitted["scaler"].transform(np.log1p(np.asarray(values).reshape(1, -1)))
        segment = int(fitted["model"].predict(new_scaled)[0])
        st.sidebar.success(f"Assigned to cluster {segment}")
        if hasattr(fitted["model"], "predict_proba"):
            membership = float(fitted["model"].predict_proba(new_scaled)[0, segment])
            st.sidebar.caption(f"Gaussian mixture membership: {membership:.1%}")

    st.caption(f"Data source: {source} | {len(spending):,} customer records | {len(FEATURES)} spending features")
    a, b, c, d = st.columns(4)
    a.metric("Method / clusters", f"{algorithm} / {clusters}")
    b.metric("Silhouette", f"{fitted['metrics']['silhouette_score']:.3f}")
    c.metric("Davies-Bouldin", f"{fitted['metrics']['davies_bouldin_score']:.3f}",
             help="Lower values generally indicate tighter, better-separated clusters.")
    d.metric("ARI stability (5 seeds)", f"{fitted['metrics']['stability_ari_mean']:.3f}", help="Mean adjusted Rand agreement across alternate random seeds.")
    st.caption(f"Analyzed {len(spending):,} customers across six transformed spending features. Internal scores compare structure; they do not establish business value.")

    map_tab, profile_tab, compare_tab = st.tabs(
        ["Segment map", "Profiles and customers", "Choose cluster count"]
    )
    with map_tab:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for cluster in sorted(np.unique(labels)):
            mask = labels == cluster
            ax.scatter(coordinates[mask, 0], coordinates[mask, 1],
                       label=f"Cluster {cluster} ({mask.sum()})", alpha=0.8)
        ax.set(title="Spending segments projected into two PCA dimensions",
               xlabel="Principal component 1", ylabel="Principal component 2")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("PCA is a display projection; clustering uses all six transformed spending features.")

    with profile_tab:
        st.subheader("Average spending profiles")
        st.dataframe(profiles, hide_index=True)
        relative = profiles.set_index("cluster")[FEATURES].div(spending.mean()).round(2)
        st.write("Average spend relative to the full dataset (1.0 means equal to overall average)")
        st.bar_chart(relative.T)
        selected = st.selectbox("Inspect customer records in cluster", sorted(np.unique(labels)))
        subset = records.loc[labels == selected].copy()
        subset["cluster"] = selected
        subset["total_spend"] = spending.loc[labels == selected].sum(axis=1).to_numpy()
        st.dataframe(subset.sort_values("total_spend", ascending=False).head(15), hide_index=True)
        if "Channel" in records.columns:
            st.write("Channel mix by cluster (row percentages)")
            st.dataframe(pd.crosstab(labels, records["Channel"], normalize="index").mul(100).round(1))
        if "Region" in records.columns:
            st.write("Region mix by cluster (row percentages)")
            st.dataframe(pd.crosstab(labels, records["Region"], normalize="index").mul(100).round(1))
        exported = records.copy()
        exported["cluster"] = labels
        if hasattr(fitted["model"], "predict_proba"):
            exported["membership_probability"] = fitted["model"].predict_proba(fitted["scaled"]).max(axis=1)
        st.download_button("Download assignments CSV", exported.to_csv(index=False),
                           file_name="customer_segments.csv", mime="text/csv")

    with compare_tab:
        st.subheader("Compare methods and cluster counts")
        st.dataframe(scores, hide_index=True)
        st.write("**Silhouette** (higher is better)")
        st.line_chart(scores.pivot(index="clusters", columns="algorithm", values="silhouette_score"))
        st.write("**Davies-Bouldin** (lower is better)")
        st.line_chart(scores.pivot(index="clusters", columns="algorithm", values="davies_bouldin_score"))
        st.write("**Assignment stability across five alternate seeds**")
        st.line_chart(scores.pivot(index="clusters", columns="algorithm", values="stability_ari_mean"))
        kmeans_scores = scores[scores["algorithm"] == "K-Means"].set_index("clusters")
        st.write("**K-Means inertia** (lower as k grows; use the elbow as a heuristic)")
        st.line_chart(kmeans_scores[["inertia"]])
        gmm_scores = scores[scores["algorithm"] == "Gaussian Mixture"].set_index("clusters")
        st.write("**Gaussian Mixture information criteria** (lower is preferred within this model family)")
        st.line_chart(gmm_scores[["aic", "bic"]])
        best = scores.loc[scores["silhouette_score"].idxmax()]
        st.write(f"Highest silhouette in the comparison: {best['algorithm']} with {int(best['clusters'])} clusters ({best['silhouette_score']:.3f}).")
        st.caption("ARI stability compares repeated-seed partitions; 1 means identical assignments up to label order. AIC and BIC apply to Gaussian Mixtures. Treat every diagnostic as a heuristic and verify that the profiles are useful for the intended business question.")


if __name__ == "__main__":
    main()
