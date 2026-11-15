# Project Report: Wholesale Customer Segmentation Studio

## Objective
Explore customer groups based on annual spending across six product categories and provide an interactive way to inspect segment quality and profiles.

## Dataset
The UCI Wholesale customers dataset has 440 records. Six spending fields are used for clustering. Channel and Region are excluded from fitting and used only for descriptive comparison. Dataset attribution and CC BY 4.0 licensing are documented in data/README.md.

## Method
The workflow validates values, applies log1p to reduce skew, standardizes six features, and compares K-Means and full-covariance Gaussian Mixture models from k=2 through k=8. Both methods are compared with silhouette, Davies-Bouldin, Calinski-Harabasz, and adjusted Rand stability across five alternate seeds. K-Means reports inertia; Gaussian Mixture reports AIC/BIC and soft membership probabilities. PCA is used only as a two-dimensional display projection. The dashboard reports spending signatures, channel/region mixes, and sample customers.

## Results
On the fixed seed, K-Means at k=2 scored 0.2903 silhouette, 1.3515 Davies-Bouldin, and 1.0000 mean ARI stability; Gaussian Mixture at k=2 scored 0.2656, 1.3908, and 1.0000, respectively. K-Means at k=2 had the top silhouette among the 14 method/count settings. Run `python train.py` to regenerate metrics, profiles, assignments, and both comparison figures. Internal scores are heuristics, so the segments remain exploratory rather than validated business groups.

## Limitations
The sample represents one historical wholesale distributor. Cluster IDs have no inherent order, and internal metrics do not establish business value. The results should not be generalized to other populations or used as validated marketing policy.
