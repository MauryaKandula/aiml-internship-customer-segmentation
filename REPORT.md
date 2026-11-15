# Project Report: Wholesale Customer Segmentation Studio

## Objective
Explore customer groups based on annual spending across six product categories and provide an interactive way to inspect segment quality and profiles.

## Dataset
The UCI Wholesale customers dataset has 440 records. Six spending fields are used for clustering. Channel and Region are excluded from fitting and used only for descriptive comparison. Dataset attribution and CC BY 4.0 licensing are documented in data/README.md.

## Method
The workflow validates values, applies log1p to reduce skew, standardizes six features, and fits reproducible K-Means. Candidate counts are compared using inertia, silhouette, Davies-Bouldin, and Calinski-Harabasz scores. Five alternate random seeds are compared with adjusted Rand agreement to assess partition stability. PCA provides a two-dimensional display projection. The dashboard reports spending signatures, channel/region mixes, and sample customers.

## Results
On the bundled dataset, two clusters gave the highest silhouette score among k=2 through k=8: 0.2903. The Davies-Bouldin score was 1.3515, and mean adjusted Rand agreement across five alternate seeds was 1.0000. This indicates stable assignments with modest separation; the segments remain exploratory rather than validated business groups. Run python train.py to regenerate the metrics, profiles, assignments, and PCA figure.

## Limitations
The sample represents one historical wholesale distributor. Cluster IDs have no inherent order, and internal metrics do not establish business value. The results should not be generalized to other populations or used as validated marketing policy.
