# Project Report: Wholesale Customer Segmentation with K-Means

## Objective
Explore groups of wholesale customers with similar annual spending patterns. This is unsupervised learning because no customer segment labels are supplied.

## Dataset
The UCI Wholesale customers dataset has 440 records and six numeric spending features: Fresh, Milk, Grocery, Frozen, Detergents_Paper, and Delicassen. Channel and Region are retained for descriptive comparison but excluded from fitting. See data/README.md for the citation and license.

## Method
The workflow validates the CSV, applies log1p to reduce skew, standardizes the spending features, then fits K-Means with a fixed seed. It compares candidate cluster counts using inertia and silhouette score, summarizes average feature spending by cluster, and visualizes the results with a two-component PCA projection. Preprocessing and the fitted model are saved for future inference.

## Results
Run python train.py to generate metrics.json, cluster_profiles.csv, cluster_selection.csv, customer assignments, and a PCA image under artifacts/. The chart is a two-dimensional projection and does not preserve every distance from the full feature space.

## Limitations
The dataset is small and comes from one wholesale distributor. K-Means assumes compact groups under Euclidean distance. These clusters are exploratory examples, not validated business segments.
