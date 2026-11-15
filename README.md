# Wholesale Customer Segmentation with K-Means

An unsupervised learning project that groups wholesale customers by annual spending in six product categories. The included UCI dataset is processed with log1p and standardization before K-Means. The workflow saves customer assignments, cluster profiles, model-selection scores, metrics, a model, and a PCA visualization.

## Setup
Use Python 3.10 or newer.

    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt

The CSV is included under data/ and licensed CC BY 4.0. Attribution is in data/README.md.

## Run
    python train.py
    python predict.py --spend 6000 800 9000 300 1000 250

The default model has four clusters. Use python train.py --clusters 3 to change it. Outputs are saved under artifacts/. The selection CSV compares counts from 2 to 8 using inertia and silhouette score; use the scores with the project purpose because no single metric validates a business segment.

The six prediction values must be ordered Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen. The dataset contains 440 records. Channel and Region are kept for description and excluded from fitting. See REPORT.md for method and limitations.
