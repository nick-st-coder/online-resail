# Online Retail Customer Segmentation

Customer segmentation with **unsupervised learning** on the [Online Retail II](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) dataset — a UK-based, non-store online retailer selling unique all-occasion gift-ware (Dec 2009 – Dec 2011), with many wholesale customers.

My project discovers distinct customer segments from transactional purchasing behavior and determines the most appropriate number of clusters using multiple unsupervised techniques, such as (K-Means, Gaussian Mixture Models, agglomerative clustering) and cluster validity metrics (silhouette, Davies-Bouldin, inertia/elbow, BIC/AIC).

---

## Business problem I solved

The company has no labels telling it who its customers are — only raw transactions. The goal is to `find hidden patterns in purchasing behavior` and group customers into actionable segments, so the business can:
- **Identify customers at risk of churning** (long recency, low frequency) and start reactivation campaigns — or stop spending budget on them.

- **Recognize high-value customers** and target them with personalized offers to grow their spend even further.

- Understand the overall **value ladder** of the customer base to allocate marketing and retention effort proportionally.

## What I've built
1. **EDA notebook** (`notebooks/EDA.ipynb`) — data cleaning, outlier analysis, RFM analysis, feature engineering, and a customer-level feature matrix.

2. **Modeling notebook** (`notebooks/modeling.ipynb`) — cluster selection (elbow / silhouette / Davies-Bouldin), cross-validation with GMM (BIC/AIC) and agglomerative clustering, and the final K-Means model with segment interpretation.

3. **Production code** (`src/`) — feature building, the clustering model, an MLflow-tracked pyfunc wrapper, and a FastAPI + Gradio serving app.

4. **CI** (`.github/workflows/ci.yml`) — lint, tests, and a smoke validation of the registered MLflow model.

## Problems I faced during the project

### Data quality
- **~25% of rows had missing `Customer ID`** — dropped, because customer-level clustering is impossible without knowing which customer bought what.

- **Duplicated rows** — identical invoice/stockcode/customer rows were dropped so they don't distort the model.

- **Cancelled orders** — invoices starting with `C` (e.g. `C489xxx`) were flagged and aggregated into a `sum_cancelled_orders` feature, useful for churn-risk analysis.

- **Invalid price/quantity rows** — negative or zero values (including test stock codes like `TEST001` and `M`) were dropped.

- **Wrong dtypes** — `InvoiceDate` was coerced to datetime.

### Skewness and outliers
- **Heavy right-skew** on every monetary/volume feature (skewness 6–52) — required `log1p` + `StandardScaler` before K-Means; only `recency_days` and `customer_lifetime_days` stayed near-raw.

- **Product-level outliers are real sales** (very cheap items sold in bulk, or very expensive items sold once) — kept.

- **Two "whale" customers** (rare, huge-quantity purchases) distorted the cluster plots — removed from the analysis set so the remaining segments could be studied properly.

### Modeling pitfalls
- **Row-level duplication** — 68% of rows were exact duplicate feature vectors (the same customer repeated across transactions), making clusters look overlapping. Fixed by clustering on **customer-level** data (one row per customer).

- **PCA projection loss** — 2 PCs explained only ~52% of variance (65% after log1p), so 2D plots squash real separation; the overlap was mostly a projection artifact, not bad clustering.

- **Metric disagreement on k** — elbow suggested k=3, silhouette k=2, Davies-Bouldin k=5. Resolved by choosing the best trade-off (see below).

- **GMM BIC/AIC kept decreasing through k=5** — the data has finer structure than 3 segments under GMM's assumptions; this doesn't invalidate the K-Means k=3, but it's an honest caveat.

- **Agglomerative clustering is O(n²)** — subsampled to 1,000 customers for the dendrogram.

## Achievements / results

### Data
- Cleaned **779,423 transactions → 5,878 customers** with no remaining nulls, duplicates, or invalid rows.

- Engineered 11 behavioral + RFM-style features per customer (recency, frequency, monetary, order value, cancellations, lifetime, etc.).

### Cluster selection
- Compared K-Means across k=2–5 with three validity metrics, cross-checked with **GMM (BIC/AIC)** and an **agglomerative dendrogram** (Adjusted Rand Index agreement with K-Means at k=3).

- Chose **k=3** — the best trade-off between the elbow, silhouette, and Davies-Bouldin.

### The three segments

| Segment | Share | Profile |
|---|---|---|
| **Occasional bulk buyers** | 34% | Buy expensive products in bulk but rarely (~1.4 orders, once per ~349 days); highest avg unit price (£18.7); rarely cancel. *e.g. a design studio buying premium monitors once a year.* |
| **Everyday value shoppers** | 43% | Largest segment; regular buyers of cheap products (£3.2 avg unit price) ~once per 160 days (4.2 orders); ~£1,237 total spend. *e.g. a household restocking everyday goods.* |
| **High-volume resellers** | 23% | Biggest spenders: 17.4 orders (once per ~57 days), ~608 items per order, ~£9,977 revenue — **8× the middle segment**; cancel the most (typical for bulk operations). *e.g. a wholesaler stocking inventory.* |

**Key takeaway:** the clusters form a clear **value ladder** — occasional premium buyers → regular value shoppers → high-volume resellers. Resellers drive the bulk of revenue and are the most valuable to retain despite their high cancellation rate.

### Business insights
- **Revenue is concentrated**: the top 10% of customers generate ~64% of total revenue.

- **~28% of customers placed only one order** — a large low-frequency tail.

- **~43% of customers cancelled at least one order**; cancellations correlate with revenue (0.63) — heavy buyers cancel more simply because they order more.

- **RFM backbone validated**: Frequency and Monetary are redundant (log-scale r = 0.85); Recency adds independent churn-risk information (r ≈ −0.5 with activity).

- **No leakage or drift**: features are per-customer aggregates over the full window, and monthly order volume is stable apart from a seasonal Nov–Dec bump.

## Trade-offs
- **k=3 vs. k=2/k=5** — silhouette prefers 2, Davies-Bouldin prefers 5; k=3 was chosen for the elbow + interpretability, accepting slightly lower silhouette.

- **Dropping ~25% of rows** (missing `Customer ID`) — necessary for customer-level analysis, but loses potentially valid transactions.

- **Excluding the 2 whale customers** from the analysis set — they distorted visualization; they remain a real (if tiny) segment worth handling separately in practice.

- **PCA over UMAP/t-SNE** — PCA is fast and deterministic on 780k rows, but explains only ~65% of variance in 2D; prettier embeddings would be stochastic and slow without fixing the underlying duplication.

- **K-Means over GMM** — K-Means assumes spherical, equal-variance clusters; GMM's BIC/AIC suggest more components exist, so the "right k" depends on the algorithm's assumptions.

- **Agglomerative on a subsample** — the dendrogram is a visual sanity check on 1,000 of 5,878 customers, not a full-scale fit.

- **`log1p` + scaling** — tames skew and handles zeros, but makes cluster centroids interpretable only on the transformed scale.

## Stack I used

pandas · numpy · matplotlib · seaborn · scikit-learn · scipy · statsmodels · FastAPI · Gradio · `uv` for environment/dependency management.

## Getting started

```bash
uv sync --dev
uv run jupyter notebook notebooks/EDA.ipynb
uv run python scripts/run_clustering.py
uv run python scripts/train_and_register.py
uv run uvicorn src.app.main:app --reload
```