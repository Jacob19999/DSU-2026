## Reason Embedding Strategy (`REASON_VISIT_NAME`)

This document defines how we will add **semantic embeddings** of `REASON_VISIT_NAME`
into the block-level master history, and how downstream pipelines are expected to
consume those features.

The goals:

- Capture **clinical similarity** between reasons (e.g. "CHEST PAIN" ≈ "CP",
  "SOB" ≈ "SHORTNESS OF BREATH") beyond simple top-N counts.
- Provide a **dense, low-dimensional representation** that can be aggregated to
  the `(site, date, block)` level and used by any GBDT-based pipeline.
- Preserve the existing evaluation contract (no new targets, same grid, no
  leakage).

---

### 1. High-level design

We introduce an **optional** embedding path in the Data Source layer:

- **Input**: visit-level `REASON_VISIT_NAME` (string) with per-visit encounters.
- **Model**: biomedical text encoder → fixed-length vector for each reason.
- **Aggregation**: pool visit-level vectors to **block-level** embeddings:
  - 1 row per `(site, date, block)`,
  - 1–k numeric columns per row (e.g. `reason_emb_0` … `reason_emb_{k-1}`).
- **Output**: additional numeric columns **appended** to `master_block_history`
  when enabled.

This is wired by the `DataSourceConfig.use_reason_embeddings` flag and the CLI
flag:

```bash
python "Pipelines/data_source/run_data_source.py" \
  --use-reason-embeddings
```

When **disabled** (default), `master_block_history` is unchanged and pipelines
behave exactly as before.

---

### 2. Embedding model choice

We distinguish between **primary** and **fallback** encoders.

- **Primary encoder (medical)**:
  - A **biomedical sentence-level Transformer** (e.g. ClinicalBERT/BioBERT
    variant wrapped as a sentence-transformer, or SapBERT-style concept model).
  - Requirements:
    - Accept a short text (`REASON_VISIT_NAME`) and return a single vector.
    - Trained / fine-tuned for **sentence or concept similarity**, not just
      masked LM.
    - Supports **L2-normalized embeddings** where cosine similarity encodes
      semantic similarity.
  - Rationale:
    - Reason strings are short, clinical phrases; generic encoders miss a lot
      of domain-specific structure (e.g. "STEMI", "PNA", "DKA").
    - Concept-aligned biomedical models (SapBERT-style) explicitly pull
      synonyms together in embedding space, which is ideal for grouping reasons.

- **Fallback encoder (generic)**:
  - A generic **sentence-transformer** (e.g. `all-MiniLM-L6-v2`) or a simple
    TF–IDF + SVD pipeline.
  - Used only when the primary biomedical model is unavailable in the runtime
    environment.
  - Still produces a fixed-length vector per reason; quality is lower, but
    representation remains compatible with the same aggregation and downstream
    feature usage.

Implementation detail:

- The actual model loading / inference will live **outside** the Data Source
  ingestion loop (e.g. cached per unique reason), to avoid per-row latency.
- Data Source only assumes:
  - we have a mapping `reason_visit_name -> np.ndarray[dim]` at ingestion time.

---

### 3. Similarity metric and grouping behaviour

Although this project only needs **embeddings as features**, we design them to
support **reason grouping** and diagnostics:

- **Vector normalization**:
  - We L2-normalize all reason vectors:
    - `e_hat = e / ||e||_2`
- **Similarity metric**:
  - Use **cosine similarity**:
    - `sim(i, j) = e_hat_i · e_hat_j`
  - And corresponding cosine distance:
    - `dist(i, j) = 1 - sim(i, j)`
- **Potential use-cases**:
  - Clustering reasons into latent macro-groups via:
    - spherical k-means on normalized embeddings,
    - or hierarchical clustering with cosine distance.
  - Identifying near-duplicates / synonym sets from nearest neighbours in
    embedding space.

The **master dataset itself** does *not* need explicit group labels; only the
block-level embedding vectors are required for the current pipelines. Clusters
and similarities are primarily for analysis and debugging.

---

### 4. Data Source contract (`embedding.add_embedding_features`)

The Data Source layer exposes a hook in `Pipelines/data_source/embedding.py`:

- Function:

```python
def add_embedding_features(
    block_df: pd.DataFrame,
    visits: pd.DataFrame,
    *,
    data_config: Optional[DataSourceConfig] = None,
    embed_config: Optional[EmbeddingConfig] = None,
) -> pd.DataFrame:
    ...
```

- **Inputs**:
  - `block_df`: current block-level master frame (one row per
    `(site, date, block)`), with core targets and case-mix counts already
    attached.
  - `visits`: the **visit-level** DataFrame from `load_visits`, including
    `reason_visit_name` and `ed_enc`.
- **Outputs**:
  - A DataFrame with the same rows as `block_df`,
  - plus **zero or more** numeric columns derived from embeddings.

Planned behaviour (not yet implemented, currently a no-op):

1. Build a lookup table of **unique reasons**:
   - Extract unique `reason_visit_name` values from `visits`.
   - For each, compute an embedding using the **primary** (or fallback) model.
2. Aggregate to **block level**:
   - For each `(site, date, block)`:
     - Gather all visits and their reasons,
     - Pull each reason’s embedding, weight by its `ed_enc` count,
     - Compute a **weighted mean** vector:
       - `e_block = sum(count_r * e_r) / sum(count_r)` (if sum > 0).
3. Emit features:
   - For dimension `d`, add columns:
     - `reason_emb_0` … `reason_emb_{d-1}`.
   - All columns must be numeric and free of NaN for **valid** rows.

Today, `add_embedding_features` is a **pure no-op stub**; it simply returns
`block_df` unchanged. This lets us wire config and pipeline contracts without
changing behaviour.

---

### 5. CLI / config wiring (`--use-reason-embeddings`)

The Data Source runner `Pipelines/data_source/run_data_source.py` exposes:

- CLI flag:

```bash
--use-reason-embeddings
```

- Config field:

```python
DataSourceConfig.use_reason_embeddings: bool
```

Behaviour:

- When `use_reason_embeddings=False` (default):
  - `add_embedding_features` is **not** called.
  - `master_block_history` has **no embedding columns**.
- When `use_reason_embeddings=True`:
  - Ingestion calls `add_embedding_features(...)` between:
    - step 4 (case-mix counts) and step 5 (calendar / temporal features).
  - Any new embedding columns produced are persisted into:
    - `master_block_history.parquet`
    - `master_block_history.csv`

This is **backwards compatible** for existing pipelines that ignore unknown
columns or select features generically.

---

### 6. Pipeline consumption of embedding features

All current pipelines select numeric feature columns using **exclusion-based**
rules:

- **Pipeline A** (`step_02_feature_eng.get_feature_columns`):
  - Excludes identifiers (`site`, `date`), targets (`total_enc`, `admitted_enc`,
    `admit_rate`), weights, raw `count_reason_*` counts, and most current-period
    `share_reason_*` features.
  - Any **numeric column not excluded** (including a future `reason_emb_*`)
    will be used as a model feature.

- **Pipeline B** (`Pipeline B/features.py:get_feature_columns`):

```python
_EXCLUDE_COLS = {
    "site", "date", "total_enc", "admitted_enc", "admit_rate",
    "sample_weight", "sample_weight_rate",
    "event_name", "event_type", "is_holiday",
    "bucket",
}

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = set(_EXCLUDE_COLS)
    exclude.update(c for c in df.columns if c.startswith("count_reason_"))
    return sorted(c for c in df.columns if c not in exclude)
```

  - Embedding columns named `reason_emb_*` will **not** be excluded and thus
    will be added to the feature set automatically.

- **Pipeline E** (`Pipeline E/features.py:get_feature_columns`):
  - Excludes identifiers, targets, raw `count_reason_*` / `share_*`, and
    internal factor state (`factor_i`, some `factor_i_lag_*`).
  - Does **not** exclude `reason_emb_*` by prefix, so those columns will be
    included when present.

Other pipelines (C, D) either:

- Work off the same `master_block_history` and use exclusion rules similar to
  A/B/E, or
- Explicitly list numeric features, in which case we can append
  `reason_emb_*` columns later if they prove useful.

**Contract:**  
> If the Data Source produces numeric columns matching `reason_emb_*`, and
> `--use-reason-embeddings` is enabled when building `master_block_history`, the
> main GBDT pipelines (A, B, E) will automatically include those columns as
> features in both **training** and **validation**.

No changes are required in their code paths beyond the generic
`get_feature_columns` logic already present.

---

### 7. Future work

- Implement `add_embedding_features` with:
  - Actual ClinicalBERT/BioBERT/SapBERT model loading,
  - Offline caching of reason-level vectors,
  - Weighted mean pooling at block level.
- Experiment with:
  - Different embedding dimensions (e.g. 64 vs 256+),
  - Additional summary stats (norms, entropy of reason mix, cluster IDs),
  - Ablation studies to quantify incremental value vs existing PCA/NMF factor
    pipeline (Pipeline E).

