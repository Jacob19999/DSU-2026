"""
Embedding features for REASON_VISIT_NAME case-mix.

Implements the full embedding pipeline described in
``Strategies/Data/embedding.md`` §4 and §7:

  1. **Model loading** — tries a biomedical sentence-transformer (SapBERT) as
     the primary encoder, falls back to a generic sentence-transformer
     (all-MiniLM-L6-v2), and finally degrades to a TF-IDF + TruncatedSVD
     pipeline when no GPU / network / sentence-transformers is available.
  2. **Reason-level caching** — unique reason strings are embedded once and
     persisted to ``<cache_dir>/reason_embeddings.parquet`` so subsequent
     ingestion runs skip model inference entirely.
  3. **Weighted mean pooling** — visit-level embeddings are aggregated to
     ``(site, date, block)`` via ed_enc-weighted mean:
         e_block = sum(count_r * e_r) / sum(count_r)
  4. **Summary statistics** — optional per-block norm, entropy of the reason
     mix, and spherical-k-means cluster IDs.

When ``use_reason_embeddings=False`` (default) none of this code runs and
``master_block_history`` is unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import DATA_DIR, DataSourceConfig

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

# Default models — tried in order until one succeeds
_PRIMARY_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache directory lives alongside the other API caches
EMBEDDING_CACHE_DIR = DATA_DIR / "cache"


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Configuration for the reason-embedding pipeline.

    Kept separate from ``DataSourceConfig`` so that embedding-specific
    hyperparameters can evolve independently.
    """

    # ── Model ──────────────────────────────────────────────────────────────
    primary_model: str = _PRIMARY_MODEL
    fallback_model: str = _FALLBACK_MODEL
    dim: int = 64                          # output dimension (SVD / PCA reduction)
    normalize: bool = True                 # L2-normalize all vectors

    # ── Aggregation ────────────────────────────────────────────────────────
    pooling: str = "weighted_mean"         # "weighted_mean" or "mean"

    # ── Extra summary stats ────────────────────────────────────────────────
    add_norm: bool = True                  # ||e_block||₂
    add_entropy: bool = True               # Shannon entropy of reason mix
    add_cluster_ids: bool = True           # spherical k-means cluster ID
    n_clusters: int = 8                    # clusters for reason grouping

    # ── Caching ────────────────────────────────────────────────────────────
    cache_dir: Path = EMBEDDING_CACHE_DIR
    cache_filename: str = "reason_embeddings.parquet"

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42


# ══════════════════════════════════════════════════════════════════════════════
#  Encoder hierarchy: SapBERT → MiniLM → TF-IDF+SVD
# ══════════════════════════════════════════════════════════════════════════════

def _try_sentence_transformer(model_name: str, texts: List[str]) -> Optional[np.ndarray]:
    """Try to encode *texts* with a sentence-transformers model.

    Returns (n_texts, raw_dim) float32 ndarray or None on failure.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
    except ImportError:
        logger.info("sentence-transformers not installed -- skipping %s", model_name)
        return None

    try:
        logger.info("Loading sentence-transformer model: %s", model_name)
        model = SentenceTransformer(model_name)
        vecs = model.encode(
            texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-norm inside the model
        )
        logger.info(
            "Encoded %d reasons -> shape %s with %s",
            len(texts), vecs.shape, model_name,
        )
        return np.asarray(vecs, dtype=np.float32)
    except Exception as exc:
        logger.warning("Failed to load/encode with %s: %s", model_name, exc)
        return None


def _tfidf_svd_encode(texts: List[str], dim: int, seed: int) -> np.ndarray:
    """Last-resort encoder: TF-IDF character n-grams → TruncatedSVD.

    Always succeeds (scikit-learn is a core dependency).
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer

    logger.info(
        "Using TF-IDF + SVD fallback encoder (dim=%d) for %d reasons",
        dim, len(texts),
    )
    # Character n-grams work well for short clinical abbreviations
    actual_dim = min(dim, len(texts) - 1, 300)  # SVD dim ≤ n_features
    pipe = make_pipeline(
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=2000,
            sublinear_tf=True,
        ),
        TruncatedSVD(n_components=actual_dim, random_state=seed),
        Normalizer(norm="l2"),
    )
    vecs = pipe.fit_transform(texts)
    return np.asarray(vecs, dtype=np.float32)


def _reduce_dim(vecs: np.ndarray, target_dim: int, seed: int) -> np.ndarray:
    """Reduce *vecs* to *target_dim* using PCA (or passthrough if already ≤)."""
    if vecs.shape[1] <= target_dim:
        return vecs
    from sklearn.decomposition import PCA

    logger.info("Reducing embedding dim %d -> %d via PCA", vecs.shape[1], target_dim)
    pca = PCA(n_components=target_dim, random_state=seed)
    reduced = pca.fit_transform(vecs)
    return np.asarray(reduced, dtype=np.float32)


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (safe for zero vectors)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vecs / norms


def _embed_reasons(
    reasons: List[str],
    config: EmbeddingConfig,
) -> np.ndarray:
    """Embed a list of unique reason strings → (n_reasons, dim) ndarray.

    Tries: primary model → fallback model → TF-IDF+SVD.
    """
    # 1. Primary biomedical encoder
    vecs = _try_sentence_transformer(config.primary_model, reasons)

    # 2. Generic sentence-transformer fallback
    if vecs is None:
        vecs = _try_sentence_transformer(config.fallback_model, reasons)

    # 3. TF-IDF + SVD (always works)
    if vecs is None:
        vecs = _tfidf_svd_encode(reasons, dim=config.dim, seed=config.seed)

    # Reduce dimension if the encoder's native dim > config.dim
    vecs = _reduce_dim(vecs, target_dim=config.dim, seed=config.seed)

    # L2-normalize
    if config.normalize:
        vecs = _l2_normalize(vecs)

    return vecs


# ══════════════════════════════════════════════════════════════════════════════
#  Reason-level caching
# ══════════════════════════════════════════════════════════════════════════════

def _load_cached_embeddings(
    config: EmbeddingConfig,
) -> Optional[pd.DataFrame]:
    """Load cached reason embeddings if available."""
    cache_path = Path(config.cache_dir) / config.cache_filename
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            logger.info("Loaded cached reason embeddings from %s (%d rows)", cache_path, len(df))
            return df
        except Exception as exc:
            logger.warning("Failed to read embedding cache: %s", exc)
    return None


def _save_cached_embeddings(
    reason_df: pd.DataFrame,
    config: EmbeddingConfig,
) -> None:
    """Persist reason embeddings to parquet cache."""
    cache_path = Path(config.cache_dir) / config.cache_filename
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        reason_df.to_parquet(cache_path, index=False)
        logger.info("Saved reason embeddings cache: %s", cache_path)
    except Exception as exc:
        logger.warning("Failed to write embedding cache: %s", exc)


def _get_reason_embeddings(
    unique_reasons: List[str],
    config: EmbeddingConfig,
) -> pd.DataFrame:
    """Return a DataFrame of (reason_visit_name, reason_emb_0, ..., reason_emb_{d-1}).

    Uses cache when available; otherwise computes and caches.
    """
    # Try cache first
    cached = _load_cached_embeddings(config)
    if cached is not None:
        cached_reasons = set(cached["reason_visit_name"])
        needed = [r for r in unique_reasons if r not in cached_reasons]
        if not needed:
            # Cache covers all reasons — filter to what we need
            return cached[cached["reason_visit_name"].isin(set(unique_reasons))].copy()
        else:
            logger.info(
                "%d new reasons not in cache -- recomputing all embeddings", len(needed),
            )

    # Compute from scratch
    vecs = _embed_reasons(unique_reasons, config)
    dim = vecs.shape[1]

    emb_cols = [f"reason_emb_{i}" for i in range(dim)]
    reason_df = pd.DataFrame(vecs, columns=emb_cols)
    reason_df.insert(0, "reason_visit_name", unique_reasons)

    # Persist
    _save_cached_embeddings(reason_df, config)

    return reason_df


# ══════════════════════════════════════════════════════════════════════════════
#  Block-level aggregation (weighted mean pooling)
# ══════════════════════════════════════════════════════════════════════════════

def _weighted_mean_pool(
    visits: pd.DataFrame,
    reason_emb_df: pd.DataFrame,
    block_df: pd.DataFrame,
    config: EmbeddingConfig,
) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregate visit-level reason embeddings to (site, date, block).

    Formula:  e_block = sum(count_r * e_r) / sum(count_r)
    where count_r = ed_enc for that reason in that block.

    The pre-normalization norm is stored in ``_raw_norm`` so that
    ``_add_block_norm`` can expose it as a feature *before* we
    L2-normalize the final block vectors.  (A weighted mean of unit
    vectors has ||e|| < 1 when the mix is dispersed, and ||e|| ~ 1 when
    dominated by one reason — that signal is lost after re-normalization.)
    """
    emb_cols = [c for c in reason_emb_df.columns if c.startswith("reason_emb_")]
    dim = len(emb_cols)

    # 1. Aggregate visits to (site, date, block, reason_visit_name) counts
    visit_agg = (
        visits
        .groupby(["site", "date", "block", "reason_visit_name"], as_index=False)["ed_enc"]
        .sum()
    )

    # 2. Join embeddings onto visit-level aggregates
    merged = visit_agg.merge(reason_emb_df, on="reason_visit_name", how="left")

    # 3. Weighted mean pooling per (site, date, block)
    for col in emb_cols:
        merged[f"_w_{col}"] = merged["ed_enc"] * merged[col]

    w_cols = [f"_w_{c}" for c in emb_cols]
    agg_dict = {wc: "sum" for wc in w_cols}
    agg_dict["ed_enc"] = "sum"

    block_emb = merged.groupby(["site", "date", "block"], as_index=False).agg(agg_dict)

    # Divide weighted sums by total count
    for i, col in enumerate(emb_cols):
        block_emb[col] = np.where(
            block_emb["ed_enc"] > 0,
            block_emb[w_cols[i]] / block_emb["ed_enc"],
            0.0,
        )

    # Drop working columns
    block_emb = block_emb.drop(columns=w_cols + ["ed_enc"])

    # Capture pre-normalization norm (concentration of reason mix).
    # Weighted mean of L2-unit vectors: ||e_block|| in (0, 1].
    #   ~1  → block dominated by one reason (or very similar reasons)
    #   <1  → block has a dispersed, heterogeneous reason mix
    block_emb["_raw_norm"] = np.linalg.norm(
        block_emb[emb_cols].values, axis=1,
    )

    # Re-normalize block-level vectors so that cosine similarity between
    # any two block vectors equals their dot product (strategy doc §3).
    if config.normalize:
        arr = block_emb[emb_cols].values
        block_emb[emb_cols] = _l2_normalize(arr)

    return block_emb, emb_cols


# ══════════════════════════════════════════════════════════════════════════════
#  Extra summary statistics
# ══════════════════════════════════════════════════════════════════════════════

def _add_block_norm(block_df: pd.DataFrame, emb_cols: List[str]) -> pd.DataFrame:
    """Add pre-normalization ||e_block||₂ as a feature.

    This is the norm of the weighted-mean vector *before* L2 re-normalization.
    It lives in (0, 1] for blocks with visits and captures how concentrated
    the reason mix is:
      ~1  → dominated by one reason (or very similar reasons)
      <1  → dispersed / heterogeneous reason mix
    """
    if "_raw_norm" in block_df.columns:
        # Use the pre-normalization norm saved by _weighted_mean_pool
        block_df["reason_emb_norm"] = block_df["_raw_norm"].fillna(0.0)
        block_df = block_df.drop(columns=["_raw_norm"])
    else:
        # Fallback: compute from current (possibly re-normalized) vectors
        block_df["reason_emb_norm"] = np.linalg.norm(
            block_df[emb_cols].values, axis=1,
        )
    return block_df


def _add_reason_entropy(
    visits: pd.DataFrame,
    block_df: pd.DataFrame,
) -> pd.DataFrame:
    """Shannon entropy of the reason distribution per (site, date, block)."""
    visit_agg = (
        visits
        .groupby(["site", "date", "block", "reason_visit_name"], as_index=False)["ed_enc"]
        .sum()
    )
    # Compute per-block totals
    block_totals = visit_agg.groupby(["site", "date", "block"])["ed_enc"].transform("sum")
    probs = visit_agg["ed_enc"] / block_totals.clip(lower=1)

    # Shannon entropy: -sum(p * log(p))
    visit_agg["_h"] = np.where(probs > 0, -probs * np.log2(probs), 0.0)

    entropy = (
        visit_agg
        .groupby(["site", "date", "block"], as_index=False)["_h"]
        .sum()
        .rename(columns={"_h": "reason_emb_entropy"})
    )

    block_df = block_df.merge(entropy, on=["site", "date", "block"], how="left")
    block_df["reason_emb_entropy"] = block_df["reason_emb_entropy"].fillna(0.0)
    return block_df


def _add_cluster_ids(
    reason_emb_df: pd.DataFrame,
    visits: pd.DataFrame,
    block_df: pd.DataFrame,
    n_clusters: int,
    seed: int,
) -> pd.DataFrame:
    """Assign each reason to a cluster via spherical k-means, then add
    the dominant cluster ID per block as a categorical feature.
    """
    from sklearn.cluster import KMeans

    emb_cols = [c for c in reason_emb_df.columns if c.startswith("reason_emb_")]
    X = reason_emb_df[emb_cols].values
    actual_k = min(n_clusters, len(X))

    km = KMeans(n_clusters=actual_k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)

    reason_clusters = reason_emb_df[["reason_visit_name"]].copy()
    reason_clusters["_cluster"] = labels

    # Aggregate visits to (site, date, block, reason) and join cluster
    visit_agg = (
        visits
        .groupby(["site", "date", "block", "reason_visit_name"], as_index=False)["ed_enc"]
        .sum()
    )
    visit_agg = visit_agg.merge(reason_clusters, on="reason_visit_name", how="left")

    # Dominant cluster per block = cluster with highest total ed_enc
    cluster_weight = (
        visit_agg
        .groupby(["site", "date", "block", "_cluster"], as_index=False)["ed_enc"]
        .sum()
    )
    idx_max = cluster_weight.groupby(["site", "date", "block"])["ed_enc"].idxmax()
    dominant = cluster_weight.loc[idx_max, ["site", "date", "block", "_cluster"]]
    dominant = dominant.rename(columns={"_cluster": "reason_emb_cluster"})

    block_df = block_df.merge(dominant, on=["site", "date", "block"], how="left")
    block_df["reason_emb_cluster"] = block_df["reason_emb_cluster"].fillna(-1).astype(int)
    return block_df


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def add_embedding_features(
    block_df: pd.DataFrame,
    visits: pd.DataFrame,
    *,
    data_config: Optional[DataSourceConfig] = None,
    embed_config: Optional[EmbeddingConfig] = None,
) -> pd.DataFrame:
    """
    Add block-level embedding features derived from REASON_VISIT_NAME.

    Implements the full pipeline from ``Strategies/Data/embedding.md``:

    1. Extract unique reason strings from ``visits``.
    2. Embed each reason via the encoder hierarchy
       (SapBERT → MiniLM → TF-IDF+SVD), with parquet-level caching.
    3. Aggregate to block level via ed_enc-weighted mean pooling.
    4. Append ``reason_emb_0`` … ``reason_emb_{d-1}`` columns to *block_df*.
    5. Optionally add summary stats: norm, entropy, dominant cluster ID.

    Parameters
    ----------
    block_df:
        Current block-level master DataFrame, after core targets and case-mix
        counts have been added.  One row per (site, date, block).
    visits:
        Raw visit-level DataFrame as returned by ``load_visits``, including
        ``reason_visit_name`` and ``ed_enc``.
    data_config:
        Optional ``DataSourceConfig`` (currently unused, reserved for future
        path overrides).
    embed_config:
        Optional ``EmbeddingConfig`` controlling model, dimension, pooling,
        and extra-stat flags.  Uses sensible defaults when ``None``.

    Returns
    -------
    pd.DataFrame
        ``block_df`` with additional numeric columns:
        - ``reason_emb_0`` … ``reason_emb_{d-1}``  (embedding vector)
        - ``reason_emb_norm``       (if ``add_norm=True``)
        - ``reason_emb_entropy``    (if ``add_entropy=True``)
        - ``reason_emb_cluster``    (if ``add_cluster_ids=True``)
    """
    cfg = embed_config or EmbeddingConfig()

    # ── 1. Unique reasons ─────────────────────────────────────────────────
    if "reason_visit_name" not in visits.columns:
        logger.warning("visits DataFrame missing 'reason_visit_name' -- skipping embeddings")
        return block_df

    unique_reasons = sorted(visits["reason_visit_name"].dropna().unique().tolist())
    if not unique_reasons:
        logger.warning("No non-null reason_visit_name values -- skipping embeddings")
        return block_df

    logger.info("Embedding %d unique REASON_VISIT_NAME values (dim=%d)", len(unique_reasons), cfg.dim)

    # ── 2. Embed (with caching) ──────────────────────────────────────────
    reason_emb_df = _get_reason_embeddings(unique_reasons, cfg)
    emb_cols = [c for c in reason_emb_df.columns if c.startswith("reason_emb_")]
    logger.info("Reason embedding matrix: %d reasons x %d dims", len(reason_emb_df), len(emb_cols))

    # ── 3. Weighted mean pooling to (site, date, block) ──────────────────
    block_emb, emb_cols = _weighted_mean_pool(visits, reason_emb_df, block_df, cfg)

    # ── 4. Merge onto block_df ───────────────────────────────────────────
    block_df = block_df.merge(
        block_emb, on=["site", "date", "block"], how="left",
    )
    # Fill any blocks with no visits → zero vector
    block_df[emb_cols] = block_df[emb_cols].fillna(0.0)

    # ── 5. Summary statistics ────────────────────────────────────────────
    if cfg.add_norm:
        block_df = _add_block_norm(block_df, emb_cols)
    elif "_raw_norm" in block_df.columns:
        block_df = block_df.drop(columns=["_raw_norm"])

    if cfg.add_entropy:
        block_df = _add_reason_entropy(visits, block_df)

    if cfg.add_cluster_ids:
        block_df = _add_cluster_ids(
            reason_emb_df, visits, block_df,
            n_clusters=cfg.n_clusters,
            seed=cfg.seed,
        )

    n_new = len(emb_cols) + cfg.add_norm + cfg.add_entropy + cfg.add_cluster_ids
    logger.info("Added %d embedding-derived columns to master", n_new)

    return block_df
