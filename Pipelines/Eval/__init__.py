"""
Eval: Centralized pipeline evaluation package.

Implements the eval.md contract — pipeline-agnostic scoring
from submission-shaped CSVs with cross-pipeline comparison.

Usage (CLI):
    python -m Pipelines.Eval.run_eval              # score all pipelines
    python -m Pipelines.Eval.run_eval -p A B       # score specific pipelines
    python -m Pipelines.Eval.run_eval -p A --detail # single pipeline detail

Usage (programmatic):
    from Pipelines.Eval.evaluator import build_truth, evaluate_pipeline, score_window
    from Pipelines.Eval.compare import score_all_pipelines, build_leaderboard
"""

from .config import FOLDS, Fold, discover_pipelines
from .evaluator import (
    build_truth,
    evaluate_pipeline,
    hourly_to_blocks_truth,
    score_window,
    validate_prediction_df,
    wape,
    rmse,
    mae,
    r2,
)
from .compare import (
    score_all_pipelines,
    build_leaderboard,
    convergence_analysis,
    pairwise_correlation,
)

__all__ = [
    "FOLDS", "Fold", "discover_pipelines",
    "build_truth", "evaluate_pipeline", "hourly_to_blocks_truth",
    "score_window", "validate_prediction_df",
    "wape", "rmse", "mae", "r2",
    "score_all_pipelines", "build_leaderboard",
    "convergence_analysis", "pairwise_correlation",
]
