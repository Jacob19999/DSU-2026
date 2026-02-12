from __future__ import annotations

"""
One-shot orchestrator to run Data Source + all model pipelines + central Eval.

Typical usage (from repo root):

    # Fast(ish) full stack: data source → A–E (skip tuning) → Eval
    python Pipelines/run_all_pipelines.py

    # Skip data source (e.g. master_block_history already built)
    python Pipelines/run_all_pipelines.py --skip-data-source

    # Run only a subset of pipelines
    python Pipelines/run_all_pipelines.py --pipelines A C E

    # Allow each pipeline to run its full tuning (can be very slow)
    python Pipelines/run_all_pipelines.py --full-tune

    # Ask CV pipelines (B, D, E) to also generate final submissions
    python Pipelines/run_all_pipelines.py --submit

    # Full "final result": full tuning + final forecasts for all pipelines
    python Pipelines/run_all_pipelines.py --final
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run_cmd(label: str, args: list[str]) -> None:
    """
    Run a single step as a subprocess, printing a simple timing banner.
    """
    sep = "=" * 80
    print(f"\n{sep}")
    print(f"[{label}] Starting")
    print(sep)
    t0 = time.time()

    try:
        subprocess.run([PYTHON, *args], cwd=REPO_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        print(f"[{label}] FAILED after {elapsed / 60:.1f} min "
              f"(exit code {exc.returncode})")
        # Re-raise so the orchestrator exits non‑zero for shell scripts
        raise
    else:
        elapsed = time.time() - t0
        print(f"[{label}] DONE in {elapsed / 60:.1f} min")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Data Source ingestion, all model pipelines (A–E) and central "
            "Eval in one go."
        ),
    )
    parser.add_argument(
        "--skip-data-source",
        action="store_true",
        help="Skip Data Source ingestion step.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip centralized Eval step.",
    )
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=list("ABCDE"),
        default=list("ABCDE"),
        help="Subset of pipelines to run (default: A B C D E).",
    )
    parser.add_argument(
        "--full-tune",
        action="store_true",
        help=(
            "Let each pipeline use its default tuning behavior. "
            "By default this orchestrator adds --skip-tune where supported "
            "for faster iteration."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Ask CV pipelines (B, D, E) to also generate Sept–Oct 2025 "
            "submission files (mode=submit)."
        ),
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help=(
            "Convenience flag: run with full tuning and generate final Sept–Oct "
            "2025 submissions for all pipelines. Equivalent to --full-tune "
            "+ --submit, plus --final-forecast for pipelines A and C."
        ),
    )
    parser.add_argument(
        "--use-reason-embeddings",
        action="store_true",
        help=(
            "Pass --use-reason-embeddings to the Data Source step, enabling "
            "block-level embedding features derived from REASON_VISIT_NAME."
        ),
    )
    args = parser.parse_args()

    # Interpret "final" as sugar over lower-level knobs.
    full_tune = args.full_tune or args.final
    submit = args.submit or args.final

    # 1) Data Source
    if not args.skip_data_source:
        ds_extra: list[str] = []
        if args.use_reason_embeddings:
            ds_extra.append("--use-reason-embeddings")
        _run_cmd(
            "Data Source",
            ["-m", "Pipelines.data_source.run_data_source", *ds_extra],
        )

    # 2) Pipelines
    # Pipeline A
    if "A" in args.pipelines:
        extra: list[str] = [] if full_tune else ["--skip-tune"]
        if submit:
            extra.append("--final-forecast")
        _run_cmd(
            "Pipeline A",
            ["Pipelines/Pipeline A/run_pipeline.py", *extra],
        )

    # Pipeline B
    if "B" in args.pipelines:
        extra = [] if full_tune else ["--skip-tune"]
        mode = "submit" if submit else "cv"
        _run_cmd(
            "Pipeline B",
            ["Pipelines/Pipeline B/run_pipeline.py", "--mode", mode, *extra],
        )

    # Pipeline C
    if "C" in args.pipelines:
        extra = [] if full_tune else ["--skip-tune"]
        if submit:
            extra.append("--final-forecast")
        _run_cmd(
            "Pipeline C",
            ["Pipelines/Pipeline C/run_pipeline.py", *extra],
        )

    # Pipeline D
    if "D" in args.pipelines:
        extra = [] if full_tune else ["--skip-tune"]
        mode = "submit" if submit else "cv"
        _run_cmd(
            "Pipeline D",
            ["Pipelines/Pipeline D/run_pipeline.py", "--mode", mode, *extra],
        )

    # Pipeline E
    if "E" in args.pipelines:
        extra = [] if full_tune else ["--skip-tune"]
        mode = "submit" if submit else "cv"
        _run_cmd(
            "Pipeline E",
            ["Pipelines/Pipeline E/run_pipeline.py", "--mode", mode, *extra],
        )

    # 3) Centralized Eval over whatever pipelines produced preds
    if not args.skip_eval:
        _run_cmd(
            "Central Eval",
            ["-m", "Pipelines.Eval.run_eval"],
        )


if __name__ == "__main__":
    main()

