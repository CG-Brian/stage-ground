"""Research figures (spec §10), matplotlib only.

`matplotlib.use("Agg")` is set at import time so these functions work
headless (CI, tests) without a display backend.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from stageground.evaluation.records import PredictionRecord

ERROR_CATEGORIES = [
    "hallucinated_stage",
    "wrong_stage_with_supporting_evidence",
    "evidence_span_not_found",
    "evidence_does_not_support_prediction",
    "missed_explicit_stage",
    "over_abstention",
    "invalid_normalization",
    "invalid_schema_output",
    "other",
]


def _annotated_scatter(df: pd.DataFrame, *, x: str, y: str, xlabel: str, ylabel: str, title: str, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df[x], df[y])
    for _, row in df.iterrows():
        ax.annotate(str(row["arm"]), (row[x], row[y]), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


def plot_accuracy_vs_unsupported(table_by_arm: pd.DataFrame, outpath: str | Path) -> None:
    """Scatter, one point per arm: x=unsupported_rate, y=accuracy. Visualizes
    how apparently high accuracy can coexist with poor grounding (spec §10 Fig 1)."""
    _annotated_scatter(
        table_by_arm, x="unsupported_rate", y="accuracy",
        xlabel="Unsupported Rate", ylabel="Accuracy",
        title="Accuracy vs Unsupported Rate", outpath=outpath,
    )


def plot_coverage_vs_supported_accuracy(table_by_arm: pd.DataFrame, outpath: str | Path) -> None:
    """Scatter, one point per arm: x=coverage, y=supported_accuracy. Makes the
    abstention tradeoff visible (spec §10 Fig 2)."""
    _annotated_scatter(
        table_by_arm, x="coverage", y="supported_accuracy",
        xlabel="Coverage", ylabel="Supported Accuracy",
        title="Coverage vs Supported Accuracy", outpath=outpath,
    )


def plot_error_breakdown(
    records: list[PredictionRecord], outpath: str | Path, *, target: str | None = None
) -> None:
    """Grouped bar chart: error category on the x-axis, one bar group per arm
    (spec §10 Fig 3). `target=None` uses all records; pass e.g. `target='M'`
    for the M-stage-specific breakdown."""
    subset = records if target is None else [r for r in records if r.target == target]
    arms = sorted({r.arm for r in subset})
    counts = {arm: Counter() for arm in arms}
    for r in subset:
        for err in r.errors:
            counts[r.arm][err] += 1

    x = range(len(ERROR_CATEGORIES))
    width = 0.8 / max(len(arms), 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, arm in enumerate(arms):
        heights = [counts[arm].get(cat, 0) for cat in ERROR_CATEGORIES]
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, heights, width=width, label=arm)

    ax.set_xticks([xi + width * (len(arms) - 1) / 2 for xi in x])
    ax.set_xticklabels(ERROR_CATEGORIES, rotation=45, ha="right")
    ax.set_ylabel("Count")
    title = "Error breakdown by arm" if target is None else f"Error breakdown by arm ({target}-stage)"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
