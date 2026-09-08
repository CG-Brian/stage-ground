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

from stageground.evaluation.metrics import compute_all_metrics
from stageground.evaluation.records import PredictionRecord

TARGETS = ("T", "N", "M")

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


def plot_accuracy_vs_span_unsupported(table_by_arm: pd.DataFrame, outpath: str | Path) -> None:
    """Scatter, one point per arm: x=span_unsupported_rate, y=accuracy. A
    secondary/diagnostic figure -- prefer `plot_accuracy_vs_semantic_unsupported`
    as the main grounding figure (spec §13)."""
    _annotated_scatter(
        table_by_arm, x="span_unsupported_rate", y="accuracy",
        xlabel="Span Unsupported Rate", ylabel="Accuracy",
        title="Accuracy vs Span Unsupported Rate", outpath=outpath,
    )


def plot_accuracy_vs_semantic_unsupported(table_by_arm: pd.DataFrame, outpath: str | Path) -> None:
    """Scatter, one point per arm: x=semantic_unsupported_rate, y=accuracy.
    The MAIN grounding figure per spec §13: visualizes how apparently high
    accuracy can coexist with poor (semantic) grounding. semantic_unsupported_rate
    is still an automated heuristic -- see `metrics.py` module docstring."""
    _annotated_scatter(
        table_by_arm, x="semantic_unsupported_rate", y="accuracy",
        xlabel="Semantic Unsupported Rate (heuristic)", ylabel="Accuracy",
        title="Accuracy vs Semantic Unsupported Rate (heuristic)", outpath=outpath,
    )


def plot_coverage_vs_semantic_supported_accuracy(table_by_arm: pd.DataFrame, outpath: str | Path) -> None:
    """Scatter, one point per arm: x=coverage, y=semantic_supported_accuracy.
    Makes the abstention tradeoff visible (spec §10 Fig 2)."""
    _annotated_scatter(
        table_by_arm, x="coverage", y="semantic_supported_accuracy",
        xlabel="Coverage", ylabel="Semantic Supported Accuracy (heuristic)",
        title="Coverage vs Semantic Supported Accuracy (heuristic)", outpath=outpath,
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


def plot_semantic_unsupported_by_target(records: list[PredictionRecord], outpath: str | Path) -> None:
    """Grouped bar chart: T/N/M on the x-axis, one bar per arm, height =
    semantic_unsupported_rate_over_evaluable for that (arm, target) pair.
    Arms are sorted (which, for this project's arm names, already yields
    A_zero_shot / C_constrained / C_plus_unknown / D_grounded)."""
    arms = sorted({r.arm for r in records})
    rates = {
        arm: [
            compute_all_metrics([r for r in records if r.arm == arm], target=t)
            ["semantic_unsupported_rate_over_evaluable"]
            for t in TARGETS
        ]
        for arm in arms
    }

    x = range(len(TARGETS))
    width = 0.8 / max(len(arms), 1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, arm in enumerate(arms):
        offsets = [xi + i * width for xi in x]
        ax.bar(offsets, rates[arm], width=width, label=arm)

    ax.set_xticks([xi + width * (len(arms) - 1) / 2 for xi in x])
    ax.set_xticklabels(TARGETS)
    ax.set_ylabel("Semantic Unsupported Rate (heuristic)")
    ax.set_title("Semantic Unsupported Rate by Target and Arm")
    ax.legend()
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)


def plot_m_accuracy_vs_semantic_supported(records: list[PredictionRecord], outpath: str | Path) -> None:
    """Grouped bar chart, M-target only: one arm per x position, two bars each
    (raw accuracy, semantic_supported_accuracy_over_evaluable) -- makes the
    accuracy/source-support gap directly visible per arm, rather than a
    scatter's floating points."""
    m_records = [r for r in records if r.target == "M"]
    arms = sorted({r.arm for r in m_records})
    accuracy_vals, semantic_vals = [], []
    for arm in arms:
        m = compute_all_metrics([r for r in m_records if r.arm == arm], target="M")
        accuracy_vals.append(m["accuracy"])
        semantic_vals.append(m["semantic_supported_accuracy_over_evaluable"])

    x = range(len(arms))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([xi - width / 2 for xi in x], accuracy_vals, width=width, label="Accuracy")
    ax.bar([xi + width / 2 for xi in x], semantic_vals, width=width, label="Semantic Supported Accuracy (heuristic)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(arms, rotation=20, ha="right")
    ax.set_ylabel("Rate")
    ax.set_title("M-stage: Accuracy vs Semantic Supported Accuracy")
    ax.legend()
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    plt.close(fig)
