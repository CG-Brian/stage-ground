"""Analyses specific to interpreting one ablation experiment's central
questions -- M0 prediction deep-dive, gold/predicted label distributions,
and coverage-vs-reliability -- as opposed to the generic per-arm/per-target
metrics already covered by `metrics.py`/`tables.py`.

None of these functions label a pattern "guessing" or "prior-driven"
automatically; they only compute the raw ingredients (accuracy, coverage,
proportion, semantic/span support rates) a reader needs to draw that
conclusion themselves, conservatively.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

from stageground.evaluation.metrics import compute_all_metrics
from stageground.evaluation.records import PredictionRecord

TARGETS = ("T", "N", "M")


def m0_prediction_analysis(records: list[PredictionRecord]) -> pd.DataFrame:
    """Per arm, among M-target predictions of `"M0"`: count, accuracy (over
    the subset with ground truth available), proportion of all asserted
    (non-abstained) M predictions, semantic-supported rate, span-grounded
    rate. This is the raw material for "does high M0 accuracy coexist with
    little source support" -- it does not itself conclude anything."""
    m_records = [r for r in records if r.target == "M"]
    arms = sorted({r.arm for r in m_records})
    rows = []
    for arm in arms:
        arm_m = [r for r in m_records if r.arm == arm]
        asserted = [r for r in arm_m if not r.abstained]
        m0_preds = [r for r in asserted if r.prediction == "M0"]
        m0_evaluable = [r for r in m0_preds if r.ground_truth is not None]

        n_m0 = len(m0_preds)
        accuracy = (sum(1 for r in m0_evaluable if r.correct) / len(m0_evaluable)) if m0_evaluable else float("nan")
        proportion_of_asserted = (n_m0 / len(asserted)) if asserted else float("nan")
        semantic_rate = (
            sum(1 for r in m0_preds if r.evidence_semantically_supports_prediction) / n_m0
        ) if n_m0 else float("nan")
        span_rate = (sum(1 for r in m0_preds if r.evidence_span_found) / n_m0) if n_m0 else float("nan")

        rows.append({
            "arm": arm,
            "n_m0_predictions": n_m0,
            "n_m0_evaluable": len(m0_evaluable),
            "m0_accuracy": accuracy,
            "m0_proportion_of_asserted_m": proportion_of_asserted,
            "m0_semantic_supported_rate": semantic_rate,
            "m0_span_grounded_rate": span_rate,
        })
    return pd.DataFrame(rows, columns=[
        "arm", "n_m0_predictions", "n_m0_evaluable", "m0_accuracy",
        "m0_proportion_of_asserted_m", "m0_semantic_supported_rate", "m0_span_grounded_rate",
    ])


def gold_label_distribution(records: list[PredictionRecord]) -> pd.DataFrame:
    """Gold label frequency per target, deduplicated by case_id -- gold
    doesn't vary by arm, so counting across all arms without deduping would
    just multiply every count by the number of arms."""
    rows = []
    for target in TARGETS:
        seen: dict[str, str | None] = {}
        for r in records:
            if r.target == target and r.case_id not in seen:
                seen[r.case_id] = r.ground_truth
        counts = Counter(v for v in seen.values() if v is not None)
        total = sum(counts.values())
        for value, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            rows.append({
                "target": target, "value": value, "count": count,
                "fraction": (count / total) if total else float("nan"),
            })
    return pd.DataFrame(rows, columns=["target", "value", "count", "fraction"])


def predicted_label_distribution(records: list[PredictionRecord]) -> pd.DataFrame:
    """Per-arm, per-target predicted-value frequency (prediction DOES vary by
    arm, so no deduplication here, unlike `gold_label_distribution`)."""
    rows = []
    for target in TARGETS:
        target_records = [r for r in records if r.target == target]
        arms = sorted({r.arm for r in target_records})
        for arm in arms:
            arm_records = [r for r in target_records if r.arm == arm]
            counts = Counter(r.prediction for r in arm_records)
            total = sum(counts.values())
            for value, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
                rows.append({
                    "target": target, "arm": arm, "value": value, "count": count,
                    "fraction": (count / total) if total else float("nan"),
                })
    return pd.DataFrame(rows, columns=["target", "arm", "value", "count", "fraction"])


def coverage_vs_accuracy_table(records: list[PredictionRecord], *, target: str | None = None) -> pd.DataFrame:
    """Per arm: coverage, accuracy_over_asserted, semantic_supported_accuracy_over_asserted.
    Answers whether improved reliability is caused by an arm simply
    abstaining more, or by it being more reliable when it DOES assert."""
    subset = records if target is None else [r for r in records if r.target == target]
    arms = sorted({r.arm for r in subset})
    rows = []
    for arm in arms:
        m = compute_all_metrics([r for r in subset if r.arm == arm], target=None)
        rows.append({
            "arm": arm,
            "coverage": m["coverage"],
            "accuracy_over_asserted": m["accuracy_over_asserted"],
            "semantic_supported_accuracy_over_asserted": m["semantic_supported_accuracy_over_asserted"],
        })
    return pd.DataFrame(rows, columns=[
        "arm", "coverage", "accuracy_over_asserted", "semantic_supported_accuracy_over_asserted",
    ])
