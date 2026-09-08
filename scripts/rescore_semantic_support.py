"""One-off: recompute `evidence_semantically_supports_prediction` on an
existing `predictions.jsonl` using the CURRENT semantic-support heuristic
(`stageground.evaluation.semantic_support`), without calling the LLM API,
and print/save a before-vs-after comparison of the derived metrics.

Model outputs (prediction, evidence, correct, abstained, evidence_span_found,
raw_model_output) are never touched -- only the semantic-support-derived
field (and metrics computed from it) are recomputed.

Usage:
    uv run python scripts/rescore_semantic_support.py \
        --predictions results/<experiment_id>/predictions.jsonl \
        --output results/<experiment_id>/semantic_support_rescore.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from stageground.evaluation.metrics import compute_all_metrics
from stageground.evaluation.records import PredictionRecord, from_jsonl
from stageground.evaluation.semantic_support import evidence_semantically_supports_prediction
from stageground.evaluation.tables import build_comparison_table

COLUMNS = [
    "accuracy", "semantic_supported_accuracy", "span_unsupported_rate",
    "semantic_unsupported_rate", "abstention_rate", "coverage",
]
SCOPES = [("overall", None), ("T", "T"), ("N", "N"), ("M", "M")]


def rescore(records: list[PredictionRecord]) -> list[PredictionRecord]:
    """Returns a NEW list of records with `evidence_semantically_supports_prediction`
    recomputed via the current heuristic. Abstained records are untouched
    (both grounding fields are already `None` and stay that way)."""
    updated = []
    for rec in records:
        if rec.abstained:
            updated.append(rec)
            continue
        new_semantic = (
            evidence_semantically_supports_prediction(rec.prediction, rec.evidence, rec.target)
            if rec.evidence_span_found else False
        )
        updated.append(dataclasses.replace(rec, evidence_semantically_supports_prediction=new_semantic))
    return updated


def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.4f}"  # x != x iff NaN


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def _comparison_table_text(before: list[PredictionRecord], after: list[PredictionRecord]) -> str:
    lines = ["# Semantic-support heuristic: before vs after re-score\n"]
    for label, target in SCOPES:
        before_df = build_comparison_table(before, target=target).set_index("arm")
        after_df = build_comparison_table(after, target=target).set_index("arm")
        lines.append(f"\n## {label}\n")
        for arm in before_df.index:
            rows = []
            for col in COLUMNS:
                b, a = before_df.loc[arm, col], after_df.loc[arm, col]
                delta = a - b
                rows.append({"metric": col, "before": _fmt(b), "after": _fmt(a), "delta": _fmt(delta)})
            lines.append(f"\n### {arm}\n")
            lines.append(_markdown_table(rows, ["metric", "before", "after", "delta"]))
    return "\n".join(lines)


def write_rescored_metrics_json(records: list[PredictionRecord], outdir: Path) -> None:
    """Writes metrics_rescored.json / metrics_by_target_rescored.json in the
    SAME shape as stageground.evaluate.run_evaluation's metrics.json /
    metrics_by_target.json, computed from the RESCORED records -- so a
    pilot-vs-main comparison against a later experiment (which always uses
    whatever heuristic is current) compares like with like, instead of
    conflating a heuristic change with a sample-size effect."""
    arms = sorted({r.arm for r in records})
    metrics_json = {
        arm: compute_all_metrics([r for r in records if r.arm == arm], target=None) for arm in arms
    }
    metrics_by_target_json = {
        arm: {
            target: compute_all_metrics([r for r in records if r.arm == arm], target=target)
            for target in ("T", "N", "M")
        }
        for arm in arms
    }
    (outdir / "metrics_rescored.json").write_text(json.dumps(metrics_json, indent=2))
    (outdir / "metrics_by_target_rescored.json").write_text(json.dumps(metrics_by_target_json, indent=2))
    print(f"wrote {outdir / 'metrics_rescored.json'} and metrics_by_target_rescored.json")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", default=None, help="optional path to write the comparison as Markdown")
    ap.add_argument("--emit-metrics-json", action="store_true",
                     help="also write metrics_rescored.json/metrics_by_target_rescored.json "
                          "next to --predictions, for a fair pilot-vs-main comparison")
    args = ap.parse_args(argv)

    before = from_jsonl(args.predictions)
    after = rescore(before)

    report = _comparison_table_text(before, after)
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nwrote {args.output}")

    if args.emit_metrics_json:
        write_rescored_metrics_json(after, Path(args.predictions).parent)


if __name__ == "__main__":
    main()
