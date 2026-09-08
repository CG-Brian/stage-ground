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
from pathlib import Path

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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", default=None, help="optional path to write the comparison as Markdown")
    args = ap.parse_args(argv)

    before = from_jsonl(args.predictions)
    after = rescore(before)

    report = _comparison_table_text(before, after)
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
