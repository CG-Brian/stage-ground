"""One-off: recompute `automated_semantic_support` in a completed audit
bundle's `key.jsonl` using the CURRENT semantic-support heuristic
(`stageground.evaluation.semantic_support`), without calling the LLM API or
touching any human-filled `reviewer.jsonl` field, and report a before/after
comparison of human-vs-heuristic agreement (percent agreement, Cohen's
kappa, precision, recall, specificity) overall and per target.

`evidence` isn't stored in key.jsonl (it's reviewer-facing only, per the
blinding design), so this joins key.jsonl + reviewer.jsonl on audit_id to
recompute -- the reviewer's OWN human_* judgments are never read or altered.

Usage:
    uv run python scripts/rescore_audit_semantic_support.py \
        --audit-dir audit/audit_001 \
        --output audit/audit_001/semantic_support_rescore.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stageground.evaluation.audit import read_audit_jsonl
from stageground.evaluation.audit_scoring import score_reviewed_audit
from stageground.evaluation.semantic_support import evidence_semantically_supports_prediction

SCOPES = ["overall", "T", "N", "M"]
ROW_FIELDS = ["n", "percent_agreement", "cohens_kappa", "precision", "recall", "specificity"]


def rescore_key_rows(key_rows: list[dict], reviewer_rows: list[dict]) -> list[dict]:
    """Returns a NEW list of key rows with `automated_semantic_support`
    recomputed. Abstained rows (automated_evidence_span_found is None) and
    span-not-found rows are left exactly as before (recomputing wouldn't
    change them: no evidence to re-evaluate, or the gating rule already
    forces False)."""
    evidence_by_id = {r["audit_id"]: r.get("evidence") for r in reviewer_rows}
    updated = []
    for row in key_rows:
        row = dict(row)
        span_found = row.get("automated_evidence_span_found")
        if span_found is None:
            pass  # abstained -- leave automated_semantic_support as None
        elif span_found is False:
            row["automated_semantic_support"] = False
        else:
            evidence = evidence_by_id.get(row["audit_id"])
            row["automated_semantic_support"] = evidence_semantically_supports_prediction(
                row["prediction"], evidence, row["target"]
            )
        updated.append(row)
    return updated


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float) and x != x:  # NaN
        return "n/a"
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def _comparison_report(before: dict, after: dict) -> str:
    lines = ["# Audit re-score: semantic_support agreement, before vs after\n"]
    for scope in SCOPES:
        lines.append(f"\n## {scope}\n")
        rows = []
        for field in ROW_FIELDS:
            b = before[scope]["semantic_support"][field]
            a = after[scope]["semantic_support"][field]
            delta = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            rows.append({"metric": field, "before": _fmt(b), "after": _fmt(a), "delta": _fmt(delta)})
        lines.append(_markdown_table(rows, ["metric", "before", "after", "delta"]))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-dir", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    audit_dir = Path(args.audit_dir)
    reviewer_rows = read_audit_jsonl(audit_dir / "reviewer.jsonl")
    key_rows_before = read_audit_jsonl(audit_dir / "key.jsonl")
    key_rows_after = rescore_key_rows(key_rows_before, reviewer_rows)

    before = score_reviewed_audit(reviewer_rows, key_rows_before)
    after = score_reviewed_audit(reviewer_rows, key_rows_after)

    report = _comparison_report(before, after)
    print(report)

    if args.output:
        Path(args.output).write_text(report)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
