"""One-off, no-API-call export: curate a small, diverse set of example
records from the committed main-experiment `predictions.jsonl` for the
frontend's case explorer, with a windowed report excerpt (reusing the same
`_report_excerpt` helper the blinded audit sheets use, so excerpt selection
logic isn't duplicated). TCGA pathology reports are public, de-identified
research data (patient identifiers are TCGA barcodes, not names/MRNs), so
showing excerpts is safe; this script does not add or remove any redaction
beyond what the source dataset already has.

    uv run python scripts/export_case_examples.py

Writes web/src/data/generated/case-examples.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from stageground.evaluation.audit import _report_excerpt
from stageground.evaluation.records import from_jsonl

ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = ROOT / "results" / "20260908T022659_gpt-4o_seed42_n1000"
OUT_PATH = ROOT / "web" / "src" / "data" / "generated" / "case-examples.json"

RNG = random.Random(42)


def category_of(r) -> str | None:
    if r.abstained:
        return "abstained"
    if r.target == "M" and r.prediction == "M0" and r.correct and not r.evidence_semantically_supports_prediction:
        return "m0_unsupported"
    if r.correct and r.evidence_span_found and r.evidence_semantically_supports_prediction:
        return "correct_grounded"
    if r.correct and r.evidence_span_found and not r.evidence_semantically_supports_prediction:
        return "correct_unsupported"
    if not r.correct and r.evidence_span_found and r.evidence_semantically_supports_prediction:
        return "wrong_grounded"
    return None


def main() -> None:
    records = from_jsonl(MAIN_DIR / "predictions.jsonl")
    dataset = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet")
    text_by_case = dict(zip(dataset["patient_filename"], dataset["text"]))

    # Prefer D_grounded/C_constrained for narrative relevance (they're the
    # arms the landing page spends the most time on); fall back to any arm
    # if a category has no examples there.
    preferred_arms = ["D_grounded", "C_constrained", "C_plus_unknown", "A_zero_shot"]
    by_category: dict[str, list] = {
        "correct_grounded": [], "correct_unsupported": [], "wrong_grounded": [],
        "abstained": [], "m0_unsupported": [],
    }
    for r in records:
        cat = category_of(r)
        if cat is not None and r.case_id in text_by_case:
            by_category[cat].append(r)

    n_per_category = {
        "correct_grounded": 3, "correct_unsupported": 3, "wrong_grounded": 2,
        "abstained": 2, "m0_unsupported": 3,
    }

    chosen = []
    for cat, n in n_per_category.items():
        pool = by_category[cat]
        pool.sort(key=lambda r: preferred_arms.index(r.arm) if r.arm in preferred_arms else 99)
        # keep target diversity: don't pick 3 of the same target if avoidable
        picked, seen_targets = [], set()
        for r in pool:
            if len(picked) >= n:
                break
            if r.target not in seen_targets or len(pool) <= n:
                picked.append(r)
                seen_targets.add(r.target)
        if len(picked) < n:
            for r in pool:
                if r not in picked and len(picked) < n:
                    picked.append(r)
        for r in picked:
            chosen.append((cat, r))

    cases = []
    for i, (cat, r) in enumerate(chosen):
        text = text_by_case[r.case_id]
        excerpt, truncated = _report_excerpt(text, r.evidence)
        cases.append({
            "id": f"case-{i:02d}",
            "category": cat,
            "caseId": r.case_id,
            "arm": r.arm,
            "target": r.target,
            "groundTruth": r.ground_truth,
            "prediction": r.prediction,
            "evidence": r.evidence,
            "reportExcerpt": excerpt,
            "excerptTruncated": truncated,
            "evidenceSpanFound": r.evidence_span_found,
            "semanticSupport": r.evidence_semantically_supports_prediction,
            "correct": r.correct,
            "abstained": r.abstained,
            "errors": r.errors,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"cases": cases}, indent=2))
    print(f"wrote {len(cases)} cases -> {OUT_PATH}")
    for c in cases:
        print(f"  [{c['category']}] {c['arm']} / {c['target']}  gold={c['groundTruth']} pred={c['prediction']}")


if __name__ == "__main__":
    main()
