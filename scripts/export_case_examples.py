"""One-off, no-API-call export: build the interactive sandbox's case data from
the committed main-experiment `predictions.jsonl`. Unlike a flat per-record
export, this groups by (case_id, target) so every exported case carries ALL
FOUR arms' predictions on the SAME report -- the shape the sandbox's
single-arm tabs and compare-all table both need to show how one report is
handled differently by each prompting strategy.

TCGA pathology reports are public, de-identified research data (patient
identifiers are TCGA barcodes, not names/MRNs) -- confirmed with the user
before this script was first written. `_scrub` below is a defensive,
best-effort regex pass over every exported report text regardless, since a
"basic safety scrub" on exported examples is required independent of that
confirmation.

    uv run python scripts/export_case_examples.py

Writes web/src/data/generated/case-examples.json.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from stageground.evaluation.records import PredictionRecord, from_jsonl

ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = ROOT / "results" / "20260908T022659_gpt-4o_seed42_n1000"
OUT_PATH = ROOT / "web" / "src" / "data" / "generated" / "case-examples.json"

ARMS = ["A_zero_shot", "C_constrained", "C_plus_unknown", "D_grounded"]

# The sandbox's default example (spec: gold=M0, prediction=M0, correct=true,
# semantic_support=false on D_grounded) -- picked because the OTHER three
# arms diverge sharply on the exact same report (A fabricates evidence, C/C+
# predict M1 off a genuinely ambiguous "metastases ... spleen" phrase), so a
# single default case demonstrates the whole ablation, not just the M0
# pattern. Found via inspection of the committed predictions, not invented.
DEFAULT_CASE_ID = "TCGA-BR-8371.1836e793-dfac-4fa3-bd1d-a44e2f1692a7"
DEFAULT_TARGET = "M"

_PII_PATTERNS = re.compile(
    r"(MRN\s*[:#]?\s*\S+|DOB\s*[:#]?\s*\S+|Date of Birth\s*[:#]?\s*\S+|"
    r"SSN\s*[:#]?\s*\S+|Social Security\s*\S*|Phone\s*[:#]?\s*\S+|"
    r"\bDr\.\s+[A-Z][a-z]+|Mr\.\s+[A-Z][a-z]+|Mrs\.\s+[A-Z][a-z]+|Ms\.\s+[A-Z][a-z]+)",
    re.IGNORECASE,
)


def _scrub(text: str) -> str:
    return _PII_PATTERNS.sub("[redacted]", text)


def _category_tags(arms: dict[str, PredictionRecord], target: str, gold: str | None) -> list[str]:
    tags: set[str] = set()
    for r in arms.values():
        if r.abstained:
            tags.add("abstained")
            continue
        if r.correct and r.evidence_span_found and r.evidence_semantically_supports_prediction:
            tags.add("correct_grounded")
        if r.correct and r.evidence_span_found and not r.evidence_semantically_supports_prediction:
            tags.add("correct_unsupported")
        if not r.correct and r.evidence_semantically_supports_prediction:
            tags.add("wrong_grounded")
        if r.evidence_span_found and not r.evidence_semantically_supports_prediction:
            tags.add("span_not_supporting")
        if (
            target == "M"
            and gold == "M0"
            and r.prediction == "M0"
            and r.correct
            and not r.evidence_semantically_supports_prediction
        ):
            tags.add("m0_unsupported")
    return sorted(tags)


def _prediction_view(r: PredictionRecord) -> dict:
    return {
        "prediction": r.prediction,
        "evidence": r.evidence,
        "correct": r.correct,
        "abstained": r.abstained,
        "evidenceSpanFound": r.evidence_span_found,
        "semanticSupport": r.evidence_semantically_supports_prediction,
    }


def main() -> None:
    records = from_jsonl(MAIN_DIR / "predictions.jsonl")
    dataset = pd.read_parquet(ROOT / "data" / "processed" / "dataset.parquet")
    text_by_case = dict(zip(dataset["patient_filename"], dataset["text"]))

    grouped: dict[tuple[str, str], dict[str, PredictionRecord]] = defaultdict(dict)
    for r in records:
        grouped[(r.case_id, r.target)][r.arm] = r

    # Only keep complete groups (all 4 arms present) with a locatable report.
    complete = {
        key: arms
        for key, arms in grouped.items()
        if set(arms) == set(ARMS) and key[0] in text_by_case
    }

    tagged = {
        key: _category_tags(arms, key[1], next(iter(arms.values())).ground_truth)
        for key, arms in complete.items()
    }

    quotas = {
        "correct_grounded": 3,
        "correct_unsupported": 3,
        "wrong_grounded": 3,
        "abstained": 2,
        "m0_unsupported": 3,
        "span_not_supporting": 2,
    }
    chosen_keys: list[tuple[str, str]] = [(DEFAULT_CASE_ID, DEFAULT_TARGET)]
    seen_targets_per_category: dict[str, set[str]] = defaultdict(set)

    for category, quota in quotas.items():
        pool = [k for k, tags in tagged.items() if category in tags and k not in chosen_keys]
        pool.sort(key=lambda k: k[1])  # stable order (T, then N, then M) before diversity pass
        picked = []
        for k in pool:
            if len(picked) >= quota:
                break
            if k[1] not in seen_targets_per_category[category] or len(pool) <= quota:
                picked.append(k)
                seen_targets_per_category[category].add(k[1])
        if len(picked) < quota:
            for k in pool:
                if k not in picked and len(picked) < quota:
                    picked.append(k)
        chosen_keys.extend(picked)

    # de-dupe while preserving order (default case, then category picks)
    ordered_unique: list[tuple[str, str]] = []
    for k in chosen_keys:
        if k not in ordered_unique:
            ordered_unique.append(k)

    cases = []
    for i, key in enumerate(ordered_unique):
        case_id, target = key
        arms = complete[key]
        gold = next(iter(arms.values())).ground_truth
        report = _scrub(text_by_case[case_id])
        cases.append({
            "id": f"case-{i:02d}",
            "displayId": f"Case {i + 1:03d}",
            "caseId": case_id,
            "target": target,
            "gold": gold,
            "report": report,
            "categories": tagged[key],
            "arms": {arm: _prediction_view(arms[arm]) for arm in ARMS},
        })

    default_case_id = cases[0]["id"]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"cases": cases, "defaultCaseId": default_case_id}, indent=2))
    print(f"wrote {len(cases)} cases -> {OUT_PATH}")
    for c in cases:
        print(f"  [{c['id']}] {c['target']} gold={c['gold']} categories={c['categories']}")


if __name__ == "__main__":
    main()
