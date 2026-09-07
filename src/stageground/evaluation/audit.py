"""Manual audit infrastructure (spec §6).

Generates a JSONL sheet (fits this project's JSON-heavy `results/cases/`
convention better than CSV, since fields like `automated_errors` are lists)
for a larger blind/manual audit than the v0 pilot's 25-case Track-C audit,
plus a scorer comparing automated judgments to filled-in human labels.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from stageground.evaluation.records import PredictionRecord

MAX_EXCERPT_CHARS = 4000


def build_audit_sheet(
    records: list[PredictionRecord], texts: dict[str, str], *, n: int, seed: int
) -> list[dict]:
    """Deterministic (seeded) sample of `min(n, len(records))` rows. Each row
    carries the automated judgments (`automated_evidence_span_found`,
    `automated_semantic_support`, `automated_errors`) plus blank human_*
    fields for a reviewer to fill in and re-save."""
    k = min(n, len(records))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(records)), k))

    rows = []
    for i in indices:
        rec = records[i]
        text = texts.get(rec.case_id, "")
        excerpt = (
            text if len(text) <= MAX_EXCERPT_CHARS
            else text[:MAX_EXCERPT_CHARS] + "\n\n[... truncated for audit sheet length ...]"
        )
        rows.append({
            "case_id": rec.case_id,
            "arm": rec.arm,
            "target": rec.target,
            "report_excerpt": excerpt,
            "ground_truth": rec.ground_truth,
            "prediction": rec.prediction,
            "evidence": rec.evidence,
            "automated_evidence_span_found": rec.evidence_span_found,
            "automated_semantic_support": rec.evidence_semantically_supports_prediction,
            "automated_errors": list(rec.errors),
            "human_supported": None,
            "human_evidence_correct": None,
            "human_prediction_correct": None,
            "human_error_type": "",
            "reviewer_notes": "",
        })
    return rows


def write_audit_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def read_audit_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _paired_bool_labels(rows: list[dict], *, automated_key, human_key: str) -> tuple[list[bool], list[bool]]:
    autos, humans = [], []
    for row in rows:
        human_val = row.get(human_key)
        if human_val is None:
            continue
        auto_val = automated_key(row)
        if auto_val is None:
            continue
        autos.append(bool(auto_val))
        humans.append(bool(human_val))
    return autos, humans


def _score_pair(autos: list[bool], humans: list[bool], *, label: str) -> dict:
    n_scored = len(autos)
    if n_scored == 0:
        return {
            f"n_scored_{label}": 0,
            f"percent_agreement_{label}": float("nan"),
            f"cohens_kappa_{label}": None,
        }

    agree = sum(1 for a, h in zip(autos, humans) if a == h)
    percent_agreement = agree / n_scored

    kappa = None
    if n_scored >= 2 and len(set(autos) | set(humans)) >= 2:
        from sklearn.metrics import cohen_kappa_score

        value = float(cohen_kappa_score(autos, humans))
        kappa = None if value != value else value  # NaN (zero-variance) -> None

    return {
        f"n_scored_{label}": n_scored,
        f"percent_agreement_{label}": percent_agreement,
        f"cohens_kappa_{label}": kappa,
    }


def score_audit(rows: list[dict]) -> dict:
    """Compares automated judgments to filled-in human labels wherever a
    human field is populated (not None), reporting percent agreement + Cohen's
    kappa for two judgment types: 'supported' and 'prediction_correct'
    (derived automatically as `prediction == ground_truth`). Rows with an
    unfilled human field for a given comparison are simply excluded from that
    comparison's `n_scored`, never causing a crash.

    NOTE: the 'supported' comparison pairs `human_supported` against
    `automated_evidence_span_found` (span-only), not the semantic-support
    field -- `human_supported`'s own name is ambiguous about which notion of
    grounding the reviewer judged and is a candidate for a future rename to
    e.g. `human_evidence_span_found` / `human_semantic_support`, out of
    scope for this pass."""
    supported_autos, supported_humans = _paired_bool_labels(
        rows, automated_key=lambda r: r.get("automated_evidence_span_found"), human_key="human_supported"
    )
    correct_autos, correct_humans = _paired_bool_labels(
        rows,
        automated_key=lambda r: (
            r.get("prediction") == r.get("ground_truth") if r.get("ground_truth") is not None else None
        ),
        human_key="human_prediction_correct",
    )

    result = {}
    result.update(_score_pair(supported_autos, supported_humans, label="supported"))
    result.update(_score_pair(correct_autos, correct_humans, label="prediction_correct"))
    return result
