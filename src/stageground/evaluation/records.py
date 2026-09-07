"""Standardized per-prediction record (spec §8).

One `PredictionRecord` per (case, arm, target) triple. This is the atomic
unit every downstream module (metrics, error taxonomy, tables, plots,
bootstrap, audit) consumes -- keeping one vocabulary avoids the "scattered
scoring logic" problem the v0 pilot's `scripts/03_score.py` had (accuracy,
grounding, and Track-C tagging were computed inline, separately, per script).

Raw model output is never discarded (spec §8): `raw_model_output` always
carries the full, un-normalized dict the arm produced for that target.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from stageground.evaluation.error_taxonomy import classify_errors
from stageground.evaluation.normalize import grounded, norm_pred


@dataclass
class PredictionRecord:
    case_id: str
    arm: str
    target: str  # "T" | "N" | "M"
    ground_truth: str | None  # canonicalized gold, or None if unavailable
    prediction: str  # canonicalized value, "unknown", or "INVALID"
    evidence: str | None
    correct: bool | None  # None iff ground_truth is None (not evaluable)
    abstained: bool
    supported: bool | None  # None iff abstained (supported/unsupported only applies to assertions)
    errors: list[str] = field(default_factory=list)
    raw_model_output: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        return cls(**d)


def build_record(
    *,
    case_id: str,
    arm: str,
    target: str,
    ground_truth: str | None,
    raw_output: dict | None,
    report_text: str,
    schema_valid: bool = True,
) -> PredictionRecord:
    """Build a PredictionRecord from one arm's raw output for one target field.

    `raw_output` is the field-level dict (`{"value", "evidence", "confidence",
    "reason"}`) for this target, or None/ignored when `schema_valid` is False
    (the model's JSON failed shape validation entirely, so no field values can
    be trusted). `ground_truth` must already be canonicalized (or None).
    """
    if not schema_valid:
        prediction = "INVALID"
        evidence = None
        abstained = False
        raw_value = None
    else:
        raw_output = raw_output or {}
        raw_value = raw_output.get("value")
        evidence = raw_output.get("evidence")
        prediction = norm_pred(raw_value)
        abstained = prediction == "unknown"

    correct = None if ground_truth is None else (prediction == ground_truth)
    supported = None if abstained else bool(evidence is not None and grounded(evidence, report_text))

    errors = classify_errors(
        ground_truth=ground_truth,
        prediction=prediction,
        evidence=evidence,
        report_text=report_text,
        raw_value=raw_value,
        schema_valid=schema_valid,
    )

    return PredictionRecord(
        case_id=case_id,
        arm=arm,
        target=target,
        ground_truth=ground_truth,
        prediction=prediction,
        evidence=evidence,
        correct=correct,
        abstained=abstained,
        supported=supported,
        errors=errors,
        raw_model_output=raw_output if schema_valid else {},
    )


def to_jsonl(records: list[PredictionRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False))
            f.write("\n")


def from_jsonl(path: str | Path) -> list[PredictionRecord]:
    path = Path(path)
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(PredictionRecord.from_dict(json.loads(line)))
    return records
