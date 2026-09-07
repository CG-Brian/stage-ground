"""Standardized per-prediction record (spec §8).

One `PredictionRecord` per (case, arm, target) triple. This is the atomic
unit every downstream module (metrics, error taxonomy, tables, plots,
bootstrap, audit) consumes -- keeping one vocabulary avoids the "scattered
scoring logic" problem the v0 pilot's `scripts/03_score.py` had (accuracy,
grounding, and Track-C tagging were computed inline, separately, per script).

Raw model output is never discarded (spec §8): `raw_model_output` always
carries the full, un-normalized dict the arm produced for that target.

Grounding is deliberately split into two fields, not one ambiguous flag:
  - `evidence_span_found`: does the evidence string literally (verbatim,
    OCR-noise-tolerant) appear in the report? A syntactic fact.
  - `evidence_semantically_supports_prediction`: does that evidence actually
    say what was predicted? An automated *heuristic* proxy (regex stage-token
    match), NOT equivalent to human semantic judgment -- see
    `stageground.evaluation.metrics` module docstring and the manual audit
    tooling in `stageground.evaluation.audit`.
Fabricated (ungrounded) evidence cannot semantically support anything, so
`evidence_semantically_supports_prediction` is forced `False` whenever
`evidence_span_found` is `False` -- it is never computed independently of
span-grounding. Both are `None` for abstained predictions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from stageground.evaluation.error_taxonomy import classify_errors
from stageground.evaluation.normalize import grounded, norm_pred, value_supported_by_text


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
    evidence_span_found: bool | None  # None iff abstained; else: is `evidence` verbatim in the report?
    evidence_semantically_supports_prediction: bool | None  # None iff abstained; heuristic, see module docstring
    errors: list[str] = field(default_factory=list)
    raw_model_output: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PredictionRecord":
        """Loads current-schema JSONL directly. Also loads OLD JSONL written
        before the span/semantic split: a legacy `supported` key (with no
        `evidence_span_found` key present) is mapped onto `evidence_span_found`
        -- `supported` was, in practice, always span-grounding only --
        and `evidence_semantically_supports_prediction` defaults to `None`
        ("not computed by the old schema"), never `False` ("computed and
        found unsupported"). Never mutates the caller's dict. Unknown extra
        keys are ignored for forward-compat."""
        d = dict(d)
        if "evidence_span_found" not in d and "supported" in d:
            d["evidence_span_found"] = d.pop("supported")
            d.setdefault("evidence_semantically_supports_prediction", None)
        d.pop("supported", None)  # drop a stray legacy key if both happen to be present

        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


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

    if abstained:
        evidence_span_found = None
        evidence_semantically_supports_prediction = None
    else:
        evidence_span_found = bool(evidence is not None and grounded(evidence, report_text))
        evidence_semantically_supports_prediction = (
            bool(value_supported_by_text(prediction, evidence)) if evidence_span_found else False
        )

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
        evidence_span_found=evidence_span_found,
        evidence_semantically_supports_prediction=evidence_semantically_supports_prediction,
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
