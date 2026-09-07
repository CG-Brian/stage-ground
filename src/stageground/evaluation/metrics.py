"""Explicit metric definitions (spec §3).

Every function takes a `list[PredictionRecord]` -- already filtered to the
arm/target of interest by the caller, or passed as-is to aggregate across
targets. All metrics are scoped to **evaluable cases** (ground truth
available for that target) so every number in a comparison table shares the
same population and denominators are never ambiguous. A denominator of zero
never raises: it returns `float("nan")`, so callers (tables/JSON output) can
render "n/a" instead of crashing on edge cases like "all predictions
unknown" or "no ground truth in this sample" (spec §12).

Two related but distinct notions of "no textual basis" are deliberately kept
separate, matching the v0 pilot's validated (92%-agreement-audited)
groundedness check rather than silently redefining it:

  - `supported` (on PredictionRecord, set at build time) = the evidence span
    is a verbatim, OCR-noise-tolerant substring of the report. This is what
    `unsupported_rate_*` and `supported_accuracy_*` are built on.
  - `evidence_semantic_support_rate` (this module) is a second, independent,
    weaker heuristic: does the evidence string additionally contain a stage
    token that canonicalizes to the predicted value? It is reported
    separately and never folds into `supported`, per spec §3's instruction
    not to silently collapse span-found and semantically-supports into one
    score.
"""

from __future__ import annotations

from stageground.evaluation.normalize import DOMAIN, value_supported_by_text
from stageground.evaluation.records import PredictionRecord

NAN = float("nan")


def _evaluable(records: list[PredictionRecord]) -> list[PredictionRecord]:
    return [r for r in records if r.ground_truth is not None]


def _asserted(records: list[PredictionRecord]) -> list[PredictionRecord]:
    """Evaluable AND non-abstained -- the 'non-abstained predictions' /
    'asserted' population referenced by the *_over_asserted metric variants."""
    return [r for r in _evaluable(records) if not r.abstained]


def accuracy(records: list[PredictionRecord]) -> float:
    """correct predictions / evaluable ground-truth cases"""
    ev = _evaluable(records)
    if not ev:
        return NAN
    return sum(1 for r in ev if r.correct) / len(ev)


def abstention_rate(records: list[PredictionRecord]) -> float:
    """unknown predictions / evaluable cases"""
    ev = _evaluable(records)
    if not ev:
        return NAN
    return sum(1 for r in ev if r.abstained) / len(ev)


def coverage(records: list[PredictionRecord]) -> float:
    """1 - abstention_rate"""
    ar = abstention_rate(records)
    return NAN if ar != ar else 1 - ar  # ar != ar iff ar is NaN


def unsupported_rate_over_evaluable(records: list[PredictionRecord]) -> float:
    """unsupported predictions / all evaluable cases"""
    ev = _evaluable(records)
    if not ev:
        return NAN
    return sum(1 for r in ev if r.supported is False) / len(ev)


def unsupported_rate_over_asserted(records: list[PredictionRecord]) -> float:
    """unsupported predictions / non-abstained predictions"""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(1 for r in asserted if r.supported is False) / len(asserted)


def supported_accuracy_over_evaluable(records: list[PredictionRecord]) -> float:
    """supported_correct / all evaluable cases"""
    ev = _evaluable(records)
    if not ev:
        return NAN
    return sum(1 for r in ev if r.correct and r.supported) / len(ev)


def supported_accuracy_over_asserted(records: list[PredictionRecord]) -> float:
    """supported_correct / non-abstained cases"""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(1 for r in asserted if r.correct and r.supported) / len(asserted)


def allowed_value_compliance(records: list[PredictionRecord]) -> float:
    """Fraction of evaluable cases whose RAW (pre-canonicalization) model
    value is already an exact allowed-domain token (T0-T4/TX, N0-N3/NX,
    M0/M1/MX, or 'unknown'), needing no canonicalization. Ported unchanged
    from the v0 pilot's `scripts/03_score.py` `allowed_value_rate` -- a
    format-compliance metric, independent of correctness or grounding."""
    ev = _evaluable(records)
    if not ev:
        return NAN
    compliant = sum(1 for r in ev if r.raw_model_output.get("value") in DOMAIN)
    return compliant / len(ev)


def evidence_span_found_rate(records: list[PredictionRecord]) -> float:
    """Fraction of non-abstained, evaluable predictions whose evidence is a
    verbatim (OCR-noise-tolerant) substring of the report -- identical to
    `record.supported`, exposed as a standalone rate (spec §3 'evidence_span_found')."""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(1 for r in asserted if r.supported) / len(asserted)


def evidence_semantic_support_rate(records: list[PredictionRecord]) -> float:
    """Fraction of non-abstained, evaluable predictions whose evidence string
    contains a stage token that canonicalizes to the predicted value (spec §3
    'evidence_semantically_supports_prediction'). A syntactic proxy, not true
    NLU -- see module docstring."""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(1 for r in asserted if value_supported_by_text(r.prediction, r.evidence)) / len(asserted)


def compute_all_metrics(records: list[PredictionRecord], *, target: str | None = None) -> dict:
    """Bundle every metric above for one target (T/N/M), or pooled across all
    targets present in `records` if target is None. This is what
    `evaluate.py` calls to build metrics.json / metrics_by_target.json."""
    subset = records if target is None else [r for r in records if r.target == target]
    ev = _evaluable(subset)
    asserted = _asserted(subset)
    return {
        "target": target if target is not None else "overall",
        "n_total": len(subset),
        "n_evaluable": len(ev),
        "n_asserted": len(asserted),
        "accuracy": accuracy(subset),
        "abstention_rate": abstention_rate(subset),
        "coverage": coverage(subset),
        "unsupported_rate_over_evaluable": unsupported_rate_over_evaluable(subset),
        "unsupported_rate_over_asserted": unsupported_rate_over_asserted(subset),
        "supported_accuracy_over_evaluable": supported_accuracy_over_evaluable(subset),
        "supported_accuracy_over_asserted": supported_accuracy_over_asserted(subset),
        "allowed_value_compliance": allowed_value_compliance(subset),
        "evidence_span_found_rate": evidence_span_found_rate(subset),
        "evidence_semantic_support_rate": evidence_semantic_support_rate(subset),
    }
