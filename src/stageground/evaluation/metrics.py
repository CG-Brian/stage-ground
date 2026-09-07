"""Explicit metric definitions (spec §3).

Every function takes a `list[PredictionRecord]` -- already filtered to the
arm/target of interest by the caller, or passed as-is to aggregate across
targets. All metrics are scoped to **evaluable cases** (ground truth
available for that target) so every number in a comparison table shares the
same population and denominators are never ambiguous. A denominator of zero
never raises: it returns `float("nan")`, so callers (tables/JSON output) can
render "n/a" instead of crashing on edge cases like "all predictions
unknown" or "no ground truth in this sample" (spec §12).

Metric names are precise about WHICH notion of grounding they use, matching
the two explicit `PredictionRecord` fields (see that module's docstring):

  - `span_*` metrics use `evidence_span_found` only -- a syntactic fact
    (verbatim, OCR-noise-tolerant substring match against the report). This
    is exactly the v0 pilot's validated (92%-agreement-audited) `grounded()`
    check; nothing about that historical definition has changed.
  - `semantic_*` metrics additionally require
    `evidence_semantically_supports_prediction` -- an automated HEURISTIC
    proxy (regex stage-token match inside the evidence string), explicitly
    NOT equivalent to human semantic judgment. Treat `semantic_*` numbers as
    provisional until corroborated by the manual audit tooling in
    `stageground.evaluation.audit`.

`unsupported_rate_over_evaluable`/`_over_asserted` and
`supported_accuracy_over_evaluable`/`_over_asserted` are kept as DEPRECATED
aliases for `span_unsupported_rate_*` and `span_grounded_accuracy_*`
respectively (their original semantics were always span-only, never
semantic) -- new code should call the `span_*` names directly.
"""

from __future__ import annotations

from stageground.evaluation.normalize import DOMAIN
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


def span_unsupported_rate_over_evaluable(records: list[PredictionRecord]) -> float:
    """Non-abstained predictions with no evidence span found / all evaluable
    cases. Span-only: does not consider whether the evidence, if grounded,
    actually says what was predicted."""
    ev = _evaluable(records)
    if not ev:
        return NAN
    return sum(1 for r in ev if r.evidence_span_found is False) / len(ev)


def span_unsupported_rate_over_asserted(records: list[PredictionRecord]) -> float:
    """Same numerator as `span_unsupported_rate_over_evaluable` / non-abstained
    (evaluable) predictions."""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(1 for r in asserted if r.evidence_span_found is False) / len(asserted)


def semantic_unsupported_rate_over_evaluable(records: list[PredictionRecord]) -> float:
    """Non-abstained predictions where the evidence span is absent OR the
    evidence does not semantically support the predicted value (heuristic
    proxy) / all evaluable cases. Strictly >= `span_unsupported_rate_over_evaluable`
    for the same population, since every span-not-found case is also
    semantic-not-supported by construction."""
    ev = _evaluable(records)
    if not ev:
        return NAN
    return sum(
        1 for r in ev
        if r.evidence_span_found is False or r.evidence_semantically_supports_prediction is False
    ) / len(ev)


def semantic_unsupported_rate_over_asserted(records: list[PredictionRecord]) -> float:
    """Same numerator as `semantic_unsupported_rate_over_evaluable` /
    non-abstained (evaluable) predictions."""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(
        1 for r in asserted
        if r.evidence_span_found is False or r.evidence_semantically_supports_prediction is False
    ) / len(asserted)


def span_grounded_accuracy_over_evaluable(records: list[PredictionRecord]) -> float:
    """correct AND evidence_span_found / all evaluable cases. Span-only --
    does not require the evidence to actually support the predicted value,
    only that it's real (verbatim) text from the report."""
    ev = _evaluable(records)
    if not ev:
        return NAN
    return sum(1 for r in ev if r.correct and r.evidence_span_found) / len(ev)


def span_grounded_accuracy_over_asserted(records: list[PredictionRecord]) -> float:
    """Same numerator as `span_grounded_accuracy_over_evaluable` / non-abstained
    (evaluable) predictions."""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(1 for r in asserted if r.correct and r.evidence_span_found) / len(asserted)


def semantic_supported_accuracy_over_evaluable(records: list[PredictionRecord]) -> float:
    """correct AND evidence_span_found AND evidence_semantically_supports_prediction
    / all evaluable cases. The `AND evidence_span_found` conjunct is
    redundant (semantic support already implies span-found by construction)
    but kept explicit per spec §5's definition. This is the closest proxy to
    "true grounding" this module computes automatically -- still a heuristic,
    see module docstring."""
    ev = _evaluable(records)
    if not ev:
        return NAN
    return sum(
        1 for r in ev
        if r.correct and r.evidence_span_found and r.evidence_semantically_supports_prediction
    ) / len(ev)


def semantic_supported_accuracy_over_asserted(records: list[PredictionRecord]) -> float:
    """Same numerator as `semantic_supported_accuracy_over_evaluable` /
    non-abstained (evaluable) predictions."""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(
        1 for r in asserted
        if r.correct and r.evidence_span_found and r.evidence_semantically_supports_prediction
    ) / len(asserted)


# --- Deprecated aliases (pre-span/semantic-split names) ---
# Kept because the original names were always span-only, never semantic;
# renamed rather than redefined. New code should use the span_* names.

def unsupported_rate_over_evaluable(records: list[PredictionRecord]) -> float:
    """Deprecated alias for `span_unsupported_rate_over_evaluable`."""
    return span_unsupported_rate_over_evaluable(records)


def unsupported_rate_over_asserted(records: list[PredictionRecord]) -> float:
    """Deprecated alias for `span_unsupported_rate_over_asserted`."""
    return span_unsupported_rate_over_asserted(records)


def supported_accuracy_over_evaluable(records: list[PredictionRecord]) -> float:
    """Deprecated alias for `span_grounded_accuracy_over_evaluable`."""
    return span_grounded_accuracy_over_evaluable(records)


def supported_accuracy_over_asserted(records: list[PredictionRecord]) -> float:
    """Deprecated alias for `span_grounded_accuracy_over_asserted`."""
    return span_grounded_accuracy_over_asserted(records)


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
    verbatim (OCR-noise-tolerant) substring of the report."""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(1 for r in asserted if r.evidence_span_found) / len(asserted)


def evidence_semantic_support_rate(records: list[PredictionRecord]) -> float:
    """Fraction of non-abstained, evaluable predictions whose evidence
    semantically supports the predicted value (heuristic proxy -- see module
    docstring), among those with a span found."""
    asserted = _asserted(records)
    if not asserted:
        return NAN
    return sum(1 for r in asserted if r.evidence_semantically_supports_prediction) / len(asserted)


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
        "span_unsupported_rate_over_evaluable": span_unsupported_rate_over_evaluable(subset),
        "span_unsupported_rate_over_asserted": span_unsupported_rate_over_asserted(subset),
        "semantic_unsupported_rate_over_evaluable": semantic_unsupported_rate_over_evaluable(subset),
        "semantic_unsupported_rate_over_asserted": semantic_unsupported_rate_over_asserted(subset),
        "span_grounded_accuracy_over_evaluable": span_grounded_accuracy_over_evaluable(subset),
        "span_grounded_accuracy_over_asserted": span_grounded_accuracy_over_asserted(subset),
        "semantic_supported_accuracy_over_evaluable": semantic_supported_accuracy_over_evaluable(subset),
        "semantic_supported_accuracy_over_asserted": semantic_supported_accuracy_over_asserted(subset),
        "allowed_value_compliance": allowed_value_compliance(subset),
        "evidence_span_found_rate": evidence_span_found_rate(subset),
        "evidence_semantic_support_rate": evidence_semantic_support_rate(subset),
        # deprecated aliases, kept for anything written against the pre-split schema
        "unsupported_rate_over_evaluable": unsupported_rate_over_evaluable(subset),
        "unsupported_rate_over_asserted": unsupported_rate_over_asserted(subset),
        "supported_accuracy_over_evaluable": supported_accuracy_over_evaluable(subset),
        "supported_accuracy_over_asserted": supported_accuracy_over_asserted(subset),
    }
