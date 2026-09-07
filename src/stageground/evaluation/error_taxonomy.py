"""Error taxonomy for incorrect or problematic predictions (spec §4).

Multiple flags may coexist on a single prediction -- this module never
forces a case into exactly one category. Categories:

  hallucinated_stage                 asserted a wrong value with no textual basis
  wrong_stage_with_supporting_evidence  asserted a wrong value the text DOES support
                                      (candidate registry/gold-discordance, not fabrication)
  evidence_span_not_found            no evidence given, or it isn't a verbatim
                                      (OCR-noise-tolerant) substring of the report
  evidence_does_not_support_prediction  evidence span IS in the report, but it
                                      doesn't contain a token supporting the
                                      predicted value (syntactic proxy, see
                                      stageground.evaluation.normalize.value_supported_by_text)
  missed_explicit_stage              abstained, but an explicit stage token for
                                      the gold value was present in the report
  over_abstention                    same underlying condition as
                                      missed_explicit_stage -- both fire together
                                      when an abstention disagrees with a
                                      textually-findable gold value. Kept as two
                                      names because M-stage analysis (spec §5)
                                      cares about "was there literally a token"
                                      while the abstention-tradeoff framing
                                      (spec §7) cares about "was this abstention
                                      a miss." This overlap is intentional, not
                                      a bug.
  invalid_normalization              raw value present but not a recognizable
                                      stage token (canonicalize() failed)
  invalid_schema_output              the model's raw JSON failed schema validation
  other                              reserved fallback bucket; not reachable by
                                      the current rules below, kept so the
                                      taxonomy vocabulary has a catch-all if new
                                      failure modes are added later
"""

from __future__ import annotations

from stageground.evaluation.normalize import grounded, value_supported_by_text


def classify_errors(
    *,
    ground_truth: str | None,
    prediction: str,
    evidence: str | None,
    report_text: str,
    raw_value: object,
    schema_valid: bool,
) -> list[str]:
    """Classify one (target) prediction into zero or more taxonomy flags.

    `prediction` and `ground_truth` are already normalized/canonicalized
    ("T3" / "unknown" / "INVALID"), never raw model strings. `raw_value` is
    only used for documentation/debugging context by callers; it does not
    affect classification (the normalized `prediction` already encodes
    whether normalization failed via "INVALID").
    """
    if not schema_valid:
        return ["invalid_schema_output"]

    if prediction == "INVALID":
        return ["invalid_normalization"]

    if prediction == "unknown":
        flags: list[str] = []
        if ground_truth is not None and value_supported_by_text(ground_truth, report_text):
            flags.append("over_abstention")
            flags.append("missed_explicit_stage")
        return flags

    # prediction asserts a real stage value
    flags = []
    span_found = evidence is not None and grounded(evidence, report_text)
    if not span_found:
        flags.append("evidence_span_not_found")
    elif not value_supported_by_text(prediction, evidence):
        flags.append("evidence_does_not_support_prediction")

    if ground_truth is not None and prediction != ground_truth:
        if "evidence_span_not_found" in flags:
            flags.append("hallucinated_stage")
        else:
            flags.append("wrong_stage_with_supporting_evidence")

    return flags
