"""Spec item 12: unit tests for the evidence_span_found /
evidence_semantically_supports_prediction split.

See also tests/test_metrics.py::test_span_grounded_vs_semantic_supported_accuracy_diverge
for Case 5's full "must not count toward semantic-supported accuracy" claim.
"""

from stageground.evaluation.records import build_record
from stageground.evaluation.metrics import (
    semantic_supported_accuracy_over_evaluable,
    span_grounded_accuracy_over_evaluable,
)


def _rec(prediction_value, evidence, report_text, ground_truth="M1"):
    return build_record(
        case_id="c1", arm="D_grounded", target="M", ground_truth=ground_truth,
        raw_output={"value": prediction_value, "evidence": evidence,
                    "confidence": "high", "reason": None},
        report_text=report_text,
    )


def test_case1_grounded_and_semantically_supportive():
    rec = _rec("M1", "M1 metastatic disease", "Findings: M1 metastatic disease confirmed in liver.")
    assert rec.evidence_span_found is True
    assert rec.evidence_semantically_supports_prediction is True


def test_case2_grounded_but_not_semantically_supportive():
    rec = _rec(
        "M1", "No distant metastasis identified",
        "Findings: No distant metastasis identified on imaging review.",
    )
    assert rec.evidence_span_found is True
    assert rec.evidence_semantically_supports_prediction is False


def test_case3_evidence_string_not_present_in_report():
    rec = _rec("M1", "M1 confirmed by PET scan", "Findings: no metastatic disease noted anywhere.")
    assert rec.evidence_span_found is False
    assert rec.evidence_semantically_supports_prediction is False


def test_case4_abstention_has_none_for_both_fields():
    rec = build_record(
        case_id="c1", arm="D_grounded", target="M", ground_truth="M0",
        raw_output={"value": "unknown", "evidence": None, "confidence": "low",
                    "reason": "no basis"},
        report_text="Findings: no staging information available.",
    )
    assert rec.evidence_span_found is None
    assert rec.evidence_semantically_supports_prediction is None


def test_case5_correct_but_irrelevant_evidence_excluded_from_semantic_accuracy():
    # Correct prediction (M1 == gold M1), evidence IS present in the report
    # (span found), but is irrelevant to the M-stage claim -- must count
    # toward span-grounded accuracy but NOT semantic-supported accuracy.
    rec = _rec(
        "M1", "Nodes: 2 of 14 involved",
        "Nodes: 2 of 14 involved. Distant metastasis: M1.",
    )
    assert rec.evidence_span_found is True
    assert rec.evidence_semantically_supports_prediction is False
    assert span_grounded_accuracy_over_evaluable([rec]) == 1.0
    assert semantic_supported_accuracy_over_evaluable([rec]) == 0.0
