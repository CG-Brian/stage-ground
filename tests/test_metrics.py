import math

import pytest

from stageground.evaluation import metrics
from stageground.evaluation.records import build_record


def _rec(case_id, ground_truth, raw_value, evidence, report_text):
    return build_record(
        case_id=case_id, arm="D_grounded", target="T", ground_truth=ground_truth,
        raw_output={"value": raw_value, "evidence": evidence, "confidence": "high", "reason": None},
        report_text=report_text,
    )


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


# --- edge cases (spec §12) ---

def test_all_unknown():
    recs = [
        _rec("c1", "T2", None, None, "no descriptor"),
        _rec("c2", "T3", None, None, "no descriptor"),
        _rec("c3", "T1", None, None, "no descriptor"),
    ]
    assert metrics.accuracy(recs) == 0.0
    assert metrics.abstention_rate(recs) == 1.0
    assert metrics.coverage(recs) == 0.0
    assert metrics.span_unsupported_rate_over_evaluable(recs) == 0.0  # no assertions at all
    assert metrics.semantic_unsupported_rate_over_evaluable(recs) == 0.0
    assert _isnan(metrics.span_unsupported_rate_over_asserted(recs))  # denominator 0
    assert _isnan(metrics.semantic_unsupported_rate_over_asserted(recs))
    assert _isnan(metrics.span_grounded_accuracy_over_asserted(recs))
    assert _isnan(metrics.semantic_supported_accuracy_over_asserted(recs))


def test_accuracy_over_asserted_nan_when_all_abstained():
    recs = [
        _rec("c1", "T2", None, None, "no descriptor"),
        _rec("c2", "T3", None, None, "no descriptor"),
    ]
    assert _isnan(metrics.accuracy_over_asserted(recs))


def test_no_abstentions_full_coverage():
    recs = [
        _rec("c1", "T2", "T2", "T2 present", "Report: T2 present."),
        _rec("c2", "T3", "T1", "T1 present", "Report: T1 present."),
    ]
    assert metrics.abstention_rate(recs) == 0.0
    assert metrics.coverage(recs) == 1.0


def test_no_valid_ground_truth():
    recs = [
        _rec("c1", None, "T2", "T2 present", "Report: T2 present."),
        _rec("c2", None, "T1", None, "Report: nothing."),
    ]
    assert _isnan(metrics.accuracy(recs))
    all_metrics = metrics.compute_all_metrics(recs)
    assert all_metrics["n_evaluable"] == 0


def test_correct_but_unsupported_counts_for_accuracy_not_span_grounded_accuracy():
    recs = [_rec("c1", "T1", "T1", None, "Report mentions pT1 elsewhere, no evidence given.")]
    assert metrics.accuracy(recs) == 1.0
    assert metrics.span_grounded_accuracy_over_evaluable(recs) == 0.0
    assert metrics.semantic_supported_accuracy_over_evaluable(recs) == 0.0
    assert metrics.span_unsupported_rate_over_evaluable(recs) == 1.0
    assert metrics.semantic_unsupported_rate_over_evaluable(recs) == 1.0


def test_incorrect_but_grounded_counts_for_evidence_not_accuracy():
    recs = [_rec("c1", "T3", "T1", "T1", "Report: T1 confirmed.")]
    assert metrics.accuracy(recs) == 0.0
    assert metrics.evidence_span_found_rate(recs) == 1.0


def test_span_grounded_vs_semantic_supported_accuracy_diverge():
    # correct (T2==T2), evidence "Nodes 0/12" IS present in the report (span
    # found), but contains no T-stage token supporting T2 -- this is exactly
    # spec item 12 Case 5 ("correct but irrelevant evidence must not count
    # toward semantic-supported accuracy").
    rec = _rec("c1", "T2", "T2", "Nodes 0/12", "Tumor T2. Nodes 0/12 negative.")
    assert metrics.span_grounded_accuracy_over_evaluable([rec]) == 1.0
    assert metrics.semantic_supported_accuracy_over_evaluable([rec]) == 0.0


def test_malformed_model_output_is_evaluable_and_incorrect():
    rec = build_record(
        case_id="c1", arm="A_zero_shot", target="T", ground_truth="T2",
        raw_output=None, report_text="Report: T2.", schema_valid=False,
    )
    assert metrics.accuracy([rec]) == 0.0
    assert rec.errors == ["invalid_schema_output"]


# --- hand-computed mixed scenario ---

def _mixed_records():
    r1 = _rec("c1", "T2", "T2", "T2", "Tumor classified as T2 with clear margins.")
    r2 = _rec("c2", "T3", "T1", "T1", "Tumor classified as T1, well differentiated.")
    r3 = _rec("c3", "T1", "T1", None, "Tumor present, staging pT1 mentioned elsewhere in report.")
    r4 = _rec("c4", "T4", None, None, "No staging information available.")  # abstains
    r5 = _rec("c5", None, "T2", "T2", "Tumor classified as T2.")  # no ground truth
    return [r1, r2, r3, r4, r5]


def test_mixed_scenario_all_metrics():
    recs = _mixed_records()
    all_metrics = metrics.compute_all_metrics(recs, target="T")

    assert all_metrics["n_total"] == 5
    assert all_metrics["n_evaluable"] == 4
    assert all_metrics["n_asserted"] == 3
    assert all_metrics["accuracy_over_asserted"] == pytest.approx(2 / 3)

    assert all_metrics["accuracy"] == pytest.approx(2 / 4)
    assert all_metrics["abstention_rate"] == pytest.approx(1 / 4)
    assert all_metrics["coverage"] == pytest.approx(3 / 4)
    assert all_metrics["span_unsupported_rate_over_evaluable"] == pytest.approx(1 / 4)
    assert all_metrics["span_unsupported_rate_over_asserted"] == pytest.approx(1 / 3)
    assert all_metrics["span_grounded_accuracy_over_evaluable"] == pytest.approx(1 / 4)
    assert all_metrics["span_grounded_accuracy_over_asserted"] == pytest.approx(1 / 3)
    # r1/r2 both have grounded evidence that also semantically matches their
    # own (possibly wrong) prediction; only r1 is additionally correct, so
    # semantic_supported_accuracy equals span_grounded_accuracy here (this
    # scenario doesn't itself distinguish them -- see
    # test_span_grounded_vs_semantic_supported_accuracy_diverge for that).
    assert all_metrics["semantic_unsupported_rate_over_evaluable"] == pytest.approx(1 / 4)
    assert all_metrics["semantic_unsupported_rate_over_asserted"] == pytest.approx(1 / 3)
    assert all_metrics["semantic_supported_accuracy_over_evaluable"] == pytest.approx(1 / 4)
    assert all_metrics["semantic_supported_accuracy_over_asserted"] == pytest.approx(1 / 3)
    # r4's raw value is a bare `None` (free-form null abstention), not the
    # literal string "unknown" -> not format-compliant by the pilot's
    # allowed_value_rate definition (r1/r2/r3 are compliant: 3/4).
    assert all_metrics["allowed_value_compliance"] == pytest.approx(3 / 4)
    assert all_metrics["evidence_span_found_rate"] == pytest.approx(2 / 3)
    assert all_metrics["evidence_semantic_support_rate"] == pytest.approx(2 / 3)

    # deprecated aliases must equal their span_* replacements exactly
    assert all_metrics["unsupported_rate_over_evaluable"] == all_metrics["span_unsupported_rate_over_evaluable"]
    assert all_metrics["unsupported_rate_over_asserted"] == all_metrics["span_unsupported_rate_over_asserted"]
    assert all_metrics["supported_accuracy_over_evaluable"] == all_metrics["span_grounded_accuracy_over_evaluable"]
    assert all_metrics["supported_accuracy_over_asserted"] == all_metrics["span_grounded_accuracy_over_asserted"]


def test_compute_all_metrics_filters_by_target():
    t_rec = build_record(
        case_id="c1", arm="D_grounded", target="T", ground_truth="T2",
        raw_output={"value": "T2", "evidence": "T2", "confidence": "high", "reason": None},
        report_text="T2 present.",
    )
    n_rec = build_record(
        case_id="c1", arm="D_grounded", target="N", ground_truth="N0",
        raw_output={"value": "N1", "evidence": None, "confidence": "high", "reason": None},
        report_text="T2 present.",
    )
    only_t = metrics.compute_all_metrics([t_rec, n_rec], target="T")
    assert only_t["n_total"] == 1
    assert only_t["accuracy"] == 1.0

    pooled = metrics.compute_all_metrics([t_rec, n_rec], target=None)
    assert pooled["n_total"] == 2
    assert pooled["accuracy"] == pytest.approx(0.5)
