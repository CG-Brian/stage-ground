import math

from stageground.evaluation.audit_scoring import (
    _confusion_and_agreement,
    analyze_m0_prior_prediction,
    analyze_source_sufficiency,
    score_reviewed_audit,
)


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


# --- _confusion_and_agreement ---

def test_confusion_matrix_all_four_cells():
    pairs = [(True, True), (True, False), (False, False), (False, True)]
    result = _confusion_and_agreement(pairs)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["tn"] == 1
    assert result["fn"] == 1
    assert result["n"] == 4
    assert result["percent_agreement"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["specificity"] == 0.5


def test_confusion_matrix_zero_variance_kappa_none():
    pairs = [(True, True)] * 5
    result = _confusion_and_agreement(pairs)
    assert result["percent_agreement"] == 1.0
    assert result["cohens_kappa"] is None
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["specificity"] is None  # no negatives at all


def test_confusion_matrix_empty_no_crash():
    result = _confusion_and_agreement([])
    assert result["n"] == 0
    assert _isnan(result["percent_agreement"])
    assert result["cohens_kappa"] is None
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["specificity"] is None


# --- score_reviewed_audit ---

def _key_row(audit_id, *, target, arm="C_constrained", auto_span=True, auto_semantic=True, gt="M1", pred="M1", pattern=None):
    return {
        "audit_id": audit_id, "case_id": f"case_{audit_id}", "arm": arm, "target": target,
        "ground_truth": gt, "prediction": pred,
        "automated_evidence_span_found": auto_span,
        "automated_semantic_support": auto_semantic,
        "automated_errors": [], "arm_pattern": pattern,
    }


def _reviewer_row(audit_id, *, target, human_span=True, human_semantic=True, human_suff=None, human_expl=None, human_infer=None, pred="M1"):
    return {
        "audit_id": audit_id, "target": target, "report_excerpt": "...", "report_truncated": False,
        "prediction": pred, "evidence": "M1 present",
        "human_evidence_span_found": human_span, "human_semantic_support": human_semantic,
        "human_prediction_correct": None,
        "human_source_has_explicit_stage": human_expl,
        "human_source_has_inferential_evidence": human_infer,
        "human_source_sufficient_for_stage": human_suff,
        "human_confidence": "high", "human_error_type": "", "reviewer_notes": "",
    }


def test_score_reviewed_audit_per_target_and_overall():
    key_rows = [
        _key_row("A0001", target="T", auto_span=True, auto_semantic=True),
        _key_row("A0002", target="T", auto_span=True, auto_semantic=False),
        _key_row("A0003", target="M", auto_span=True, auto_semantic=True),
        _key_row("A0004", target="M", auto_span=False, auto_semantic=False),
    ]
    reviewer_rows = [
        _reviewer_row("A0001", target="T", human_span=True, human_semantic=True),
        _reviewer_row("A0002", target="T", human_span=True, human_semantic=True),  # disagree on semantic
        _reviewer_row("A0003", target="M", human_span=True, human_semantic=True),
        _reviewer_row("A0004", target="M", human_span=False, human_semantic=False),
    ]
    result = score_reviewed_audit(reviewer_rows, key_rows)

    assert result["T"]["evidence_span_found"]["n"] == 2
    assert result["T"]["evidence_span_found"]["percent_agreement"] == 1.0
    assert result["T"]["semantic_support"]["percent_agreement"] == 0.5

    assert result["M"]["evidence_span_found"]["percent_agreement"] == 1.0
    assert result["M"]["semantic_support"]["percent_agreement"] == 1.0

    assert result["overall"]["evidence_span_found"]["n"] == 4
    assert result["overall"]["semantic_support"]["n"] == 4


def test_score_reviewed_audit_skips_unreviewed_rows():
    key_rows = [_key_row("A0001", target="M")]
    reviewer_rows = [_reviewer_row("A0001", target="M", human_span=None, human_semantic=None)]
    result = score_reviewed_audit(reviewer_rows, key_rows)
    assert result["M"]["evidence_span_found"]["n"] == 0
    assert result["M"]["semantic_support"]["n"] == 0
    assert result["overall"]["evidence_span_found"]["n"] == 0


# --- analyze_source_sufficiency ---

def test_analyze_source_sufficiency_per_target():
    key_rows = [
        _key_row("A0001", target="T"),
        _key_row("A0002", target="M", pattern="priority_predicts_comparison_abstains"),
        _key_row("A0003", target="M", pattern="both_predict"),
    ]
    reviewer_rows = [
        _reviewer_row("A0001", target="T", human_suff=True, human_expl=True, human_infer=False),
        _reviewer_row("A0002", target="M", human_suff=False, human_expl=False, human_infer=True),
        _reviewer_row("A0003", target="M", human_suff=True, human_expl=True, human_infer=True),
    ]
    result = analyze_source_sufficiency(reviewer_rows, key_rows)

    assert result["T"]["pct_source_sufficient"] == 1.0
    assert result["M"]["pct_source_sufficient"] == 0.5  # 1 of 2 M rows sufficient
    assert result["M_by_arm_pattern"]["priority_predicts_comparison_abstains"]["pct_source_sufficient"] == 0.0
    assert result["M_by_arm_pattern"]["both_predict"]["pct_source_sufficient"] == 1.0


def test_analyze_source_sufficiency_abstention_still_counts_when_filled():
    key_rows = [_key_row("A0001", target="M", pred="unknown", auto_span=None, auto_semantic=None)]
    reviewer_rows = [
        _reviewer_row("A0001", target="M", human_span=None, human_semantic=None, human_suff=True, human_expl=False, human_infer=False),
    ]
    result = analyze_source_sufficiency(reviewer_rows, key_rows)
    assert result["M"]["n_source_sufficient_scored"] == 1
    assert result["M"]["pct_source_sufficient"] == 1.0
    # and grounding scoring is untouched by this (separate function), doesn't crash
    grounding = score_reviewed_audit(reviewer_rows, key_rows)
    assert grounding["M"]["evidence_span_found"]["n"] == 0


# --- analyze_m0_prior_prediction ---

def test_analyze_m0_prior_prediction():
    key_rows = [
        _key_row("A0001", target="M", arm="C_constrained", pred="M0", gt="M0"),
        _key_row("A0002", target="M", arm="C_constrained", pred="M0", gt="M1"),  # wrong
        _key_row("A0003", target="M", arm="D_grounded", pred="M0", gt="M0"),  # different arm, excluded
        _key_row("A0004", target="T", arm="C_constrained", pred="M0", gt="M0"),  # different target, excluded
    ]
    reviewer_rows = [
        _reviewer_row("A0001", target="M", human_suff=True, human_semantic=True, human_expl=True, pred="M0"),
        _reviewer_row("A0002", target="M", human_suff=False, human_semantic=False, human_expl=False, pred="M0"),
        _reviewer_row("A0003", target="M", human_suff=True, human_semantic=True, human_expl=True, pred="M0"),
        _reviewer_row("A0004", target="T", human_suff=True, human_semantic=True, human_expl=True, pred="M0"),
    ]
    result = analyze_m0_prior_prediction(reviewer_rows, key_rows, priority_arm="C_constrained")
    assert result["n"] == 2
    assert result["accuracy"] == 0.5
    assert result["pct_source_sufficient"] == 0.5
    assert result["pct_semantic_support"] == 0.5
    assert result["pct_explicit_m_token_present"] == 0.5


def test_analyze_m0_prior_prediction_no_matching_rows():
    result = analyze_m0_prior_prediction([], [], priority_arm="C_constrained")
    assert result["n"] == 0
    assert _isnan(result["accuracy"])
