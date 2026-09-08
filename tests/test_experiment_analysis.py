import math

import pytest

from stageground.evaluation.experiment_analysis import (
    coverage_vs_accuracy_table,
    error_taxonomy_rates,
    gold_label_distribution,
    m0_prediction_analysis,
    predicted_label_distribution,
)
from stageground.evaluation.records import build_record


def _isnan(x):
    return isinstance(x, float) and math.isnan(x)


def _rec(case_id, arm, target, gt, pred, evidence, text):
    return build_record(
        case_id=case_id, arm=arm, target=target, ground_truth=gt,
        raw_output={"value": pred, "evidence": evidence, "confidence": "high", "reason": None},
        report_text=text,
    )


# --- m0_prediction_analysis ---

def _m0_pool():
    records = []
    # arm X: 4 M-target cases -- 2 M0 predictions (1 correct+grounded, 1 wrong),
    # 1 M1 prediction, 1 abstention
    records.append(_rec("c1", "X", "M", "M0", "M0", "No distant metastasis identified.",
                         "No distant metastasis identified in this specimen."))
    records.append(_rec("c2", "X", "M", "M1", "M0", None, "Findings unrelated to metastasis."))
    records.append(_rec("c3", "X", "M", "M1", "M1", "M1", "Distant metastasis confirmed (M1)."))
    records.append(_rec("c4", "X", "M", "M0", None, None, "No staging information available."))  # abstains
    return records


def test_m0_prediction_analysis_counts_and_rates():
    df = m0_prediction_analysis(_m0_pool()).set_index("arm")
    row = df.loc["X"]
    assert row["n_m0_predictions"] == 2  # c1, c2
    assert row["n_m0_evaluable"] == 2
    assert row["m0_accuracy"] == pytest.approx(0.5)  # c1 correct, c2 wrong
    # asserted M predictions: c1, c2, c3 (c4 abstains) -> 2/3 are M0
    assert row["m0_proportion_of_asserted_m"] == pytest.approx(2 / 3)
    # c1 has grounded+semantically-supporting evidence, c2 has none -> 1/2 each
    assert row["m0_semantic_supported_rate"] == pytest.approx(0.5)
    assert row["m0_span_grounded_rate"] == pytest.approx(0.5)


def test_m0_prediction_analysis_no_m0_predictions_no_crash():
    records = [_rec("c1", "Y", "M", "M1", "M1", "M1", "Distant metastasis confirmed (M1).")]
    df = m0_prediction_analysis(records).set_index("arm")
    row = df.loc["Y"]
    assert row["n_m0_predictions"] == 0
    assert _isnan(row["m0_accuracy"])
    assert _isnan(row["m0_semantic_supported_rate"])


# --- gold_label_distribution ---

def test_gold_label_distribution_dedupes_across_arms():
    records = [
        _rec("c1", "X", "M", "M0", "M0", None, "text"),
        _rec("c1", "Y", "M", "M0", "M1", None, "text"),  # same case, different arm, SAME gold
        _rec("c2", "X", "M", "M1", "M1", None, "text"),
    ]
    df = gold_label_distribution(records)
    m_rows = df[df["target"] == "M"].set_index("value")
    assert m_rows.loc["M0", "count"] == 1  # deduped, not 2
    assert m_rows.loc["M1", "count"] == 1
    assert m_rows["count"].sum() == 2  # 2 distinct cases, not 3 rows


def test_gold_label_distribution_ignores_missing_gold():
    records = [_rec("c1", "X", "T", None, "T2", None, "text")]
    df = gold_label_distribution(records)
    assert df[df["target"] == "T"].empty


# --- predicted_label_distribution ---

def test_predicted_label_distribution_per_arm_no_dedup():
    records = [
        _rec("c1", "X", "M", "M0", "M0", None, "text"),
        _rec("c2", "X", "M", "M0", "M0", None, "text"),
        _rec("c1", "Y", "M", "M0", "M1", None, "text"),
    ]
    df = predicted_label_distribution(records)
    x_rows = df[(df["target"] == "M") & (df["arm"] == "X")].set_index("value")
    assert x_rows.loc["M0", "count"] == 2
    y_rows = df[(df["target"] == "M") & (df["arm"] == "Y")].set_index("value")
    assert y_rows.loc["M1", "count"] == 1


# --- coverage_vs_accuracy_table ---

def test_coverage_vs_accuracy_table():
    records = [
        _rec("c1", "X", "T", "T2", "T2", "T2", "Tumor classified as T2."),
        _rec("c2", "X", "T", "T3", "T1", "T1", "Tumor classified as T1."),
        _rec("c3", "X", "T", "T4", None, None, "No staging information available."),  # abstains
    ]
    df = coverage_vs_accuracy_table(records, target="T").set_index("arm")
    row = df.loc["X"]
    assert row["coverage"] == pytest.approx(2 / 3)
    assert row["accuracy_over_asserted"] == pytest.approx(0.5)  # c1 correct, c2 wrong, of 2 asserted


def test_coverage_vs_accuracy_table_overall_pools_targets():
    records = [
        _rec("c1", "X", "T", "T2", "T2", "T2", "T2 present."),
        _rec("c1", "X", "N", "N0", "N1", None, "N0 present."),
    ]
    df = coverage_vs_accuracy_table(records, target=None).set_index("arm")
    assert df.loc["X", "coverage"] == 1.0  # neither abstains
    assert df.loc["X", "accuracy_over_asserted"] == pytest.approx(0.5)  # 1 of 2 correct


# --- error_taxonomy_rates ---

def test_error_taxonomy_rates_counts_and_denominators():
    records = [
        _rec("c1", "X", "T", "T2", "T2", "T2", "Tumor classified as T2."),  # correct+grounded -> no errors
        _rec("c2", "X", "T", "T3", "T1", "widespread invasion",  # wrong, fabricated evidence
             "Tumor classified as T1."),
        _rec("c3", "X", "T", "T2", None, None, "Report mentions pT2 elsewhere in text."),  # abstains, gold findable
    ]
    df = error_taxonomy_rates(records, target="T").set_index("error_category")
    assert df.loc["evidence_span_not_found", "count"] == 1
    assert df.loc["evidence_span_not_found", "n_evaluable"] == 3
    assert df.loc["evidence_span_not_found", "rate_over_evaluable"] == pytest.approx(1 / 3)
    assert df.loc["hallucinated_stage", "count"] == 1
    assert df.loc["over_abstention", "count"] == 1
    assert df.loc["missed_explicit_stage", "count"] == 1
    assert df.loc["wrong_stage_with_supporting_evidence", "count"] == 0


def test_error_taxonomy_rates_no_crash_on_empty_records():
    df = error_taxonomy_rates([], target="M")
    assert list(df.columns) == ["arm", "error_category", "count", "n_evaluable", "rate_over_evaluable"]
    assert df.empty
