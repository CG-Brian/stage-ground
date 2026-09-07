from stageground.evaluation.mstage_analysis import (
    M_CATEGORY_AMBIGUOUS,
    M_CATEGORY_EXPLICIT_TOKEN,
    M_CATEGORY_METASTATIC_DESCRIBED,
    M_CATEGORY_NO_EVIDENCE,
    build_mstage_annotation_sheet,
    classify_m_evidence,
)
from stageground.evaluation.records import build_record


def test_explicit_m_token():
    text = "Distant metastasis confirmed (pM1) in liver."
    assert classify_m_evidence(text) == M_CATEGORY_EXPLICIT_TOKEN


def test_metastatic_described_without_token():
    text = "Metastatic disease noted in regional lymph nodes."
    assert classify_m_evidence(text) == M_CATEGORY_METASTATIC_DESCRIBED


def test_no_m_evidence():
    text = "Primary tumor pT3 invades pericolic fat. Nodes 0/12."
    assert classify_m_evidence(text) == M_CATEGORY_NO_EVIDENCE


def test_ambiguous_when_token_and_hedged_metastatic_language_conflict():
    text = (
        "Final diagnosis: pM0. Suspicious for possible metastatic disease; "
        "further imaging pending."
    )
    assert classify_m_evidence(text) == M_CATEGORY_AMBIGUOUS


def test_empty_text_is_no_evidence():
    assert classify_m_evidence("") == M_CATEGORY_NO_EVIDENCE
    assert classify_m_evidence(None) == M_CATEGORY_NO_EVIDENCE


def test_annotation_sheet_has_required_fields():
    text = "Distant metastasis confirmed (pM1) in liver."
    rec = build_record(
        case_id="c1", arm="D_grounded", target="M", ground_truth="M1",
        raw_output={"value": "M1", "evidence": "pM1", "confidence": "high", "reason": None},
        report_text=text,
    )
    rows = build_mstage_annotation_sheet([rec], {"c1": text})
    assert len(rows) == 1
    row = rows[0]
    for key in (
        "case_id", "arm", "report_excerpt", "automated_category",
        "prediction", "ground_truth", "supported", "human_category", "notes",
    ):
        assert key in row
    assert row["automated_category"] == M_CATEGORY_EXPLICIT_TOKEN
    assert row["human_category"] == ""
    assert row["notes"] == ""


def test_annotation_sheet_only_includes_requested_target():
    text = "Distant metastasis confirmed (pM1) in liver."
    m_rec = build_record(
        case_id="c1", arm="D_grounded", target="M", ground_truth="M1",
        raw_output={"value": "M1", "evidence": "pM1", "confidence": "high", "reason": None},
        report_text=text,
    )
    t_rec = build_record(
        case_id="c1", arm="D_grounded", target="T", ground_truth="T2",
        raw_output={"value": "T2", "evidence": None, "confidence": "high", "reason": None},
        report_text=text,
    )
    rows = build_mstage_annotation_sheet([m_rec, t_rec], {"c1": text})
    assert len(rows) == 1
    assert rows[0]["case_id"] == "c1"
