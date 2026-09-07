import tempfile
from pathlib import Path

from stageground.evaluation.audit import (
    build_audit_sheet,
    read_audit_jsonl,
    score_audit,
    write_audit_jsonl,
)
from stageground.evaluation.records import build_record


def _recs(n=10):
    out = []
    for i in range(n):
        text = f"Tumor classified as T{i % 4} in report {i}."
        out.append(build_record(
            case_id=f"c{i}", arm="D_grounded", target="T", ground_truth=f"T{i % 4}",
            raw_output={"value": f"T{i % 4}", "evidence": f"T{i % 4}", "confidence": "high", "reason": None},
            report_text=text,
        ))
    return out


def test_build_audit_sheet_respects_n_and_is_deterministic():
    recs = _recs(20)
    texts = {r.case_id: f"report for {r.case_id}" for r in recs}
    rows1 = build_audit_sheet(recs, texts, n=5, seed=42)
    rows2 = build_audit_sheet(recs, texts, n=5, seed=42)
    assert len(rows1) == 5
    assert [r["case_id"] for r in rows1] == [r["case_id"] for r in rows2]


def test_build_audit_sheet_caps_at_available_records():
    recs = _recs(3)
    texts = {r.case_id: "text" for r in recs}
    rows = build_audit_sheet(recs, texts, n=100, seed=1)
    assert len(rows) == 3


def test_audit_rows_have_blank_human_fields():
    recs = _recs(5)
    texts = {r.case_id: "text" for r in recs}
    rows = build_audit_sheet(recs, texts, n=5, seed=1)
    for row in rows:
        assert row["human_supported"] is None
        assert row["human_evidence_correct"] is None
        assert row["human_prediction_correct"] is None
        assert row["human_error_type"] == ""
        assert row["reviewer_notes"] == ""
        assert "automated_supported" in row
        assert "automated_errors" in row


def test_jsonl_roundtrip():
    recs = _recs(4)
    texts = {r.case_id: "text" for r in recs}
    rows = build_audit_sheet(recs, texts, n=4, seed=1)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "audit.jsonl"
        write_audit_jsonl(rows, path)
        restored = read_audit_jsonl(path)
        assert restored == rows


def test_score_audit_no_human_labels_filled():
    rows = [
        {"case_id": "c1", "prediction": "T1", "ground_truth": "T1",
         "automated_supported": True, "human_supported": None,
         "human_prediction_correct": None},
    ]
    result = score_audit(rows)
    assert result["n_scored_supported"] == 0
    assert result["cohens_kappa_supported"] is None


def test_score_audit_perfect_agreement_with_variance():
    rows = [
        {"case_id": "c1", "prediction": "T1", "ground_truth": "T1",
         "automated_supported": True, "human_supported": True,
         "human_prediction_correct": True},
        {"case_id": "c2", "prediction": "T2", "ground_truth": "T3",
         "automated_supported": False, "human_supported": False,
         "human_prediction_correct": False},
        {"case_id": "c3", "prediction": "T1", "ground_truth": "T1",
         "automated_supported": True, "human_supported": True,
         "human_prediction_correct": True},
        {"case_id": "c4", "prediction": "T2", "ground_truth": "T3",
         "automated_supported": False, "human_supported": False,
         "human_prediction_correct": False},
    ]
    result = score_audit(rows)
    assert result["percent_agreement_supported"] == 1.0
    assert result["cohens_kappa_supported"] == 1.0
    assert result["percent_agreement_prediction_correct"] == 1.0


def test_score_audit_one_disagreement():
    rows = [
        {"case_id": "c1", "prediction": "T1", "ground_truth": "T1",
         "automated_supported": True, "human_supported": True,
         "human_prediction_correct": None},
        {"case_id": "c2", "prediction": "T2", "ground_truth": "T3",
         "automated_supported": False, "human_supported": False,
         "human_prediction_correct": None},
        {"case_id": "c3", "prediction": "T1", "ground_truth": "T1",
         "automated_supported": True, "human_supported": False,  # disagreement
         "human_prediction_correct": None},
    ]
    result = score_audit(rows)
    assert result["n_scored_supported"] == 3
    assert abs(result["percent_agreement_supported"] - 2 / 3) < 1e-9


def test_score_audit_degenerate_constant_labels_kappa_none():
    # all agree and all True -> zero variance, kappa undefined -> None,
    # but percent agreement is still meaningful (100%)
    rows = [
        {"case_id": f"c{i}", "prediction": "T1", "ground_truth": "T1",
         "automated_supported": True, "human_supported": True,
         "human_prediction_correct": None}
        for i in range(5)
    ]
    result = score_audit(rows)
    assert result["percent_agreement_supported"] == 1.0
    assert result["cohens_kappa_supported"] is None
