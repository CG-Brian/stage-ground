import json

from stageground.evaluation.records import (
    PredictionRecord,
    build_record,
    from_jsonl,
    to_jsonl,
)

REPORT = "Primary tumor pT3 invades pericolic fat. No distant metastasis identified (pM0)."


def test_correct_and_supported():
    rec = build_record(
        case_id="c1", arm="D_grounded", target="T", ground_truth="T3",
        raw_output={"value": "T3", "evidence": "pT3 invades pericolic fat",
                    "confidence": "high", "reason": None},
        report_text=REPORT,
    )
    assert rec.prediction == "T3"
    assert rec.correct is True
    assert rec.abstained is False
    assert rec.supported is True
    assert rec.errors == []


def test_incorrect_prediction():
    rec = build_record(
        case_id="c2", arm="D_grounded", target="M", ground_truth="M0",
        raw_output={"value": "M1", "evidence": "pM0", "confidence": "low", "reason": None},
        report_text=REPORT,
    )
    assert rec.correct is False


def test_no_ground_truth_not_evaluable():
    rec = build_record(
        case_id="c3", arm="C_constrained", target="N", ground_truth=None,
        raw_output={"value": "N0", "evidence": None, "confidence": "medium", "reason": None},
        report_text=REPORT,
    )
    assert rec.correct is None


def test_abstained_supported_is_none():
    rec = build_record(
        case_id="c4", arm="D_grounded", target="M", ground_truth="M0",
        raw_output={"value": "unknown", "evidence": None, "confidence": "low",
                    "reason": "no basis"},
        report_text="no staging descriptors at all",
    )
    assert rec.abstained is True
    assert rec.supported is None
    assert rec.correct is False  # 'unknown' != 'M0'


def test_malformed_output_defaults_to_invalid_not_silently_abstained():
    rec = build_record(
        case_id="c5", arm="A_zero_shot", target="T", ground_truth="T2",
        raw_output=None,
        report_text=REPORT,
        schema_valid=False,
    )
    assert rec.prediction == "INVALID"
    assert rec.abstained is False
    assert rec.errors == ["invalid_schema_output"]
    assert rec.correct is False
    assert rec.supported is False


def test_raw_model_output_preserved_verbatim():
    raw = {"value": "T3", "evidence": "pT3", "confidence": "high", "reason": None}
    rec = build_record(
        case_id="c6", arm="D_grounded", target="T", ground_truth="T3",
        raw_output=raw, report_text=REPORT,
    )
    assert rec.raw_model_output == raw


def test_jsonl_roundtrip_preserves_all_fields():
    rec = build_record(
        case_id="c7", arm="D_grounded", target="M", ground_truth="M0",
        raw_output={"value": "M1", "evidence": "fabricated span",
                    "confidence": "low", "reason": None},
        report_text=REPORT,
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "predictions.jsonl"
        to_jsonl([rec], path)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["case_id"] == "c7"
        assert parsed["raw_model_output"]["evidence"] == "fabricated span"

        restored = from_jsonl(path)
        assert restored == [rec]


def test_prediction_record_is_a_dataclass_with_expected_fields():
    rec = build_record(
        case_id="c8", arm="A_zero_shot", target="T", ground_truth=None,
        raw_output={"value": "T1", "evidence": None, "confidence": "low", "reason": None},
        report_text=REPORT,
    )
    assert isinstance(rec, PredictionRecord)
    for field in ("case_id", "arm", "target", "ground_truth", "prediction",
                  "evidence", "correct", "abstained", "supported", "errors",
                  "raw_model_output"):
        assert hasattr(rec, field)
