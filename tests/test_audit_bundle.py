import json

import pandas as pd

from stageground.evaluation.audit import (
    _report_excerpt,
    build_audit_sheet,
    build_reviewer_and_key,
    main,
    score_audit,
    write_audit_bundle,
)
from stageground.evaluation.records import PredictionRecord, to_jsonl

GROUNDING_BLIND_FIELDS = {
    "audit_id", "target", "report_excerpt", "report_truncated",
    "prediction", "evidence",
    "human_evidence_span_found", "human_semantic_support", "human_prediction_correct",
    "human_source_has_explicit_stage", "human_source_has_inferential_evidence",
    "human_source_sufficient_for_stage", "human_confidence", "human_error_type",
    "reviewer_notes",
}
WITH_GOLD_FIELDS = GROUNDING_BLIND_FIELDS | {"ground_truth"}


def _rec(case_id, arm, target, *, abstained=False, evidence="pM1 present", semantic=True, gt="M1", pred="M1"):
    return PredictionRecord(
        case_id=case_id, arm=arm, target=target,
        ground_truth=gt, prediction="unknown" if abstained else pred,
        evidence=None if abstained else evidence,
        correct=(pred == gt), abstained=abstained,
        evidence_span_found=None if abstained else True,
        evidence_semantically_supports_prediction=None if abstained else semantic,
        errors=[], raw_model_output={"value": pred, "evidence": evidence},
    )


def _pool():
    records = []
    for i in range(6):
        records.append(_rec(f"c{i}", "C_constrained", "M"))
        records.append(_rec(f"c{i}", "D_grounded", "M", abstained=(i % 2 == 0)))
    for i in range(4):
        records.append(_rec(f"t{i}", "C_constrained", "T", gt="T2", pred="T2"))
    return records


def _texts(records):
    return {r.case_id: f"Report body for {r.case_id}. pM1 present. Additional narrative text." for r in records}


# --- blinding ---

def test_reviewer_rows_have_no_arm_key():
    records = _pool()
    reviewer, key, _ = build_reviewer_and_key(
        records, _texts(records), target_counts={"T": 2, "M": 3}, seed=1,
    )
    for row in reviewer:
        assert "arm" not in row
        assert "case_id" not in row


def test_reviewer_rows_have_no_automated_judgments():
    records = _pool()
    reviewer, key, _ = build_reviewer_and_key(
        records, _texts(records), target_counts={"T": 2, "M": 3}, seed=1,
    )
    for row in reviewer:
        assert "automated_evidence_span_found" not in row
        assert "automated_semantic_support" not in row
        assert "automated_errors" not in row


def test_grounding_blind_mode_hides_ground_truth():
    records = _pool()
    reviewer, key, _ = build_reviewer_and_key(
        records, _texts(records), target_counts={"T": 2, "M": 3}, seed=1, mode="grounding-blind",
    )
    for row in reviewer:
        assert "ground_truth" not in row
        assert set(row.keys()) == GROUNDING_BLIND_FIELDS


def test_with_gold_mode_shows_ground_truth():
    records = _pool()
    reviewer, key, _ = build_reviewer_and_key(
        records, _texts(records), target_counts={"T": 2, "M": 3}, seed=1, mode="with-gold",
    )
    key_by_id = {k["audit_id"]: k for k in key}
    for row in reviewer:
        assert set(row.keys()) == WITH_GOLD_FIELDS
        assert row["ground_truth"] == key_by_id[row["audit_id"]]["ground_truth"]


def test_invalid_mode_raises():
    import pytest
    records = _pool()
    with pytest.raises(ValueError):
        build_reviewer_and_key(records, _texts(records), target_counts={"T": 1}, seed=1, mode="bogus")


# --- key mapping ---

def test_key_preserves_mapping_to_real_case_arm():
    records = _pool()
    reviewer, key, _ = build_reviewer_and_key(
        records, _texts(records), target_counts={"T": 2, "M": 3}, seed=1,
    )
    real_triples = {(r.case_id, r.arm, r.target) for r in records}
    for row in key:
        assert (row["case_id"], row["arm"], row["target"]) in real_triples
    assert len(key) == len(reviewer)
    assert {k["audit_id"] for k in key} == {r["audit_id"] for r in reviewer}


def test_arm_pattern_is_target_aware_regardless_of_input_order():
    # Regression: arm_pattern used to be computed via a case_id-only grouping,
    # so a case with records for MULTIPLE targets (T and M here) could have
    # its M-row arm_pattern silently contaminated by the T-row's abstention
    # status, depending on which target's record appeared last in `records`.
    # Same case_id, same two arms, OPPOSITE abstention pattern on T vs M.
    shared_case = "shared_case_1"
    t_c = _rec(shared_case, "C_constrained", "T", abstained=True, gt="T2", pred="T2")
    t_d = _rec(shared_case, "D_grounded", "T", abstained=False, gt="T2", pred="T2")
    m_c = _rec(shared_case, "C_constrained", "M", abstained=False)
    m_d = _rec(shared_case, "D_grounded", "M", abstained=True)

    for records in ([t_c, t_d, m_c, m_d], [m_c, m_d, t_c, t_d], [t_c, m_c, t_d, m_d]):
        texts = {shared_case: "Report text " + "x" * 10}
        _, key, _ = build_reviewer_and_key(records, texts, target_counts={"M": 1}, seed=1)
        m_key_row = next(k for k in key if k["target"] == "M" and k["arm"] == "C_constrained")
        assert m_key_row["arm_pattern"] == "priority_predicts_comparison_abstains", records


# --- reproducibility ---

def test_build_reviewer_and_key_is_reproducible():
    records = _pool()
    texts = _texts(records)
    reviewer1, key1, report1 = build_reviewer_and_key(records, texts, target_counts={"T": 2, "M": 3}, seed=5)
    reviewer2, key2, report2 = build_reviewer_and_key(records, texts, target_counts={"T": 2, "M": 3}, seed=5)
    assert reviewer1 == reviewer2
    assert key1 == key2
    assert report1 == report2


# --- report excerpt windowing ---

def test_short_report_not_truncated():
    excerpt, truncated = _report_excerpt("short text", None)
    assert excerpt == "short text"
    assert truncated is False


def test_long_report_windows_around_evidence():
    filler = "x" * 5000
    evidence = "UNIQUE_EVIDENCE_PHRASE"
    text = filler + evidence + filler
    excerpt, truncated = _report_excerpt(text, evidence, max_chars=4000, window=200)
    assert evidence in excerpt
    assert truncated is True
    assert len(excerpt) < len(text)


def test_long_report_no_evidence_shows_full_text():
    # No evidence to window around (abstained / evidence absent) -- the
    # reviewer needs the WHOLE report for the source-sufficiency judgment,
    # so this must NOT truncate to the head, especially for M-stage/abstained
    # cases where there's no evidence by construction.
    text = "y" * 9000
    excerpt, truncated = _report_excerpt(text, None, max_chars=4000)
    assert truncated is False
    assert excerpt == text


def test_long_report_evidence_not_locatable_shows_full_text():
    text = "y" * 9000
    excerpt, truncated = _report_excerpt(text, "phrase not present anywhere", max_chars=4000)
    assert truncated is False
    assert excerpt == text


# --- abstention handling ---

def test_abstained_record_gets_null_evidence_fields_but_full_schema():
    records = [
        _rec("c1", "C_constrained", "M", abstained=False),
        _rec("c1", "D_grounded", "M", abstained=True),
    ]
    reviewer, key, _ = build_reviewer_and_key(records, _texts(records), target_counts={"M": 2}, seed=1)
    abstained_rows = [r for r, k in zip(reviewer, key) if k["prediction"] == "unknown"]
    assert len(abstained_rows) == 1
    row = abstained_rows[0]
    assert row["evidence"] is None
    # schema still complete -- reviewer can still judge source sufficiency
    assert "human_source_sufficient_for_stage" in row
    assert row["human_source_sufficient_for_stage"] is None


# --- write_audit_bundle ---

def test_write_audit_bundle_creates_expected_files(tmp_path):
    records = _pool()
    reviewer, key, sampling_report = build_reviewer_and_key(
        records, _texts(records), target_counts={"T": 2, "M": 3}, seed=1,
    )
    config = {"seed": 1, "mode": "grounding-blind", "sampling_report": sampling_report}
    write_audit_bundle(reviewer, key, config, tmp_path / "audit_001")

    outdir = tmp_path / "audit_001"
    assert (outdir / "reviewer.jsonl").exists()
    assert (outdir / "key.jsonl").exists()
    assert (outdir / "config.json").exists()
    instructions = (outdir / "INSTRUCTIONS.md").read_text()
    assert "human_semantic_support" in instructions
    assert "grounding-blind" in instructions


# --- legacy path still works (smoke test; full coverage stays in test_audit.py) ---

def test_legacy_build_audit_sheet_and_score_audit_unaffected():
    records = [_rec("c1", "D_grounded", "T", gt="T2", pred="T2")]
    rows = build_audit_sheet(records, _texts(records), n=1, seed=1)
    assert rows[0]["automated_evidence_span_found"] is True
    rows[0]["human_supported"] = True
    rows[0]["human_prediction_correct"] = True
    result = score_audit(rows)
    assert result["percent_agreement_supported"] == 1.0


# --- CLI smoke test ---

def test_cli_build_then_score(tmp_path):
    records = _pool()
    predictions_dir = tmp_path / "exp_001"
    predictions_dir.mkdir()
    predictions_path = predictions_dir / "predictions.jsonl"
    to_jsonl(records, predictions_path)

    dataset_df = pd.DataFrame([
        {"patient_filename": case_id, "text": text} for case_id, text in _texts(records).items()
    ])
    dataset_path = tmp_path / "dataset.parquet"
    dataset_df.to_parquet(dataset_path, index=False)

    output_dir = tmp_path / "audit" / "audit_001"
    main([
        "build",
        "--predictions", str(predictions_path),
        "--dataset", str(dataset_path),
        "--target-count", "T=2", "M=3",
        "--seed", "1",
        "--output", str(output_dir),
    ])

    assert (output_dir / "reviewer.jsonl").exists()
    assert (output_dir / "key.jsonl").exists()
    assert (output_dir / "INSTRUCTIONS.md").exists()
    config = json.loads((output_dir / "config.json").read_text())
    for key in (
        "seed", "target_counts", "mode", "source_experiment_id",
        "source_prediction_file", "timestamp", "achieved_composition",
    ):
        assert key in config
    assert config["source_experiment_id"] == "exp_001"

    # score an entirely unfilled bundle -- must not crash, everything NaN/0
    main(["score", "--audit-dir", str(output_dir)])
    scored = json.loads((output_dir / "scored.json").read_text())
    assert scored["agreement"]["overall"]["evidence_span_found"]["n"] == 0
    assert scored["source_sufficiency"]["overall"]["n_reviewed"] > 0
