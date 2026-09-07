
from stageground.evaluation import metrics
from stageground.evaluation.records import build_record
from stageground.evaluation.tables import build_comparison_table, write_tables


def _rec(case_id, arm, target, ground_truth, raw_value, evidence, report_text):
    return build_record(
        case_id=case_id, arm=arm, target=target, ground_truth=ground_truth,
        raw_output={"value": raw_value, "evidence": evidence, "confidence": "high", "reason": None},
        report_text=report_text,
    )


def _two_arm_records():
    # Arm X on target T: 4 evaluable, accuracy 0.5, supported_accuracy 0.25 (mirrors test_metrics mixed scenario)
    x = [
        _rec("c1", "X", "T", "T2", "T2", "T2", "Tumor classified as T2 with clear margins."),
        _rec("c2", "X", "T", "T3", "T1", "T1", "Tumor classified as T1, well differentiated."),
        _rec("c3", "X", "T", "T1", "T1", None, "Tumor present, staging pT1 mentioned elsewhere."),
        _rec("c4", "X", "T", "T4", None, None, "No staging information available."),  # abstains
    ]
    # Arm Y on target T: 2 evaluable, both correct and supported -> accuracy 1.0
    y = [
        _rec("c1", "Y", "T", "T2", "T2", "T2", "Tumor classified as T2 with clear margins."),
        _rec("c2", "Y", "T", "T3", "T3", "T3", "Tumor classified as T3."),
    ]
    return x + y


def test_build_comparison_table_per_arm_values():
    records = _two_arm_records()
    table = build_comparison_table(records, target="T")
    by_arm = table.set_index("arm")

    assert by_arm.loc["X", "n_evaluable"] == 4
    assert by_arm.loc["X", "accuracy"] == 0.5
    assert by_arm.loc["X", "supported_accuracy"] == 0.25

    assert by_arm.loc["Y", "n_evaluable"] == 2
    assert by_arm.loc["Y", "accuracy"] == 1.0
    assert by_arm.loc["Y", "supported_accuracy"] == 1.0


def test_build_comparison_table_target_none_matches_pooled_metrics():
    t_rec = _rec("c1", "X", "T", "T2", "T2", "T2", "T2 present.")
    n_rec = _rec("c1", "X", "N", "N0", "N1", None, "N0 present.")
    records = [t_rec, n_rec]

    table = build_comparison_table(records, target=None)
    expected = metrics.compute_all_metrics(records, target=None)

    row = table.set_index("arm").loc["X"]
    assert row["n_evaluable"] == expected["n_evaluable"]
    assert row["accuracy"] == expected["accuracy"]


def test_write_tables_creates_expected_files(tmp_path):
    records = _two_arm_records()
    write_tables(records, tmp_path)
    tables_dir = tmp_path / "tables"
    for stem in ("overall", "T", "N", "M"):
        assert (tables_dir / f"{stem}.csv").exists()
        assert (tables_dir / f"{stem}.md").exists()

    md = (tables_dir / "T.csv").read_text()
    assert "arm" in md
