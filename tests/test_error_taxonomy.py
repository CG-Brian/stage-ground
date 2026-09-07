from stageground.evaluation.error_taxonomy import classify_errors

REPORT_WITH_M1 = "Distant metastasis present (pM1) to liver."
REPORT_NO_M = "Primary tumor pT3 invades pericolic fat. Nodes 0/12."


def test_correct_and_supported_has_no_errors():
    errors = classify_errors(
        ground_truth="T3",
        prediction="T3",
        evidence="pT3 invades pericolic fat",
        report_text=REPORT_NO_M,
        raw_value="T3",
        schema_valid=True,
    )
    assert errors == []


def test_invalid_schema_output():
    errors = classify_errors(
        ground_truth="T3",
        prediction="unknown",
        evidence=None,
        report_text=REPORT_NO_M,
        raw_value=None,
        schema_valid=False,
    )
    assert errors == ["invalid_schema_output"]


def test_invalid_normalization():
    errors = classify_errors(
        ground_truth="T3",
        prediction="INVALID",
        evidence="garbage",
        report_text=REPORT_NO_M,
        raw_value="T9",
        schema_valid=True,
    )
    assert errors == ["invalid_normalization"]


def test_over_abstention_and_missed_explicit_stage_when_text_has_token():
    errors = classify_errors(
        ground_truth="M1",
        prediction="unknown",
        evidence=None,
        report_text=REPORT_WITH_M1,
        raw_value=None,
        schema_valid=True,
    )
    assert "over_abstention" in errors
    assert "missed_explicit_stage" in errors


def test_abstention_with_no_textual_basis_is_not_flagged():
    # principled abstention: text has no M token at all -> no error flags
    errors = classify_errors(
        ground_truth="M0",
        prediction="unknown",
        evidence=None,
        report_text=REPORT_NO_M,
        raw_value=None,
        schema_valid=True,
    )
    assert errors == []


def test_abstention_with_no_ground_truth_is_not_flagged():
    errors = classify_errors(
        ground_truth=None,
        prediction="unknown",
        evidence=None,
        report_text=REPORT_NO_M,
        raw_value=None,
        schema_valid=True,
    )
    assert errors == []


def test_evidence_span_not_found_when_evidence_missing():
    errors = classify_errors(
        ground_truth="T3",
        prediction="T3",
        evidence=None,
        report_text=REPORT_NO_M,
        raw_value="T3",
        schema_valid=True,
    )
    assert "evidence_span_not_found" in errors
    assert "hallucinated_stage" not in errors  # prediction IS correct vs gold


def test_hallucinated_stage_when_wrong_and_no_evidence():
    # asserted M1 with fabricated evidence, gold is M0 -> multiple flags coexist
    errors = classify_errors(
        ground_truth="M0",
        prediction="M1",
        evidence="widespread metastatic deposits",  # not present in report_text
        report_text=REPORT_NO_M,
        raw_value="M1",
        schema_valid=True,
    )
    assert set(errors) == {"evidence_span_not_found", "hallucinated_stage"}


def test_wrong_stage_with_supporting_evidence():
    # asserted M1, evidence genuinely grounded and supports M1, but gold says M0
    # (candidate registry-discordance case, not fabrication)
    errors = classify_errors(
        ground_truth="M0",
        prediction="M1",
        evidence="pM1",
        report_text=REPORT_WITH_M1,
        raw_value="M1",
        schema_valid=True,
    )
    assert errors == ["wrong_stage_with_supporting_evidence"]


def test_evidence_does_not_support_prediction():
    # evidence text is verbatim-grounded (appears in report) but doesn't
    # actually contain a token supporting the predicted value
    errors = classify_errors(
        ground_truth="T3",
        prediction="T3",
        evidence="Nodes 0/12",  # grounded in report, but no T3 token in it
        report_text=REPORT_NO_M,
        raw_value="T3",
        schema_valid=True,
    )
    assert errors == ["evidence_does_not_support_prediction"]


def test_prediction_correct_but_unsupported_edge_case():
    errors = classify_errors(
        ground_truth="M0",
        prediction="M0",
        evidence=None,
        report_text=REPORT_NO_M,
        raw_value="M0",
        schema_valid=True,
    )
    assert errors == ["evidence_span_not_found"]
