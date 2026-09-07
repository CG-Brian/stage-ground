from stageground.evaluation.normalize import grounded, value_supported_by_text

REPORT = "Primary tumor (pT3) invades the subserosa. No distant metastasis noted."


def test_grounded_absent_evidence_span():
    assert grounded(None, REPORT) is False


def test_grounded_empty_string_evidence():
    assert grounded("", REPORT) is False


def test_value_supported_by_text_finds_matching_token():
    assert value_supported_by_text("T3", "pT3 invades subserosa") is True


def test_value_supported_by_text_no_token_present():
    assert value_supported_by_text("T3", "no staging descriptor here") is False


def test_value_supported_by_text_wrong_stage_letter_no_match():
    assert value_supported_by_text("M1", "pT3 invades subserosa") is False


def test_value_supported_by_text_rejects_unknown_and_invalid():
    assert value_supported_by_text("unknown", "pT3 present") is False
    assert value_supported_by_text("INVALID", "pT3 present") is False


def test_value_supported_by_text_handles_none_text():
    assert value_supported_by_text("T3", None) is False


def test_value_supported_by_text_handles_empty_text():
    assert value_supported_by_text("T3", "") is False
