from stageground.evaluation.track_b import evidence_grounded
from stageground.extraction.schemas import TField

REPORT = "Primary tumor (pT3) invades the subserosa. No distant metastasis noted."


def _field(evidence, value="T3"):
    return TField(value=value, evidence=evidence, confidence="high")


def test_grounded_when_evidence_in_text():
    assert evidence_grounded(_field("pT3"), REPORT) is True


def test_grounded_is_case_insensitive():
    assert evidence_grounded(_field("PRIMARY TUMOR"), REPORT) is True


def test_fabricated_evidence_not_grounded():
    # model wrote a span that is absent from the report -> the damning failure
    assert evidence_grounded(_field("pT4b carcinoma"), REPORT) is False


def test_null_evidence_not_grounded():
    f = TField(value="unknown", evidence=None, confidence="low", reason="no basis")
    assert evidence_grounded(f, REPORT) is False


def test_substring_span_counts():
    # a partial verbatim span still counts as present
    assert evidence_grounded(_field("subserosa"), REPORT) is True
