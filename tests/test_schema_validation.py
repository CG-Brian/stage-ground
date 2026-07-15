import pytest
from pydantic import ValidationError

from stageground.extraction.schemas import Extraction, RawExtraction, TField


def _ok_field(value="T2", evidence="pathologic stage T2", reason=None):
    return {"value": value, "evidence": evidence, "confidence": "high", "reason": reason}


def test_valid_extraction():
    e = Extraction(
        T_stage=_ok_field("T2"),
        N_stage=_ok_field("N0", evidence="no nodal involvement"),
        M_stage=_ok_field("M1", evidence="distant metastasis present"),
    )
    assert e.T_stage.value == "T2"


def test_out_of_domain_value_rejected():
    with pytest.raises(ValidationError):
        TField(value="T9", evidence="x", confidence="high")


def test_M2_not_allowed():
    with pytest.raises(ValidationError):
        Extraction(
            T_stage=_ok_field("T2"),
            N_stage=_ok_field("N0"),
            M_stage=_ok_field("M2"),  # M has no M2
        )


def test_unknown_requires_reason():
    with pytest.raises(ValidationError):
        TField(value="unknown", evidence=None, confidence="low")  # no reason


def test_unknown_with_reason_ok():
    f = TField(value="unknown", evidence=None, confidence="low",
               reason="no T descriptor in report")
    assert f.value == "unknown"


def test_evidence_may_be_null():
    f = TField(value="T1", evidence=None, confidence="medium")
    assert f.evidence is None


# --- RawExtraction: lenient ingestion (what the pipeline actually parses) ---


def test_raw_accepts_out_of_domain_value():
    # 'pT3' is not in the allowed set, but shape is fine -> parses (canonicalized later)
    e = RawExtraction(
        T_stage=_ok_field("pT3"),
        N_stage=_ok_field("pN0"),
        M_stage=_ok_field("pMX"),
    )
    assert e.T_stage.value == "pT3"


def test_raw_accepts_null_value_as_abstention():
    # free-form arm emits value: null when it has no basis -> must NOT be invalid
    e = RawExtraction(
        T_stage=_ok_field("T2"),
        N_stage=_ok_field("N0"),
        M_stage={"value": None, "evidence": None, "confidence": "low", "reason": None},
    )
    assert e.M_stage.value is None


def test_raw_rejects_broken_shape():
    with pytest.raises(ValidationError):
        RawExtraction(T_stage=_ok_field("T2"), N_stage=_ok_field("N0"))  # M_stage missing