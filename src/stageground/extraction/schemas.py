"""Pydantic v2 output schema — required from ALL arms (DESIGN §4.4).

The only thing that varies across arms is constraint strength, not the
output shape, to avoid a format confound (DESIGN §4.4 fairness note).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

Confidence = Literal["high", "medium", "low"]

# value domains after canonicalization (DESIGN §4.2)
TValue = Literal["T0", "T1", "T2", "T3", "T4", "TX", "unknown"]
NValue = Literal["N0", "N1", "N2", "N3", "NX", "unknown"]
MValue = Literal["M0", "M1", "MX", "unknown"]


class Field(BaseModel):
    value: str  # constrained to the value domain for T/N/M at validation time
    evidence: str | None  # verbatim span copied from report text, or null
    confidence: Confidence
    reason: str | None = None  # required when value == "unknown"

    @model_validator(mode="after")
    def _reason_required_when_unknown(self) -> "Field":
        if self.value == "unknown" and not self.reason:
            raise ValueError("reason is required when value == 'unknown'")
        return self


class TField(Field):
    value: TValue


class NField(Field):
    value: NValue


class MField(Field):
    value:MValue


class Extraction(BaseModel):
    T_stage: TField
    N_stage: NField
    M_stage: MField
    pathologic_stage: Field | None = None  # overall AJCC group stage


# --- Lenient ingestion schema (DESIGN §4.1) ---
# Parsing checks SHAPE only; the value domain is enforced downstream by
# canonicalize() so that e.g. 'pT3' -> 'T3' is scored, not discarded.
# invalid-output rate (§6.1) = JSON/shape failures, not allowed-value misses.


class RawField(BaseModel):
    value: str | None  # null = free-form abstention; normalized to "unknown" downstream
    evidence: str | None
    confidence: Confidence
    reason: str | None = None


class RawExtraction(BaseModel):
    T_stage: RawField
    N_stage: RawField
    M_stage: RawField
    pathologic_stage: RawField | None = None
