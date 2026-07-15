"""Track B — groundedness vs report text (the signature of this study, DESIGN §6.2).

Independent of gold labels. Catches unsupported assertions and fabricated
evidence by checking that each `evidence` span appears verbatim in the report.
"""

from __future__ import annotations

from stageground.extraction.schemas import Field


def evidence_grounded(field: Field, report_text: str) -> bool:
    """True iff the field's evidence span is present verbatim in the report.

    Fabricated evidence (DESIGN §6.2) is the most damning failure; it is
    caught here. No evidence offered -> not grounded.
    """
    if field.evidence is None:
        return False
    return field.evidence.lower() in report_text.lower()
