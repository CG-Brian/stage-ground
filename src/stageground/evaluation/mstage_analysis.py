"""M-stage evidence categorization (spec §5).

M-stage shows the most interesting accuracy/grounding tradeoff in the pilot
(low coverage, M0-dominated, often requires imaging the pathology report
can't contain). This module classifies *why* M-stage evidence looks the way
it does in the report text, to help separate real extraction from dataset
priors / guessing / inferring information the source doesn't contain.

Deliberately regex-only (spec: "do not over-engineer a classifier
initially") -- a reproducible annotation format, not an NLU model. Ambiguous
or borderline cases are meant to be resolved by a human via
`build_mstage_annotation_sheet`'s blank `human_category` field, not by
sharpening the regex indefinitely.
"""

from __future__ import annotations

import re

from stageground.evaluation.records import PredictionRecord

M_CATEGORY_EXPLICIT_TOKEN = "explicit_m_token"
M_CATEGORY_METASTATIC_DESCRIBED = "metastatic_described_no_token"
M_CATEGORY_NO_EVIDENCE = "no_m_evidence"
M_CATEGORY_AMBIGUOUS = "ambiguous_insufficient"

_M_TOKEN_RE = re.compile(r"\bp?M[01X][A-Za-z0-9]*\b", re.IGNORECASE)
_METASTATIC_RE = re.compile(r"\bmetasta(?:sis|tic|ses)\b", re.IGNORECASE)
_HEDGE_RE = re.compile(
    r"\b(?:cannot be (?:assessed|determined)|indeterminate|rule out|"
    r"suspicious for|possible metasta\w*|pending(?: further)?|equivocal)\b",
    re.IGNORECASE,
)

EXCERPT_WINDOW = 150  # chars of context on each side of the first match
EXCERPT_FALLBACK_LEN = 1000


def classify_m_evidence(report_text: str | None) -> str:
    """One of the four M_CATEGORY_* constants, from regex signals alone:

    - explicit_m_token: an M-stage token (pM0/M1/MX/...) is present, with no
      conflicting hedge language.
    - metastatic_described_no_token: 'metasta(sis|tic|ses)' language is
      present but no explicit M-stage token.
    - ambiguous_insufficient: BOTH an explicit token and hedged/uncertain
      metastatic language are present (e.g. an M0 token alongside "suspicious
      for possible metastatic disease, further imaging pending") -- flagged
      for manual review rather than trusted automatically.
    - no_m_evidence: neither signal present.
    """
    if not report_text:
        return M_CATEGORY_NO_EVIDENCE

    has_token = bool(_M_TOKEN_RE.search(report_text))
    has_metastatic_language = bool(_METASTATIC_RE.search(report_text))
    has_hedge = bool(_HEDGE_RE.search(report_text))

    if has_token and has_metastatic_language and has_hedge:
        return M_CATEGORY_AMBIGUOUS
    if has_token:
        return M_CATEGORY_EXPLICIT_TOKEN
    if has_metastatic_language:
        return M_CATEGORY_METASTATIC_DESCRIBED
    return M_CATEGORY_NO_EVIDENCE


def _excerpt(report_text: str) -> str:
    match = _M_TOKEN_RE.search(report_text) or _METASTATIC_RE.search(report_text)
    if match is None:
        return report_text[:EXCERPT_FALLBACK_LEN]
    start = max(0, match.start() - EXCERPT_WINDOW)
    end = min(len(report_text), match.end() + EXCERPT_WINDOW)
    prefix = "...\n" if start > 0 else ""
    suffix = "\n..." if end < len(report_text) else ""
    return prefix + report_text[start:end] + suffix


def build_mstage_annotation_sheet(
    records: list[PredictionRecord], texts: dict[str, str], *, target: str = "M"
) -> list[dict]:
    """One row per `target`-target record: reproducible annotation format for
    a manual audit of M-stage extraction, per spec §5. Blank `human_category`
    / `notes` fields are left for a reviewer to fill in."""
    rows = []
    for rec in records:
        if rec.target != target:
            continue
        text = texts.get(rec.case_id, "")
        rows.append({
            "case_id": rec.case_id,
            "arm": rec.arm,
            "report_excerpt": _excerpt(text),
            "automated_category": classify_m_evidence(text),
            "prediction": rec.prediction,
            "ground_truth": rec.ground_truth,
            "evidence_span_found": rec.evidence_span_found,
            "evidence_semantically_supports_prediction": rec.evidence_semantically_supports_prediction,
            "human_category": "",
            "notes": "",
        })
    return rows
