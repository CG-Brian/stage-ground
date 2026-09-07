"""Shared scoring helpers used by every track (single source of truth)."""

from __future__ import annotations

import re

from stageground.data.clean_labels import canonicalize

# TCGA reports are OCR'd PDFs: line-wraps often insert a stray period/comma
# mid-phrase ("under. investigation", "cannot be. assessed"), and these are
# visually indistinguishable from real sentence breaks in this corpus (both
# are "lowercase . whitespace lowercase"). A strict verbatim substring check
# treats the noise as "evidence not found" even when the model quoted the
# passage correctly (confirmed by manual audit, results/audit/). Dropping
# periods/commas entirely before matching fixes this without allowing
# word-level fabrication to pass -- the evidence still needs every word,
# in order, to actually appear in the text.


def _normalize_for_match(s: str) -> str:
    s = re.sub(r"[.,]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()

# non-abstention stage tokens
ALLOWED = {
    "T0", "T1", "T2", "T3", "T4", "TX",
    "N0", "N1", "N2", "N3", "NX",
    "M0", "M1", "MX",
}

# full value domain (DESIGN §4.2): 'unknown' IS a legal value (an abstention),
# so a raw 'unknown' must count as format-compliant, not a violation.
DOMAIN = ALLOWED | {"unknown"}


def norm_pred(value) -> str:
    """Normalize a raw model value to the scoring domain (DESIGN §4.1, §4.3).

    None / 'unknown' -> 'unknown' (abstention).
    Otherwise canonicalize ('pT3' -> 'T3'); if uncanonicalizable, 'INVALID'.
    """
    if value is None or str(value).strip().lower() == "unknown":
        return "unknown"
    try:
        return canonicalize(str(value))
    except ValueError:
        return "INVALID"


def grounded(evidence, text: str) -> bool:
    """True iff evidence appears verbatim in the report (DESIGN §6.2), modulo
    OCR line-wrap noise (see _normalize_for_match)."""
    if evidence is None:
        return False
    normalized = _normalize_for_match(str(evidence))
    if not normalized:
        return False  # empty/whitespace-only evidence conveys no basis
    return normalized in _normalize_for_match(text)


# Stage token as it may appear in report/evidence text, e.g. 'pT3a', 'MX', 'N0'.
# Generalizes stageground.evaluation.source_boundary's gold-only token regex to
# any canonical value, so it can check whether a *predicted* value (not just
# gold) is textually supported -- used by the error taxonomy and by the
# evidence_semantic_support_rate metric.
_STAGE_TOKEN_RE = {
    "T": re.compile(r"\bp?T[0-4X][A-Za-z0-9]*", re.IGNORECASE),
    "N": re.compile(r"\bp?N[0-3X][A-Za-z0-9]*", re.IGNORECASE),
    "M": re.compile(r"\bp?M[01X][A-Za-z0-9]*", re.IGNORECASE),
}


def value_supported_by_text(value: str, text: str | None) -> bool:
    """True iff some stage token in `text` canonicalizes to `value`.

    This is a syntactic proxy (regex token match), not true semantic
    understanding -- e.g. narrative prose implying a stage without an
    explicit token will not be detected. Used for:
      - checking whether an abstained-on ground truth was explicitly
        findable in the report (over_abstention / missed_explicit_stage), and
      - checking whether a prediction's *evidence span* actually contains a
        token supporting that prediction (evidence_does_not_support_prediction),
        i.e. `value_supported_by_text(prediction, evidence)`.
    Returns False for 'unknown' / 'INVALID' values or missing text, never raises.
    """
    if not text or value in ("unknown", "INVALID"):
        return False
    pattern = _STAGE_TOKEN_RE.get(value[0])
    if pattern is None:
        return False
    for tok in pattern.findall(text):
        try:
            if canonicalize(tok) == value:
                return True
        except ValueError:
            continue
    return False
