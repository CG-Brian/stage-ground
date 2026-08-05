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
    return _normalize_for_match(str(evidence)) in _normalize_for_match(text)
