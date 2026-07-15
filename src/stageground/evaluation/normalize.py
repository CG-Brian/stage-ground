"""Shared scoring helpers used by every track (single source of truth)."""

from __future__ import annotations

from stageground.data.clean_labels import canonicalize

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
    """True iff evidence appears verbatim in the report (DESIGN §6.2)."""
    if evidence is None:
        return False
    return str(evidence).lower() in text.lower()
