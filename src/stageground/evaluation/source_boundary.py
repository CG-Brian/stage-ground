"""Track C — source-boundary agreement (DESIGN §6.2b).

Crosses accuracy (vs gold) with text-support (vs report) to explain *why* a
prediction disagrees with the registry gold. The guardrail: only a principled
*abstention* against an unsupportable gold is a source-boundary success; an
asserted wrong value with no evidence is an error, not a virtue.
"""

from __future__ import annotations

import re

from stageground.data.clean_labels import canonicalize

# ordered set of Track-C tags
TAGS = [
    "correct",
    "source_boundary_abstention",  # ✓ wrong by accuracy, right by clinical reasoning
    "over_abstention",             # abstained though the text supported the gold
    "unsupported_error",           # asserted a wrong value with no evidence
    "report_gold_discordance",     # asserted a value the text supports; gold differs
]

# stage token as it may appear in the report, e.g. 'pT3a', 'MX', 'N0'
_TOKEN_RE = {s: re.compile(rf"\bp?{s}[0-4X][a-z0-9]*", re.IGNORECASE) for s in "TNM"}


def text_supports_gold(gold: str, text: str) -> bool:
    """True iff some stage token in the report canonicalizes to the gold value."""
    letter = gold[0]
    for tok in _TOKEN_RE[letter].findall(text):
        try:
            if canonicalize(tok) == gold:
                return True
        except ValueError:
            continue
    return False


def tag_case(gold: str, pred: str, pred_grounded: bool, text: str) -> str:
    """Assign one Track-C tag. `pred` is already normalized (canonical | 'unknown' | 'INVALID').

    `gold` is a real registry value (never 'unknown'); callers skip rows with no gold.
    """
    if pred == gold:
        return "correct"
    if pred == "unknown":
        # abstained against a real gold value: principled only if text can't support gold
        return "source_boundary_abstention" if not text_supports_gold(gold, text) else "over_abstention"
    # asserted a different value
    return "report_gold_discordance" if pred_grounded else "unsupported_error"


def source_boundary_success_rate(tags: list[str]) -> float:
    """Principled abstentions ÷ all gold-mismatch cases (DESIGN §6.2b headline)."""
    mismatches = [t for t in tags if t != "correct"]
    if not mismatches:
        return float("nan")
    return sum(t == "source_boundary_abstention" for t in mismatches) / len(mismatches)
