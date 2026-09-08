"""Target-aware semantic-support heuristic for `evidence_semantically_supports_prediction`.

## Why this module exists

Before this module, `evidence_semantically_supports_prediction` was computed
by a single call to `stageground.evaluation.normalize.value_supported_by_text`
-- a pure regex search for a LITERAL stage token (e.g. `T2`, `pN1`, `M0`)
inside the evidence string. That check is still layer 1 here (unchanged,
still used exactly as before) because a provisional 80-case blinded audit
(`audit/audit_001`) found it already strong across all three targets
(span-found agreement 94-97%, kappa 0.82-0.94).

The SAME audit found the overall semantic-support judgment much weaker
(precision 1.0, recall ~0.53), heavily concentrated in N-stage (recall
~0.29) and, to a lesser extent, T-stage (recall ~0.56) -- M-stage was
already strong (recall ~0.88). The failure mode was exactly what a
literal-token-only check predicts: pathology reports routinely describe
staging-relevant findings in prose ("0/14 lymph nodes involved", "tumor
invades muscularis propria") without ever writing the literal stage token,
and the old heuristic had no way to recognize that language at all.

## What this module adds

A second layer of TARGET-AWARE, conservative, regex-based evidence
categorization (spec: "explicit_stage_token / regional_node_negative /
regional_node_positive / local_invasion / tumor_size /
distant_metastasis_positive / distant_metastasis_negative / unknown").
`classify_evidence_categories` tags a piece of evidence text with zero or
more of these; `evidence_semantically_supports_prediction` applies
target-specific rules on top (see the per-target functions below) and stays
the same public boolean signature as before -- serialized
`PredictionRecord`s and all downstream metrics are unaffected.

## Design principles (why the rules stop where they stop)

- **Precision over recall.** Every new rule was chosen because it is
  difficult to satisfy by accident (a negation check runs before any
  "positive" classification; node/invasion signals are only accepted within
  a small character window of the anchoring keyword, not anywhere in the
  string, to avoid conflating unrelated clauses in a multi-topic sentence).
  When a rule's safety couldn't be established generically (see next point),
  it does not fire, and the function returns `False`.
- **No cancer-type-specific numeric thresholds.** AJCC's T/N cutoffs for
  "how much invasion/how many nodes" differ by cancer type (a tumor size or
  node count that means T2/N1 in one cancer can mean something else in
  another). So: (a) T-stage descriptive evidence ("invades muscularis
  propria", "extrathyroidal extension", etc.) only validates that a
  prediction is *plausibly non-superficial* (T2/T3/T4 as a group) -- it
  never tries to pick which one of those three is "correct"; a `T0`/`T1`
  prediction is never validated by invasion language, since that would be
  self-contradictory. (b) Explicit tumor-size mentions are recognized as a
  category (for documentation/introspection) but never independently
  justify a specific T value. (c) Descriptive node evidence supports `N0`
  (all-negative) and `N1` (some positive nodes, unspecified count) but NOT
  `N2`/`N3` -- distinguishing those requires a count-to-stage cutoff that is
  cancer-type-dependent, so only an explicit `N2`/`N3` token (layer 1)
  counts.
- **Regional nodes are never distant metastasis.** `target="M"` never
  consults the node-status categorizer at all -- only
  `distant_metastasis_positive`/`_negative` (which require the literal
  phrase "distant metastas-") can support an M prediction. A phrase like
  "metastatic carcinoma in an axillary lymph node" describes regional
  nodal disease, not `M1`, regardless of how confidently the node-status
  categorizer reads it.

## Known limitations (see also README)

- This is still a regex heuristic, not clinical reasoning: it cannot verify
  that a report's descriptive language actually maps to the SPECIFIC
  predicted T/N stage the way a pathologist would using cancer-type-specific
  AJCC criteria. It only asks "is this evidence plausibly the right *kind*
  of finding for this target and this broad prediction," which is a
  materially weaker claim than "this prediction is correct."
- OCR-variant literal tokens (e.g. "pNo" for "pN0", "N l" for "N1") are
  deliberately NOT normalized here -- the spec calls this out as unsafe
  ("only when the normalization is sufficiently safe") and the audit found
  the existing literal-token layer already strong, so it is left untouched.
- N2/N3 and T-specific-value support still require an explicit token; this
  is a known, intentional recall ceiling (see design principles above), not
  an oversight.
"""

from __future__ import annotations

import re

from stageground.evaluation.normalize import value_supported_by_text

CATEGORY_EXPLICIT_STAGE_TOKEN = "explicit_stage_token"
CATEGORY_REGIONAL_NODE_NEGATIVE = "regional_node_negative"
CATEGORY_REGIONAL_NODE_POSITIVE = "regional_node_positive"
CATEGORY_LOCAL_INVASION = "local_invasion"
CATEGORY_TUMOR_SIZE = "tumor_size"
CATEGORY_DISTANT_METASTASIS_POSITIVE = "distant_metastasis_positive"
CATEGORY_DISTANT_METASTASIS_NEGATIVE = "distant_metastasis_negative"
CATEGORY_UNKNOWN = "unknown"

_WINDOW = 50  # chars of context searched around an anchor keyword

# --- regional lymph nodes (N) ---

_LYMPH_NODE_RE = re.compile(r"lymph\s*nodes?|sentinel\s*(?:lymph\s*)?nodes?", re.IGNORECASE)
_NODE_COUNT_RE = re.compile(r"\b(\d+)\s*(?:/|out of|of)\s*(\d+)\b", re.IGNORECASE)
_NEGATIVE_WORD_RE = re.compile(
    r"\b(?:no|none|negative|free of|without evidence of|not\s+identified)\b", re.IGNORECASE
)
_POSITIVE_NODE_WORD_RE = re.compile(r"metastat|positive|involve", re.IGNORECASE)


def _classify_node_status(text: str) -> str | None:
    """'negative', 'positive', or None (no lymph-node signal detected).
    Count-based evidence (e.g. "0/7", "2 of 6") takes priority over keyword
    matching -- it is the least ambiguous signal available."""
    match = _LYMPH_NODE_RE.search(text)
    if not match:
        return None
    start = max(0, match.start() - _WINDOW)
    end = min(len(text), match.end() + _WINDOW)
    window_text = text[start:end]

    count = _NODE_COUNT_RE.search(window_text)
    if count:
        return "negative" if int(count.group(1)) == 0 else "positive"
    if _NEGATIVE_WORD_RE.search(window_text):
        return "negative"
    if _POSITIVE_NODE_WORD_RE.search(window_text):
        return "positive"
    return None


# --- local invasion (T) ---

_INVASION_VERB_RE = re.compile(
    r"\b(?:invad(?:e[sd]?|ing)|invasion|extend(?:s|ed|ing)?|extension|penetrat(?:e[sd]?|ing|ion))\b",
    re.IGNORECASE,
)
_DEEP_STRUCTURE_RE = re.compile(
    r"muscularis\s+propria|perivesical|pericolonic|pericolic|serosa|"
    r"adipose|diaphragm\w*|parenchyma|pleura\w*|renal\s+sinus|"
    r"extrathyroidal|capsul\w*|adjacent\s+(?:organ|structure)",
    re.IGNORECASE,
)
_NEGATION_BEFORE_RE = re.compile(r"\b(?:no|not|without|free of|absent|negative for)\b", re.IGNORECASE)
_PRE_NEGATION_WINDOW = 20


def _has_local_invasion(text: str) -> bool:
    verb = _INVASION_VERB_RE.search(text)
    if not verb:
        return False
    pre_start = max(0, verb.start() - _PRE_NEGATION_WINDOW)
    if _NEGATION_BEFORE_RE.search(text[pre_start:verb.start()]):
        return False  # e.g. "no invasion of muscularis propria identified"
    start = max(0, verb.start() - _WINDOW)
    end = min(len(text), verb.end() + _WINDOW)
    return bool(_DEEP_STRUCTURE_RE.search(text[start:end]))


_TUMOR_SIZE_RE = re.compile(r"\btumor\s+size\b|\b\d+(?:\.\d+)?\s*cm\b", re.IGNORECASE)


# --- distant metastasis (M) ---

_DISTANT_MET_RE = re.compile(r"distant\s+metastas\w*", re.IGNORECASE)
_PRE_MET_NEGATION_WINDOW = 30


def _classify_distant_metastasis(text: str) -> str | None:
    match = _DISTANT_MET_RE.search(text)
    if not match:
        return None
    pre_start = max(0, match.start() - _PRE_MET_NEGATION_WINDOW)
    if _NEGATIVE_WORD_RE.search(text[pre_start:match.start()]):
        return "negative"
    return "positive"


def classify_evidence_categories(evidence: str | None, prediction: str) -> set[str]:
    """Best-effort tags describing what KIND of pathology evidence `evidence`
    contains. Documentation/introspection aid: the support decision itself
    lives in `evidence_semantically_supports_prediction`, not here."""
    if not evidence:
        return set()

    categories: set[str] = set()
    if value_supported_by_text(prediction, evidence):
        categories.add(CATEGORY_EXPLICIT_STAGE_TOKEN)

    node_status = _classify_node_status(evidence)
    if node_status == "negative":
        categories.add(CATEGORY_REGIONAL_NODE_NEGATIVE)
    elif node_status == "positive":
        categories.add(CATEGORY_REGIONAL_NODE_POSITIVE)

    if _has_local_invasion(evidence):
        categories.add(CATEGORY_LOCAL_INVASION)

    if _TUMOR_SIZE_RE.search(evidence):
        categories.add(CATEGORY_TUMOR_SIZE)

    met_status = _classify_distant_metastasis(evidence)
    if met_status == "negative":
        categories.add(CATEGORY_DISTANT_METASTASIS_NEGATIVE)
    elif met_status == "positive":
        categories.add(CATEGORY_DISTANT_METASTASIS_POSITIVE)

    if not categories:
        categories.add(CATEGORY_UNKNOWN)
    return categories


def evidence_semantically_supports_prediction(prediction: str, evidence: str | None, target: str) -> bool:
    """Target-aware replacement for the old literal-token-only check. See
    module docstring for the full rationale. Returns `False` on any
    ambiguous or unrecognized evidence -- this function is deliberately
    biased toward under-claiming support, never over-claiming it."""
    if not evidence or prediction in ("unknown", "INVALID"):
        return False

    categories = classify_evidence_categories(evidence, prediction)
    if CATEGORY_EXPLICIT_STAGE_TOKEN in categories:
        return True

    if target == "N":
        if prediction == "N0":
            return CATEGORY_REGIONAL_NODE_NEGATIVE in categories
        if prediction == "N1":
            return CATEGORY_REGIONAL_NODE_POSITIVE in categories
        return False  # N2/N3 granularity requires an explicit token (layer 1)

    if target == "T":
        if prediction in ("T2", "T3", "T4"):
            return CATEGORY_LOCAL_INVASION in categories
        return False  # T0/T1 are never validated by invasion language

    if target == "M":
        if prediction == "M0":
            return CATEGORY_DISTANT_METASTASIS_NEGATIVE in categories
        if prediction == "M1":
            return CATEGORY_DISTANT_METASTASIS_POSITIVE in categories
        return False

    return False
