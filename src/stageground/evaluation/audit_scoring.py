"""Scoring for the blinded reviewer/key audit bundle (spec §12-15).

Joins `reviewer.jsonl` and `key.jsonl` on `audit_id` -- the reviewer never
touches `key.jsonl` directly; this module does the unblinding, for analysis
purposes only, after review is complete (spec §16).
"""

from __future__ import annotations

TARGETS = ("T", "N", "M")


def _join(reviewer_rows: list[dict], key_rows: list[dict]) -> list[dict]:
    key_by_id = {k["audit_id"]: k for k in key_rows}
    joined = []
    for r in reviewer_rows:
        k = key_by_id.get(r["audit_id"])
        if k is None:
            continue
        row = dict(k)
        row.update(r)  # human_* fields layered on top; shared keys (audit_id/target) agree by construction
        joined.append(row)
    return joined


def _confusion_and_agreement(pairs: list[tuple[bool, bool]]) -> dict:
    """`pairs` = [(automated, human), ...]. `automated` is the predicted
    class, `human` is the reference/actual class (spec §13: "is the regex
    heuristic under-detecting?"). Never raises on an empty or zero-variance
    input; unavailable rates are `None`."""
    n = len(pairs)
    if n == 0:
        return {
            "n": 0, "percent_agreement": float("nan"), "cohens_kappa": None,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "precision": None, "recall": None, "specificity": None,
        }

    tp = sum(1 for a, h in pairs if a and h)
    fp = sum(1 for a, h in pairs if a and not h)
    tn = sum(1 for a, h in pairs if not a and not h)
    fn = sum(1 for a, h in pairs if not a and h)
    percent_agreement = (tp + tn) / n

    kappa = None
    autos = [a for a, _ in pairs]
    humans = [h for _, h in pairs]
    if n >= 2 and len(set(autos) | set(humans)) >= 2:
        from sklearn.metrics import cohen_kappa_score

        value = float(cohen_kappa_score(autos, humans))
        kappa = None if value != value else value  # NaN (zero-variance) -> None

    return {
        "n": n,
        "percent_agreement": percent_agreement,
        "cohens_kappa": kappa,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else None,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else None,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else None,
    }


def _pairs_for_field(
    joined_rows: list[dict], *, automated_key: str, human_key: str, target: str | None = None
) -> list[tuple[bool, bool]]:
    pairs = []
    for row in joined_rows:
        if target is not None and row.get("target") != target:
            continue
        a, h = row.get(automated_key), row.get(human_key)
        if a is None or h is None:
            continue
        pairs.append((bool(a), bool(h)))
    return pairs


def score_reviewed_audit(reviewer_rows: list[dict], key_rows: list[dict]) -> dict:
    """Overall + per-target (T/N/M) agreement, Cohen's kappa, and confusion
    matrix for both `evidence_span_found` and `semantic_support` (spec §12-13).
    Rows where either side is unfilled/`None` are excluded from that specific
    comparison, matching the legacy scorer's null-skipping convention."""
    joined = _join(reviewer_rows, key_rows)
    result = {}
    for scope in (None, *TARGETS):
        scope_key = "overall" if scope is None else scope
        span_pairs = _pairs_for_field(
            joined, automated_key="automated_evidence_span_found",
            human_key="human_evidence_span_found", target=scope,
        )
        semantic_pairs = _pairs_for_field(
            joined, automated_key="automated_semantic_support",
            human_key="human_semantic_support", target=scope,
        )
        result[scope_key] = {
            "evidence_span_found": _confusion_and_agreement(span_pairs),
            "semantic_support": _confusion_and_agreement(semantic_pairs),
        }
    return result


def _pct_true(rows: list[dict], field: str) -> tuple[float, int]:
    vals = [row[field] for row in rows if row.get(field) is not None]
    n = len(vals)
    if n == 0:
        return float("nan"), 0
    return sum(1 for v in vals if v) / n, n


def _sufficiency_stats(rows: list[dict]) -> dict:
    sufficient_pct, n_suff = _pct_true(rows, "human_source_sufficient_for_stage")
    explicit_pct, n_expl = _pct_true(rows, "human_source_has_explicit_stage")
    inferential_pct, n_inf = _pct_true(rows, "human_source_has_inferential_evidence")
    return {
        "n_reviewed": len(rows),
        "pct_source_sufficient": sufficient_pct, "n_source_sufficient_scored": n_suff,
        "pct_explicit_stage": explicit_pct, "n_explicit_stage_scored": n_expl,
        "pct_inferential_evidence": inferential_pct, "n_inferential_evidence_scored": n_inf,
    }


def analyze_source_sufficiency(reviewer_rows: list[dict], key_rows: list[dict]) -> dict:
    """Per-target (+ overall) source-sufficiency percentages (spec §14), plus
    an M-stage breakdown by `arm_pattern` (from `key.jsonl`) -- e.g.
    "priority_predicts_comparison_abstains" vs "both_predict" -- central to
    the research question of whether high M accuracy reflects genuine
    extraction or dataset priors."""
    joined = _join(reviewer_rows, key_rows)
    result = {}
    for scope in (None, *TARGETS):
        scope_key = "overall" if scope is None else scope
        subset = joined if scope is None else [r for r in joined if r.get("target") == scope]
        result[scope_key] = _sufficiency_stats(subset)

    m_rows = [r for r in joined if r.get("target") == "M"]
    patterns = sorted({r.get("arm_pattern") for r in m_rows if r.get("arm_pattern") is not None})
    result["M_by_arm_pattern"] = {
        pattern: _sufficiency_stats([r for r in m_rows if r.get("arm_pattern") == pattern])
        for pattern in patterns
    }
    return result


def analyze_m0_prior_prediction(
    reviewer_rows: list[dict], key_rows: list[dict], *, priority_arm: str = "C_constrained"
) -> dict:
    """Among `priority_arm` M-target predictions of "M0": accuracy,
    source-sufficiency rate, semantic-support rate, and explicit-M-token
    rate. Field names are deliberately neutral (e.g.
    `pct_explicit_m_token_present`, not "prior_guessing_rate") -- spec §15:
    do not label high-accuracy-with-low-sufficiency cases "prior guessing"
    automatically; that interpretation is for a human to draw from these
    numbers, not for this function to assert."""
    joined = _join(reviewer_rows, key_rows)
    subset = [
        r for r in joined
        if r.get("arm") == priority_arm and r.get("target") == "M" and r.get("prediction") == "M0"
    ]
    n = len(subset)
    if n == 0:
        return {
            "n": 0, "accuracy": float("nan"),
            "pct_source_sufficient": float("nan"), "n_source_sufficient_scored": 0,
            "pct_semantic_support": float("nan"), "n_semantic_support_scored": 0,
            "pct_explicit_m_token_present": float("nan"), "n_explicit_m_token_scored": 0,
        }

    correct = sum(1 for r in subset if r.get("ground_truth") == "M0")
    sufficient_pct, n_suff = _pct_true(subset, "human_source_sufficient_for_stage")
    semantic_pct, n_sem = _pct_true(subset, "human_semantic_support")
    explicit_pct, n_expl = _pct_true(subset, "human_source_has_explicit_stage")
    return {
        "n": n,
        "accuracy": correct / n,
        "pct_source_sufficient": sufficient_pct, "n_source_sufficient_scored": n_suff,
        "pct_semantic_support": semantic_pct, "n_semantic_support_scored": n_sem,
        "pct_explicit_m_token_present": explicit_pct, "n_explicit_m_token_scored": n_expl,
    }
