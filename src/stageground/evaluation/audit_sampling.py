"""Stratified-by-target audit sampling with M-stage oversampling (spec §6-7).

Pure functions over `list[PredictionRecord]` -- no I/O, no LLM calls, fully
unit-testable. `stratified_audit_sample` never hardcodes target counts or
sample composition; callers (the `audit build` CLI) always pass them in
explicitly.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict

from stageground.evaluation.records import PredictionRecord

BOTH_PREDICT = "both_predict"
PRIORITY_PREDICTS_COMPARISON_ABSTAINS = "priority_predicts_comparison_abstains"
PRIORITY_ABSTAINS_COMPARISON_PREDICTS = "priority_abstains_comparison_predicts"
BOTH_ABSTAIN = "both_abstain"
PATTERN_UNKNOWN = "pattern_unknown"  # one or both of the two arms absent for this case


def classify_arm_pattern(
    case_records: dict[str, PredictionRecord], *, priority_arm: str, comparison_arm: str
) -> str:
    """Classify one case's (priority_arm, comparison_arm) abstention pattern.

    `case_records` is `{arm: PredictionRecord}` for a single case_id/target.
    Returns PATTERN_UNKNOWN if either arm isn't present (e.g. the source
    predictions.jsonl didn't include that arm)."""
    if priority_arm not in case_records or comparison_arm not in case_records:
        return PATTERN_UNKNOWN
    p_abstained = case_records[priority_arm].abstained
    c_abstained = case_records[comparison_arm].abstained
    if not p_abstained and not c_abstained:
        return BOTH_PREDICT
    if not p_abstained and c_abstained:
        return PRIORITY_PREDICTS_COMPARISON_ABSTAINS
    if p_abstained and not c_abstained:
        return PRIORITY_ABSTAINS_COMPARISON_PREDICTS
    return BOTH_ABSTAIN


def group_by_case(records: list[PredictionRecord]) -> dict[tuple[str, str], dict[str, PredictionRecord]]:
    """Groups records by `(case_id, target) -> {arm: record}`.

    Keyed by `(case_id, target)`, NOT `case_id` alone: a single case_id has
    one PredictionRecord per (arm, target) pair (e.g. T/N/M), so grouping by
    case_id alone would let one target's record silently overwrite another's
    for the same arm -- which record "wins" would then depend on the input
    list's iteration order, corrupting `classify_arm_pattern` for callers
    that pass in a multi-target record list."""
    by_case: dict[tuple[str, str], dict[str, PredictionRecord]] = defaultdict(dict)
    for r in records:
        by_case[(r.case_id, r.target)][r.arm] = r
    return by_case


def _take(
    pool: list[PredictionRecord], k: int, rng: random.Random
) -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    """Deterministically draw min(k, len(pool)) items from `pool` (identity-based,
    since PredictionRecord isn't hashable by value). Returns (picked, remaining)."""
    k = min(max(k, 0), len(pool))
    if k <= 0:
        return [], list(pool)
    idx = sorted(rng.sample(range(len(pool)), k))
    picked = [pool[i] for i in idx]
    picked_ids = {id(p) for p in picked}
    remaining = [p for p in pool if id(p) not in picked_ids]
    return picked, remaining


def _sample_m(
    m_records: list[PredictionRecord],
    quota: int,
    rng: random.Random,
    priority_arm: str,
    comparison_arm: str,
    priority_fraction: float,
    balance_fraction: float,
) -> tuple[list[PredictionRecord], dict]:
    """M-stage oversampling (spec §7): prioritize `priority_arm` predictions on
    cases where `comparison_arm` abstained (the most diagnostic category for
    "genuine evidence vs. prior-driven guessing"), then ensure both
    automated-semantic-support classes are represented, then fill the rest
    uniformly at random. Never crashes on an under-sized tier -- it just
    yields fewer rows, reported in the returned tier breakdown."""
    if quota <= 0:
        return [], {"priority_tier_selected": 0, "semantic_balance_tier_selected": 0, "random_fill_selected": 0}

    by_case = group_by_case(m_records)
    priority_pool = [
        r for r in m_records
        if r.arm == priority_arm
        and classify_arm_pattern(by_case[(r.case_id, r.target)], priority_arm=priority_arm, comparison_arm=comparison_arm)
        == PRIORITY_PREDICTS_COMPARISON_ABSTAINS
    ]
    priority_take = math.ceil(quota * priority_fraction)
    priority_selected, _ = _take(priority_pool, priority_take, rng)
    selected_ids = {id(r) for r in priority_selected}

    remaining_quota = quota - len(priority_selected)
    balance_selected: list[PredictionRecord] = []
    if remaining_quota > 0:
        pool_after_priority = [r for r in m_records if id(r) not in selected_ids]
        false_pool = [r for r in pool_after_priority if r.evidence_semantically_supports_prediction is False]
        true_pool = [r for r in pool_after_priority if r.evidence_semantically_supports_prediction is True]
        balance_take_each = math.floor(remaining_quota * balance_fraction)

        false_selected, _ = _take(false_pool, balance_take_each, rng)
        balance_selected.extend(false_selected)
        selected_ids.update(id(r) for r in false_selected)

        true_pool = [r for r in true_pool if id(r) not in selected_ids]
        true_selected, _ = _take(true_pool, balance_take_each, rng)
        balance_selected.extend(true_selected)
        selected_ids.update(id(r) for r in true_selected)

    fill_pool = [r for r in m_records if id(r) not in selected_ids]
    remaining_quota_final = quota - len(priority_selected) - len(balance_selected)
    fill_selected, _ = _take(fill_pool, remaining_quota_final, rng)

    all_selected = priority_selected + balance_selected + fill_selected
    tier_report = {
        "priority_tier_selected": len(priority_selected),
        "semantic_balance_tier_selected": len(balance_selected),
        "random_fill_selected": len(fill_selected),
    }
    return all_selected, tier_report


def stratified_audit_sample(
    records: list[PredictionRecord],
    *,
    target_counts: dict[str, int],
    seed: int,
    m_priority_arm: str = "C_constrained",
    m_comparison_arm: str = "D_grounded",
    m_priority_fraction: float = 0.5,
    m_semantic_balance_fraction: float = 0.3,
) -> tuple[list[PredictionRecord], dict]:
    """Deterministic (seeded), target-stratified sample, with M-stage
    oversampling per `_sample_m`. `target_counts` is e.g. `{"T": 20, "N": 20,
    "M": 40}` -- never hardcoded, always caller-supplied. Returns
    `(selected_records, sampling_report)`; `sampling_report` documents the
    ACHIEVED composition per target (and M's tiers) for `config.json`."""
    rng = random.Random(seed)
    selected: list[PredictionRecord] = []
    report: dict = {}

    for target, count in target_counts.items():
        target_records = [r for r in records if r.target == target]
        if target == "M":
            picked, tier_report = _sample_m(
                target_records, count, rng, m_priority_arm, m_comparison_arm,
                m_priority_fraction, m_semantic_balance_fraction,
            )
            report[target] = {"requested": count, "selected": len(picked), **tier_report}
        else:
            picked, _ = _take(target_records, count, rng)
            report[target] = {"requested": count, "selected": len(picked)}
        selected.extend(picked)

    return selected, report
