from stageground.evaluation.audit_sampling import (
    BOTH_ABSTAIN,
    BOTH_PREDICT,
    PATTERN_UNKNOWN,
    PRIORITY_ABSTAINS_COMPARISON_PREDICTS,
    PRIORITY_PREDICTS_COMPARISON_ABSTAINS,
    classify_arm_pattern,
    group_by_case,
    stratified_audit_sample,
)
from stageground.evaluation.records import PredictionRecord


def _rec(case_id, arm, target, *, abstained, semantic=None):
    return PredictionRecord(
        case_id=case_id, arm=arm, target=target,
        ground_truth="M0", prediction="unknown" if abstained else "M1",
        evidence=None if abstained else "M1 present",
        correct=False, abstained=abstained,
        evidence_span_found=None if abstained else True,
        evidence_semantically_supports_prediction=None if abstained else semantic,
        errors=[], raw_model_output={},
    )


# --- classify_arm_pattern ---

def test_both_predict():
    case = {"C_constrained": _rec("c1", "C_constrained", "M", abstained=False),
            "D_grounded": _rec("c1", "D_grounded", "M", abstained=False)}
    assert classify_arm_pattern(case, priority_arm="C_constrained", comparison_arm="D_grounded") == BOTH_PREDICT


def test_priority_predicts_comparison_abstains():
    case = {"C_constrained": _rec("c1", "C_constrained", "M", abstained=False),
            "D_grounded": _rec("c1", "D_grounded", "M", abstained=True)}
    assert classify_arm_pattern(case, priority_arm="C_constrained", comparison_arm="D_grounded") == PRIORITY_PREDICTS_COMPARISON_ABSTAINS


def test_priority_abstains_comparison_predicts():
    case = {"C_constrained": _rec("c1", "C_constrained", "M", abstained=True),
            "D_grounded": _rec("c1", "D_grounded", "M", abstained=False)}
    assert classify_arm_pattern(case, priority_arm="C_constrained", comparison_arm="D_grounded") == PRIORITY_ABSTAINS_COMPARISON_PREDICTS


def test_both_abstain():
    case = {"C_constrained": _rec("c1", "C_constrained", "M", abstained=True),
            "D_grounded": _rec("c1", "D_grounded", "M", abstained=True)}
    assert classify_arm_pattern(case, priority_arm="C_constrained", comparison_arm="D_grounded") == BOTH_ABSTAIN


def test_pattern_unknown_when_arm_missing():
    case = {"C_constrained": _rec("c1", "C_constrained", "M", abstained=False)}
    assert classify_arm_pattern(case, priority_arm="C_constrained", comparison_arm="D_grounded") == PATTERN_UNKNOWN


# --- group_by_case is target-aware (regression: it used to key by case_id
# alone, letting one target's record silently overwrite another's for the
# same arm, so the result depended on input list order) ---

def test_group_by_case_keys_by_case_id_and_target():
    t_rec = _rec("c1", "C_constrained", "T", abstained=True)
    m_rec = _rec("c1", "C_constrained", "M", abstained=False)
    grouped = group_by_case([t_rec, m_rec])
    assert grouped[("c1", "T")]["C_constrained"] is t_rec
    assert grouped[("c1", "M")]["C_constrained"] is m_rec


def test_classify_arm_pattern_not_contaminated_by_other_targets_regardless_of_order():
    # Same case_id, same two arms, but T and M have OPPOSITE abstention
    # patterns -- if grouping ever collapses across targets, whichever
    # target's record was inserted last would silently win.
    t_c = _rec("c1", "C_constrained", "T", abstained=True)    # T: C abstains
    t_d = _rec("c1", "D_grounded", "T", abstained=False)      # T: D predicts
    m_c = _rec("c1", "C_constrained", "M", abstained=False)   # M: C predicts
    m_d = _rec("c1", "D_grounded", "M", abstained=True)       # M: D abstains

    orderings = [
        [t_c, t_d, m_c, m_d],
        [m_c, m_d, t_c, t_d],
        [t_c, m_c, t_d, m_d],
        [m_d, t_d, m_c, t_c],
    ]
    for records in orderings:
        grouped = group_by_case(records)
        m_pattern = classify_arm_pattern(
            grouped[("c1", "M")], priority_arm="C_constrained", comparison_arm="D_grounded"
        )
        t_pattern = classify_arm_pattern(
            grouped[("c1", "T")], priority_arm="C_constrained", comparison_arm="D_grounded"
        )
        assert m_pattern == PRIORITY_PREDICTS_COMPARISON_ABSTAINS, records
        assert t_pattern == PRIORITY_ABSTAINS_COMPARISON_PREDICTS, records


# --- stratified_audit_sample ---

def _synthetic_pool(n_t=20, n_n=20, n_m_priority=8, n_m_true=6, n_m_false=6, n_m_both=10):
    records = []
    for i in range(n_t):
        records.append(_rec(f"t{i}", "C_constrained", "T", abstained=False))
    for i in range(n_n):
        records.append(_rec(f"n{i}", "C_constrained", "N", abstained=False))
    # M: priority pattern (C predicts, D abstains)
    for i in range(n_m_priority):
        records.append(_rec(f"mp{i}", "C_constrained", "M", abstained=False, semantic=(i % 2 == 0)))
        records.append(_rec(f"mp{i}", "D_grounded", "M", abstained=True))
    # M: both predict, semantic=True pool (non-priority)
    for i in range(n_m_true):
        records.append(_rec(f"mt{i}", "C_constrained", "M", abstained=False, semantic=True))
        records.append(_rec(f"mt{i}", "D_grounded", "M", abstained=False, semantic=True))
    # M: both predict, semantic=False pool (non-priority)
    for i in range(n_m_false):
        records.append(_rec(f"mf{i}", "C_constrained", "M", abstained=False, semantic=False))
        records.append(_rec(f"mf{i}", "D_grounded", "M", abstained=False, semantic=False))
    # M: both abstain (filler)
    for i in range(n_m_both):
        records.append(_rec(f"mb{i}", "C_constrained", "M", abstained=True))
        records.append(_rec(f"mb{i}", "D_grounded", "M", abstained=True))
    return records


def test_determinism_same_seed_same_selection():
    pool = _synthetic_pool()
    sel1, report1 = stratified_audit_sample(pool, target_counts={"T": 5, "N": 5, "M": 10}, seed=42)
    sel2, report2 = stratified_audit_sample(pool, target_counts={"T": 5, "N": 5, "M": 10}, seed=42)
    triples1 = [(r.case_id, r.arm, r.target) for r in sel1]
    triples2 = [(r.case_id, r.arm, r.target) for r in sel2]
    assert triples1 == triples2
    assert report1 == report2


def test_target_counts_respected():
    pool = _synthetic_pool()
    selected, report = stratified_audit_sample(pool, target_counts={"T": 3, "N": 0, "M": 5}, seed=1)
    assert sum(1 for r in selected if r.target == "T") == 3
    assert sum(1 for r in selected if r.target == "N") == 0
    assert sum(1 for r in selected if r.target == "M") == 5
    assert report["N"]["selected"] == 0


def test_m_priority_tier_is_actually_prioritized():
    pool = _synthetic_pool(n_m_priority=4, n_m_true=1, n_m_false=1, n_m_both=1)
    selected, report = stratified_audit_sample(
        pool, target_counts={"M": 4}, seed=7, m_priority_fraction=0.5,
    )
    m_selected = [r for r in selected if r.target == "M"]
    priority_case_ids = {f"mp{i}" for i in range(4)}
    n_from_priority = sum(
        1 for r in m_selected
        if r.arm == "C_constrained" and r.case_id in priority_case_ids
    )
    assert n_from_priority >= 2  # ceil(4 * 0.5) = 2
    assert report["M"]["priority_tier_selected"] >= 2


def test_m_semantic_balance_includes_both_classes():
    pool = _synthetic_pool(n_m_priority=0, n_m_true=5, n_m_false=5, n_m_both=0)
    selected, report = stratified_audit_sample(
        pool, target_counts={"M": 6}, seed=3, m_priority_fraction=0.0, m_semantic_balance_fraction=0.5,
    )
    semantics = {r.evidence_semantically_supports_prediction for r in selected if r.target == "M"}
    assert True in semantics
    assert False in semantics
    assert report["M"]["semantic_balance_tier_selected"] > 0


def test_oversized_request_does_not_crash():
    pool = _synthetic_pool(n_t=2, n_n=0, n_m_priority=0, n_m_true=0, n_m_false=0, n_m_both=1)
    selected, report = stratified_audit_sample(pool, target_counts={"T": 100, "N": 5, "M": 100}, seed=9)
    assert len(selected) <= 100 + 5 + 100
    assert report["T"]["selected"] == 2
    assert report["N"]["selected"] == 0
