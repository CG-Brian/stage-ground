"""Tests for the target-aware semantic-support heuristic, built directly from
the 80-case blinded audit's observed failure modes (spec items 8-9).
"""

from stageground.evaluation.semantic_support import (
    CATEGORY_DISTANT_METASTASIS_NEGATIVE,
    CATEGORY_DISTANT_METASTASIS_POSITIVE,
    CATEGORY_EXPLICIT_STAGE_TOKEN,
    CATEGORY_LOCAL_INVASION,
    CATEGORY_REGIONAL_NODE_NEGATIVE,
    CATEGORY_REGIONAL_NODE_POSITIVE,
    CATEGORY_TUMOR_SIZE,
    CATEGORY_UNKNOWN,
    classify_evidence_categories,
    evidence_semantically_supports_prediction as supports,
)


# --- N0 (regional-node-negative) recall failures ---

def test_n0_supported_by_zero_of_seven_count():
    assert supports("N0", "No evidence of malignancy in seven lymph nodes examined (0/7).", "N") is True


def test_n0_supported_by_all_nodes_negative_phrasing():
    assert supports("N0", "All fourteen regional lymph nodes are negative for metastatic carcinoma.", "N") is True


def test_n0_supported_by_zero_of_x_positive_phrasing():
    assert supports("N0", "0 of 7 lymph nodes positive.", "N") is True


def test_n0_supported_by_no_tumor_seen():
    assert supports("N0", "No tumor seen in seven lymph nodes.", "N") is True


def test_n0_supported_by_negative_for_metastatic_carcinoma():
    assert supports("N0", "Lymph nodes negative for metastatic carcinoma.", "N") is True


# --- N-positive (regional-node-positive) recall failures ---

def test_n1_supported_by_explicit_count():
    assert supports("N1", "1/3 lymph nodes positive.", "N") is True


def test_n1_supported_by_word_count_and_metastatic_language():
    assert supports("N1", "Metastatic carcinoma in one of three sentinel lymph nodes.", "N") is True


def test_n1_supported_by_micrometastatic_carcinoma():
    assert supports("N1", "Micrometastatic carcinoma in sentinel lymph node.", "N") is True


def test_n1_supported_by_regional_count():
    assert supports("N1", "2 of 6 regional lymph nodes involved.", "N") is True


def test_n2_descriptive_only_stays_conservative():
    # "multiple regional lymph nodes" has no explicit count and no explicit
    # N2 token -- node-count-to-stage cutoffs are cancer-type-dependent, so
    # this must NOT be claimed as N2 support (spec: "only mark true if
    # current granularity rules can safely support N2; otherwise conservative").
    assert supports("N2", "Metastases identified in multiple regional lymph nodes.", "N") is False


def test_n2_explicit_token_still_supported():
    # Layer 1 (explicit token) is unaffected by the new descriptive layer.
    assert supports("N2", "pN2 with capsular extension.", "N") is True


# --- target separation: regional nodes must never support M ---

def test_regional_node_metastasis_does_not_support_m1():
    assert supports("M1", "Metastatic carcinoma in one axillary lymph node.", "M") is False


def test_regional_node_positive_does_not_leak_into_m_categories():
    categories = classify_evidence_categories("Metastatic carcinoma in one axillary lymph node.", "M1")
    assert CATEGORY_REGIONAL_NODE_POSITIVE in categories
    assert CATEGORY_DISTANT_METASTASIS_POSITIVE not in categories


# --- T-stage local invasion ---

def test_t3_supported_by_muscularis_and_perivesical_invasion():
    assert supports("T3", "Tumor invades through muscularis propria into perivesical adipose tissue.", "T") is True


def test_t2_supported_by_generic_local_invasion_bucket():
    # Descriptive invasion language validates the T2/T3/T4 GROUP, not a
    # specific value -- see module docstring "no cancer-type numeric thresholds".
    assert supports("T2", "Tumor invades muscularis propria.", "T") is True


def test_t1_not_supported_by_deep_invasion_language():
    # T0/T1 are the most conservative predictions; invasion language
    # contradicts, not supports, them.
    assert supports("T1", "Tumor invades muscularis propria.", "T") is False


def test_t_extrathyroidal_extension_supports_t3():
    assert supports("T3", "Extrathyroidal extension identified.", "T") is True


def test_t_renal_sinus_invasion_supports_t3():
    assert supports("T3", "Renal sinus invasion present.", "T") is True


def test_t_visceral_pleural_invasion_supports_t2():
    assert supports("T2", "Visceral pleural invasion identified.", "T") is True


def test_t_serosal_penetration_supports_t4():
    assert supports("T4", "Tumor penetrates serosa.", "T") is True


def test_t_tumor_size_alone_does_not_support_any_specific_t_value():
    # Explicit spec constraint: no "tumor size > X -> T2" style rule.
    assert supports("T2", "Tumor size 3.1 cm.", "T") is False
    assert supports("T1", "Tumor size 3.1 cm.", "T") is False
    categories = classify_evidence_categories("Tumor size 3.1 cm.", "T2")
    assert CATEGORY_TUMOR_SIZE in categories


def test_t_negated_invasion_language_not_supported():
    assert supports("T2", "No invasion of muscularis propria identified.", "T") is False


# --- irrelevant evidence for M ---

def test_m0_not_supported_by_clear_margins():
    assert supports("M0", "Margins free of tumor.", "M") is False


def test_m0_not_supported_by_absence_of_lvi_mention():
    assert supports("M0", "No lymphovascular invasion identified.", "M") is False


# --- explicit tokens (layer 1) still work for all targets ---

def test_explicit_token_layer_unchanged_for_t():
    assert supports("T3", "pT3 invades pericolic fat", "T") is True


def test_explicit_token_layer_unchanged_for_n():
    assert supports("N1", "pN1", "N") is True


def test_explicit_token_layer_unchanged_for_m():
    assert supports("M1", "Distant metastasis present (pM1) to liver.", "M") is True


# --- abstention / invalid predictions never supported ---

def test_unknown_prediction_never_supported():
    assert supports("unknown", "0/7 lymph nodes positive.", "N") is False


def test_invalid_prediction_never_supported():
    assert supports("INVALID", "0/7 lymph nodes positive.", "N") is False


def test_no_evidence_never_supported():
    assert supports("N0", None, "N") is False
    assert supports("N0", "", "N") is False


# ============================================================================
# Precision regression tests (spec item 9): these must stay False so recall
# improvements elsewhere don't introduce semantic false positives.
# ============================================================================

def test_regional_node_metastasis_is_not_distant_metastasis():
    assert supports("M1", "Regional lymph node metastasis identified.", "M") is False


def test_negative_surgical_margins_is_not_m0():
    assert supports("M0", "Surgical margins negative for tumor.", "M") is False


def test_recurrence_risk_is_not_m0():
    assert supports("M0", "High risk of recurrence based on tumor grade.", "M") is False


def test_absence_of_mention_is_not_evidence_of_absence():
    # No mention of metastasis at all -- must not be silently treated as "no metastasis".
    assert supports("M0", "Tumor size 4.2 cm, grade 3.", "M") is False


def test_her2_receptor_status_is_not_tnm_evidence():
    assert supports("T2", "HER2 positive, ER/PR negative.", "T") is False
    assert supports("N0", "HER2 positive, ER/PR negative.", "N") is False
    assert supports("M0", "HER2 positive, ER/PR negative.", "M") is False


# --- classify_evidence_categories sanity ---

def test_classify_evidence_categories_unknown_bucket_for_irrelevant_text():
    categories = classify_evidence_categories("HER2 positive, ER/PR negative.", "T2")
    assert categories == {CATEGORY_UNKNOWN}


def test_classify_evidence_categories_explicit_token_detected():
    categories = classify_evidence_categories("pN2 with capsular extension.", "N2")
    assert CATEGORY_EXPLICIT_STAGE_TOKEN in categories


def test_classify_evidence_categories_node_negative_detected():
    categories = classify_evidence_categories("0/7 lymph nodes positive.", "N0")
    assert categories == {CATEGORY_REGIONAL_NODE_NEGATIVE}


def test_classify_evidence_categories_distant_met_negative_detected():
    categories = classify_evidence_categories("No evidence of distant metastasis.", "M0")
    assert CATEGORY_DISTANT_METASTASIS_NEGATIVE in categories
    assert CATEGORY_DISTANT_METASTASIS_POSITIVE not in categories


def test_classify_evidence_categories_local_invasion_detected():
    categories = classify_evidence_categories("Tumor invades muscularis propria.", "T2")
    assert CATEGORY_LOCAL_INVASION in categories
