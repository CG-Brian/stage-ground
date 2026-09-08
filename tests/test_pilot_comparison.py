import pytest

from stageground.evaluation.pilot_comparison import (
    build_pilot_vs_main_table,
    check_replication_claims,
    render_pilot_vs_main_markdown,
)


def _overall(accuracy, semantic_unsupported, semantic_supported_accuracy, span_unsupported=0.1,
             coverage=0.9, abstention=0.1):
    return {
        "accuracy": accuracy,
        "coverage": coverage,
        "abstention_rate": abstention,
        "span_unsupported_rate_over_evaluable": span_unsupported,
        "semantic_unsupported_rate_over_evaluable": semantic_unsupported,
        "semantic_supported_accuracy_over_evaluable": semantic_supported_accuracy,
    }


# Modeled loosely on the real 200-case pilot's overall numbers.
PILOT_OVERALL = {
    "A_zero_shot": _overall(0.675, 0.588, 0.163),
    "C_constrained": _overall(0.697, 0.527, 0.187),
    "C_plus_unknown": _overall(0.637, 0.432, 0.177),
    "D_grounded": _overall(0.628, 0.412, 0.210),
}


def test_build_pilot_vs_main_table_computes_diffs():
    main = {"C_constrained": _overall(0.72, 0.50, 0.20)}
    pilot = {"C_constrained": _overall(0.697, 0.527, 0.187)}
    rows = build_pilot_vs_main_table(pilot, main)
    by_metric = {r["metric"]: r for r in rows if r["arm"] == "C_constrained"}
    assert by_metric["accuracy"]["pilot_200"] == 0.697
    assert by_metric["accuracy"]["main_1000"] == 0.72
    assert by_metric["accuracy"]["diff"] == pytest.approx(0.72 - 0.697)


def test_build_pilot_vs_main_table_only_includes_arms_in_both():
    pilot = {"A_zero_shot": _overall(0.5, 0.5, 0.1)}
    main = {"C_constrained": _overall(0.6, 0.4, 0.2)}
    rows = build_pilot_vs_main_table(pilot, main)
    assert rows == []


def test_c_relatively_poor_grounding_compares_against_c_plus_and_d_not_a():
    # A (free-form) is WORSE than C here -- the claim must still hold, since
    # it's about C being worse than C+/D specifically, not the single worst arm.
    metrics = {
        "A_zero_shot": _overall(0.6, 0.70, 0.1),  # worse than C, irrelevant to the claim
        "C_constrained": _overall(0.7, 0.55, 0.15),
        "C_plus_unknown": _overall(0.6, 0.45, 0.18),
        "D_grounded": _overall(0.6, 0.40, 0.20),
    }
    results = check_replication_claims(metrics, metrics)
    assert results["C_relatively_poor_grounding"]["pilot_200"]["holds"] is True


def test_c_relatively_poor_grounding_false_when_c_actually_matches_c_plus():
    metrics = {
        "C_constrained": _overall(0.7, 0.40, 0.15),  # same rate as C+ -- C is NOT worse
        "C_plus_unknown": _overall(0.6, 0.40, 0.18),
        "D_grounded": _overall(0.6, 0.35, 0.20),
    }
    results = check_replication_claims(metrics, metrics)
    assert results["C_relatively_poor_grounding"]["pilot_200"]["holds"] is False


def test_replication_claims_all_hold_on_realistic_pilot_data():
    # main = same pattern as pilot, slightly different numbers (replicates)
    main_overall = {
        "A_zero_shot": _overall(0.66, 0.60, 0.17),
        "C_constrained": _overall(0.70, 0.55, 0.19),
        "C_plus_unknown": _overall(0.64, 0.45, 0.18),
        "D_grounded": _overall(0.63, 0.40, 0.22),
    }
    pilot_m = {
        "A_zero_shot": {"accuracy": 0.42, "semantic_supported_accuracy_over_evaluable": 0.005},
        "C_constrained": {"accuracy": 0.455, "semantic_supported_accuracy_over_evaluable": 0.005},
        "C_plus_unknown": {"accuracy": 0.255, "semantic_supported_accuracy_over_evaluable": 0.005},
        "D_grounded": {"accuracy": 0.25, "semantic_supported_accuracy_over_evaluable": 0.01},
    }
    results = check_replication_claims(PILOT_OVERALL, main_overall, pilot_m=pilot_m, main_m=pilot_m)

    assert results["C_highest_or_near_highest_accuracy"]["pilot_200"]["holds"] is True
    assert results["C_relatively_poor_grounding"]["pilot_200"]["holds"] is True
    assert results["C_plus_lower_unsupported_than_C"]["pilot_200"]["holds"] is True
    assert results["D_strongest_grounded_reliability"]["pilot_200"]["holds"] is True
    assert results["M_accuracy_much_higher_than_semantic_supported_accuracy"]["pilot_200"]["holds"] is True


def test_replication_claim_reported_false_when_it_genuinely_fails():
    # main where D is NOT the strongest grounded arm -- must report holds=False, not force it
    main_overall = {
        "C_constrained": _overall(0.70, 0.55, 0.30),  # C now has the HIGHEST semantic supported accuracy
        "D_grounded": _overall(0.63, 0.40, 0.22),
    }
    results = check_replication_claims(PILOT_OVERALL, main_overall)
    assert results["D_strongest_grounded_reliability"]["main_1000"]["holds"] is False
    assert results["D_strongest_grounded_reliability"]["main_1000"]["best_arm"] == "C_constrained"


def test_replication_claims_missing_arm_skips_gracefully():
    partial = {"A_zero_shot": _overall(0.5, 0.5, 0.1)}  # no C_constrained at all
    results = check_replication_claims(PILOT_OVERALL, partial)
    # main_1000 entries requiring C_constrained should simply be absent, not crash
    assert "main_1000" not in results.get("C_highest_or_near_highest_accuracy", {})


def test_render_pilot_vs_main_markdown_smoke():
    rows = build_pilot_vs_main_table(PILOT_OVERALL, PILOT_OVERALL)
    claims = check_replication_claims(PILOT_OVERALL, PILOT_OVERALL)
    markdown = render_pilot_vs_main_markdown(
        rows, claims, pilot_id="pilot_200", main_id="main_1000"
    )
    assert "# Pilot" in markdown
    assert "C_constrained" in markdown
    assert "HOLDS" in markdown or "does NOT hold" in markdown
