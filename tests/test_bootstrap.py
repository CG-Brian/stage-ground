import json

from stageground.evaluation.bootstrap import bootstrap_ci, paired_bootstrap_compare


def test_bootstrap_ci_constant_values():
    result = bootstrap_ci([1.0] * 50, seed=1, n_boot=500)
    assert result.point_estimate == 1.0
    assert result.ci_low == 1.0
    assert result.ci_high == 1.0


def test_bootstrap_ci_sane_interval_for_mixed_values():
    values = [0.0, 1.0] * 50
    result = bootstrap_ci(values, seed=2, n_boot=1000)
    assert result.point_estimate == 0.5
    assert 0.3 < result.ci_low < 0.5
    assert 0.5 < result.ci_high < 0.7


def test_bootstrap_ci_degenerate_single_value_no_crash():
    result = bootstrap_ci([1.0], seed=3, n_boot=200)
    assert result.point_estimate == 1.0
    assert result.ci_low == 1.0
    assert result.ci_high == 1.0


def test_bootstrap_ci_empty_values_no_crash():
    result = bootstrap_ci([], seed=4, n_boot=200)
    import math
    assert math.isnan(result.point_estimate)


def test_bootstrap_ci_drops_nans():
    import math
    values = [1.0, 1.0, float("nan"), 1.0]
    result = bootstrap_ci(values, seed=5, n_boot=200)
    assert result.point_estimate == 1.0
    assert not math.isnan(result.point_estimate)


def test_paired_bootstrap_identical_values_diff_near_zero():
    values = {f"c{i}": 1.0 for i in range(30)}
    result = paired_bootstrap_compare(
        values, values, metric_name="accuracy", arm_a="D_grounded", arm_b="D_grounded",
        seed=6, n_boot=500,
    )
    assert result.diff == 0.0
    assert result.ci_low <= 0.0 <= result.ci_high


def test_paired_bootstrap_detects_large_consistent_difference():
    values_a = {f"c{i}": 1.0 for i in range(40)}
    values_b = {f"c{i}": 0.0 for i in range(40)}
    result = paired_bootstrap_compare(
        values_a, values_b, metric_name="accuracy", arm_a="D_grounded", arm_b="C_constrained",
        seed=7, n_boot=500,
    )
    assert result.diff == 1.0
    assert result.ci_low > 0.0
    assert result.p_value == 0.0
    assert result.n_paired_cases == 40


def test_paired_bootstrap_intersects_mismatched_case_ids():
    values_a = {"a": 1.0, "b": 1.0, "c": 0.0}
    values_b = {"b": 0.0, "c": 1.0, "d": 1.0}
    result = paired_bootstrap_compare(
        values_a, values_b, metric_name="accuracy", arm_a="X", arm_b="Y", seed=8, n_boot=100,
    )
    assert result.n_paired_cases == 2  # only "b" and "c" are shared


def test_paired_bootstrap_no_shared_cases_no_crash():
    result = paired_bootstrap_compare(
        {"a": 1.0}, {"z": 1.0}, metric_name="accuracy", arm_a="X", arm_b="Y", seed=9, n_boot=100,
    )
    assert result.n_paired_cases == 0
    assert result.p_value is None


def test_results_are_json_serializable():
    ci = bootstrap_ci([0.0, 1.0] * 10, seed=10, n_boot=200)
    paired = paired_bootstrap_compare(
        {"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0},
        metric_name="accuracy", arm_a="X", arm_b="Y", seed=11, n_boot=200,
    )
    json.dumps(ci.to_dict())
    json.dumps(paired.to_dict())
