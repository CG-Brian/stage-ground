from stageground.evaluation.comparison_report import (
    format_comparison_line,
    format_p_value,
    generate_comparison_report,
)


def test_format_p_value_zero_becomes_less_than_threshold():
    assert format_p_value(0.0, 5000) == "< 0.0002"


def test_format_p_value_nonzero_formatted_normally():
    assert format_p_value(0.042, 5000) == "0.0420"


def test_format_p_value_none_is_na():
    assert format_p_value(None, 5000) == "n/a"


def _cmp(diff, ci_low, ci_high, p_value, n_boot=5000, arm_a="D_grounded", arm_b="C_constrained"):
    return {
        "metric_name": "accuracy", "arm_a": arm_a, "arm_b": arm_b,
        "diff": diff, "ci_low": ci_low, "ci_high": ci_high, "p_value": p_value,
        "n_boot": n_boot, "seed": 42, "n_paired_cases": 100,
    }


def test_format_comparison_line_marks_significant_when_ci_excludes_zero():
    line = format_comparison_line(_cmp(-0.07, -0.09, -0.05, 0.0))
    assert "**" in line
    assert "< 0.0002" in line
    assert "D_grounded" in line and "C_constrained" in line


def test_format_comparison_line_no_marker_when_ci_crosses_zero():
    line = format_comparison_line(_cmp(0.01, -0.02, 0.04, 0.4))
    assert "**" not in line


def test_generate_comparison_report_includes_all_present_scopes_and_metrics():
    bootstrap_json = {
        "overall": {"accuracy": [_cmp(-0.07, -0.09, -0.05, 0.0)]},
        "T": {"accuracy": [_cmp(-0.01, -0.03, 0.01, 0.4)]},
        "N": {},
        "M": {},
    }
    report = generate_comparison_report(bootstrap_json)
    assert "## overall" in report
    assert "## T" in report
    assert "## N" in report
    assert "Accuracy" in report
    assert "no comparisons available" in report  # N and M are empty


def test_generate_comparison_report_handles_missing_scope_gracefully():
    report = generate_comparison_report({"overall": {}})
    assert "## overall" in report
    assert "## M" in report  # still rendered, just noted as empty
