"""Human-readable primary-comparison report from a `bootstrap.json` (spec §9-10).

Formats each paired-bootstrap comparison as effect size + 95% CI first,
p-value second -- and never prints a literal `p = 0.0000` when zero
bootstrap resamples crossed zero (that's a statement about resampling
resolution, not a true zero probability); it prints `p < 1/n_boot` instead.
"""

from __future__ import annotations

SCOPES = ["overall", "T", "N", "M"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "semantic_supported_accuracy_over_evaluable": "Semantic Supported Accuracy",
    "span_unsupported_rate_over_evaluable": "Span Unsupported Rate",
    "semantic_unsupported_rate_over_evaluable": "Semantic Unsupported Rate",
    "abstention_rate": "Abstention Rate",
    "coverage": "Coverage",
}


def format_p_value(p_value: float | None, n_boot: int) -> str:
    """Never returns a literal '0.0000' for an exact-zero empirical p --
    that means no bootstrap resample crossed zero, not that the true
    probability is exactly zero."""
    if p_value is None:
        return "n/a"
    if p_value == 0.0:
        return f"< {1 / n_boot:.4f}"
    return f"{p_value:.4f}"


def format_comparison_line(cmp: dict) -> str:
    """`cmp` is one `PairedBootstrapResult.to_dict()` entry."""
    ci_low, ci_high = cmp["ci_low"], cmp["ci_high"]
    crosses_zero = ci_low <= 0.0 <= ci_high
    marker = "" if crosses_zero else " **"
    p_str = format_p_value(cmp["p_value"], cmp["n_boot"])
    return (
        f"{cmp['arm_a']} − {cmp['arm_b']} = {cmp['diff']:+.4f} "
        f"[95% CI {ci_low:+.4f}, {ci_high:+.4f}], p {p_str}{marker}"
    )


def generate_comparison_report(bootstrap_json: dict) -> str:
    """`bootstrap_json` is the parsed contents of an experiment's
    `bootstrap.json` (produced by `stageground.evaluate.run_evaluation`)."""
    lines = [
        "# Primary comparisons: paired bootstrap (95% CI)",
        "",
        "Each line: `arm_a - arm_b = diff [95% CI]`. `**` marks intervals that "
        "do not cross zero. p-values are empirical (fraction of bootstrap "
        "resamples on the opposite side of zero, doubled); `p < 1/n_boot` means "
        "not a single resample crossed zero, not that the true p-value is "
        "exactly zero. Effect size and CI are the primary evidence here, not "
        "the p-value alone.",
    ]
    for scope in SCOPES:
        scope_data = bootstrap_json.get(scope, {})
        lines.append(f"\n## {scope}\n")
        any_comparison = False
        for metric_name, label in METRIC_LABELS.items():
            comparisons = scope_data.get(metric_name, [])
            if not comparisons:
                continue
            any_comparison = True
            lines.append(f"**{label}**\n")
            for cmp in comparisons:
                lines.append(f"- {format_comparison_line(cmp)}")
            lines.append("")
        if not any_comparison:
            lines.append("(no comparisons available -- fewer than two of the three fixed-comparison arms were run)\n")
    return "\n".join(lines)
