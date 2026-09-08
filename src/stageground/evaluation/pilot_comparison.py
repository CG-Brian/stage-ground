"""Compare a 'pilot' experiment's metrics against a 'main' experiment's, to
check whether the pilot's QUALITATIVE findings replicate at scale (spec item
16) -- this is explicitly NOT hypothesis testing between sample sizes. If a
claim doesn't hold in one or both experiments, this module reports that
plainly; it never forces a narrative.
"""

from __future__ import annotations

METRIC_KEYS = [
    "accuracy",
    "coverage",
    "abstention_rate",
    "span_unsupported_rate_over_evaluable",
    "semantic_unsupported_rate_over_evaluable",
    "semantic_supported_accuracy_over_evaluable",
]

_NEAR_HIGHEST_MARGIN = 0.02  # accuracy points; "near-highest" tolerance for claim 1


def build_pilot_vs_main_table(pilot_metrics: dict, main_metrics: dict) -> list[dict]:
    """`pilot_metrics`/`main_metrics` are `{arm: {...}}` dicts -- the parsed
    contents of an experiment's (pooled, overall) `metrics.json`. Returns one
    row per (arm, metric) present in BOTH experiments."""
    rows = []
    for arm in sorted(set(pilot_metrics) & set(main_metrics)):
        for key in METRIC_KEYS:
            pilot_val = pilot_metrics[arm].get(key)
            main_val = main_metrics[arm].get(key)
            both_numeric = isinstance(pilot_val, (int, float)) and isinstance(main_val, (int, float))
            diff = (main_val - pilot_val) if both_numeric else None
            rows.append({"arm": arm, "metric": key, "pilot_200": pilot_val, "main_1000": main_val, "diff": diff})
    return rows


def _best_arm(metrics: dict, key: str, *, highest: bool = True) -> str | None:
    candidates = {arm: m.get(key) for arm, m in metrics.items() if isinstance(m.get(key), (int, float))}
    if not candidates:
        return None
    return max(candidates, key=candidates.get) if highest else min(candidates, key=candidates.get)


def check_replication_claims(
    pilot_overall: dict, main_overall: dict, *, pilot_m: dict | None = None, main_m: dict | None = None
) -> dict:
    """Checks the 5 qualitative claims from spec item 16 against pilot and
    main independently. `pilot_overall`/`main_overall` are `{arm: {...}}`
    pooled metrics; `pilot_m`/`main_m` are `{arm: {...}}` M-target-only
    metrics (from `metrics_by_target.json`'s `"M"` entries), needed for claim 5."""
    results: dict = {}

    for label, metrics in (("pilot_200", pilot_overall), ("main_1000", main_overall)):
        if "C_constrained" not in metrics:
            continue
        top_arm = _best_arm(metrics, "accuracy")
        c_acc = metrics["C_constrained"].get("accuracy")
        top_acc = metrics[top_arm].get("accuracy") if top_arm else None
        near_highest = (
            c_acc is not None and top_acc is not None and (top_acc - c_acc) <= _NEAR_HIGHEST_MARGIN
        )
        results.setdefault("C_highest_or_near_highest_accuracy", {})[label] = {
            "holds": (top_arm == "C_constrained") or near_highest,
            "top_arm": top_arm, "top_accuracy": top_acc, "C_constrained_accuracy": c_acc,
        }

        # "C has relatively poor grounding" means: structure alone (C) does
        # NOT achieve what abstention/evidence-binding (C+/D) achieve -- i.e.
        # C's semantic_unsupported_rate is higher (worse) than BOTH of theirs.
        # It does NOT mean C is the single worst of all 4 arms -- a totally
        # unconstrained baseline (A) can be just as bad or worse; that's not
        # the claim being tested.
        c_rate = metrics["C_constrained"].get("semantic_unsupported_rate_over_evaluable")
        comparison_rates = [
            metrics[arm].get("semantic_unsupported_rate_over_evaluable")
            for arm in ("C_plus_unknown", "D_grounded") if arm in metrics
        ]
        comparison_rates = [r for r in comparison_rates if isinstance(r, (int, float))]
        results.setdefault("C_relatively_poor_grounding", {})[label] = {
            "holds": bool(comparison_rates) and c_rate is not None and all(c_rate > r for r in comparison_rates),
            "C_constrained_semantic_unsupported_rate": c_rate,
            "comparison_arm_rates": {
                arm: metrics[arm].get("semantic_unsupported_rate_over_evaluable")
                for arm in ("C_plus_unknown", "D_grounded") if arm in metrics
            },
        }

        if "C_plus_unknown" in metrics:
            c_val = metrics["C_constrained"].get("semantic_unsupported_rate_over_evaluable")
            cplus_val = metrics["C_plus_unknown"].get("semantic_unsupported_rate_over_evaluable")
            results.setdefault("C_plus_lower_unsupported_than_C", {})[label] = {
                "holds": (c_val is not None and cplus_val is not None and cplus_val < c_val),
                "C_constrained": c_val, "C_plus_unknown": cplus_val,
            }

        if "D_grounded" in metrics:
            best_arm = _best_arm(metrics, "semantic_supported_accuracy_over_evaluable")
            results.setdefault("D_strongest_grounded_reliability", {})[label] = {
                "holds": best_arm == "D_grounded",
                "best_arm": best_arm,
            }

    for label, m_metrics in (("pilot_200", pilot_m), ("main_1000", main_m)):
        if not m_metrics:
            continue
        gaps = {}
        for arm, metrics in m_metrics.items():
            acc = metrics.get("accuracy")
            semantic_acc = metrics.get("semantic_supported_accuracy_over_evaluable")
            if isinstance(acc, (int, float)) and isinstance(semantic_acc, (int, float)):
                gaps[arm] = acc - semantic_acc
        results.setdefault("M_accuracy_much_higher_than_semantic_supported_accuracy", {})[label] = {
            "holds": bool(gaps) and all(gap > 0.1 for gap in gaps.values()),
            "gaps_by_arm": gaps,
        }

    return results


CLAIM_DESCRIPTIONS = {
    "C_highest_or_near_highest_accuracy": "C_constrained has the highest (or near-highest) raw accuracy",
    "C_relatively_poor_grounding": "C_constrained's grounding is worse than BOTH C_plus_unknown's and D_grounded's",
    "C_plus_lower_unsupported_than_C": "C_plus_unknown has a lower semantic-unsupported rate than C_constrained",
    "D_strongest_grounded_reliability": "D_grounded has the strongest grounded reliability (highest semantic supported accuracy)",
    "M_accuracy_much_higher_than_semantic_supported_accuracy": "M-stage raw accuracy is much higher than semantic-supported accuracy, for every arm",
}


def render_pilot_vs_main_markdown(
    table_rows: list[dict], claim_results: dict, *, pilot_id: str, main_id: str
) -> str:
    lines = [
        f"# Pilot ({pilot_id}) vs Main ({main_id})",
        "",
        "Purpose: check whether the pilot's qualitative pattern replicates at "
        "scale -- this is NOT a hypothesis test between sample sizes.",
        "",
        "## Metric comparison (overall, pooled across T/N/M)",
        "",
        "| Arm | Metric | 200-case | 1000-case | Diff |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in table_rows:
        diff_str = f"{row['diff']:+.4f}" if row["diff"] is not None else "n/a"
        lines.append(
            f"| {row['arm']} | {row['metric']} | {row['pilot_200']:.4f} | {row['main_1000']:.4f} | {diff_str} |"
        )

    lines.append("\n## Replication of pilot findings\n")
    for claim, description in CLAIM_DESCRIPTIONS.items():
        lines.append(f"**{description}**\n")
        entry = claim_results.get(claim, {})
        for label in ("pilot_200", "main_1000"):
            if label not in entry:
                lines.append(f"- {label}: not evaluable (missing arm data)")
                continue
            holds = entry[label]["holds"]
            detail = {k: v for k, v in entry[label].items() if k != "holds"}
            lines.append(f"- {label}: {'HOLDS' if holds else 'does NOT hold'} ({detail})")
        lines.append("")

    return "\n".join(lines)
