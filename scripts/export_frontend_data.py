"""One-off, no-API-call export: read the committed main-experiment result
artifacts and emit a single frontend-friendly JSON blob consumed by
`web/src/data/stageground.ts`. This is the ONLY place that reads the raw
result files for the landing page -- every number the frontend renders
traces back to one of the paths under `SOURCES` below, never hand-typed.

    uv run python scripts/export_frontend_data.py

Writes web/src/data/generated/stageground-data.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_DIR = ROOT / "results" / "20260908T022659_gpt-4o_seed42_n1000"
PILOT_DIR = ROOT / "results" / "20260907T214735_gpt-4o_seed42_n200"
OUT_PATH = ROOT / "web" / "src" / "data" / "generated" / "stageground-data.json"

ARMS = ["A_zero_shot", "C_constrained", "C_plus_unknown", "D_grounded"]
ARM_LABELS = {
    "A_zero_shot": "A · Zero-shot",
    "C_constrained": "C · Constrained",
    "C_plus_unknown": "C+ · Constrained + Unknown",
    "D_grounded": "D · Grounded",
}

SOURCES = {
    "config": "results/20260908T022659_gpt-4o_seed42_n1000/config.json",
    "metrics_overall": "results/20260908T022659_gpt-4o_seed42_n1000/metrics.json",
    "metrics_by_target": "results/20260908T022659_gpt-4o_seed42_n1000/metrics_by_target.json",
    "bootstrap": "results/20260908T022659_gpt-4o_seed42_n1000/bootstrap.json",
    "m0_analysis": "results/20260908T022659_gpt-4o_seed42_n1000/extra_analysis/m0_analysis.csv",
    "gold_label_distribution": "results/20260908T022659_gpt-4o_seed42_n1000/extra_analysis/gold_label_distribution.csv",
    "pilot_vs_main": "results/20260908T022659_gpt-4o_seed42_n1000/pilot_vs_main.md",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _num(row: dict, key: str) -> float:
    return float(row[key])


def build() -> dict:
    config = json.loads((MAIN_DIR / "config.json").read_text())
    metrics_overall = json.loads((MAIN_DIR / "metrics.json").read_text())
    metrics_by_target = json.loads((MAIN_DIR / "metrics_by_target.json").read_text())
    bootstrap = json.loads((MAIN_DIR / "bootstrap.json").read_text())
    m0_rows = {r["arm"]: r for r in _read_csv(MAIN_DIR / "extra_analysis" / "m0_analysis.csv")}
    gold_rows = _read_csv(MAIN_DIR / "extra_analysis" / "gold_label_distribution.csv")

    metadata = {
        "n_cases": config["sample_size"],
        "n_arms": len(config["arms"]),
        "n_targets": 3,
        "n_predictions": config["sample_size"] * len(config["arms"]) * 3,
        "model": config["model"]["model"],
        "seed": config["seed"],
        "experiment_id": config["experiment_id"],
        "eligible_pool": config["sampling"]["total_reports"],
    }

    arms = []
    for arm in ARMS:
        m = metrics_overall[arm]
        arms.append({
            "id": arm,
            "label": ARM_LABELS[arm],
            "accuracy": m["accuracy"],
            "coverage": m["coverage"],
            "abstentionRate": m["abstention_rate"],
            "spanUnsupportedRate": m["span_unsupported_rate_over_evaluable"],
            "semanticUnsupportedRate": m["semantic_unsupported_rate_over_evaluable"],
            "semanticSupportedAccuracy": m["semantic_supported_accuracy_over_evaluable"],
            "allowedValueCompliance": m["allowed_value_compliance"],
        })

    by_target = {}
    for target in ("T", "N", "M"):
        by_target[target] = []
        for arm in ARMS:
            m = metrics_by_target[arm][target]
            by_target[target].append({
                "id": arm,
                "label": ARM_LABELS[arm],
                "accuracy": m["accuracy"],
                "accuracyOverAsserted": m["accuracy_over_asserted"],
                "coverage": m["coverage"],
                "abstentionRate": m["abstention_rate"],
                "spanUnsupportedRate": m["span_unsupported_rate_over_evaluable"],
                "semanticUnsupportedRate": m["semantic_unsupported_rate_over_evaluable"],
                "semanticSupportedAccuracy": m["semantic_supported_accuracy_over_evaluable"],
            })

    def find_bootstrap(scope: str, metric: str, arm_a: str, arm_b: str) -> dict:
        for row in bootstrap[scope][metric]:
            if row["arm_a"] == arm_a and row["arm_b"] == arm_b:
                return {
                    "diff": row["diff"],
                    "ciLow": row["ci_low"],
                    "ciHigh": row["ci_high"],
                    "pValue": row["p_value"],
                    "nBoot": row["n_boot"],
                    "significant": not (row["ci_low"] <= 0 <= row["ci_high"]),
                }
        raise KeyError((scope, metric, arm_a, arm_b))

    evidence_binding = {
        "semanticUnsupportedRate": find_bootstrap("overall", "semantic_unsupported_rate_over_evaluable", "D_grounded", "C_plus_unknown"),
        "semanticSupportedAccuracy": find_bootstrap("overall", "semantic_supported_accuracy_over_evaluable", "D_grounded", "C_plus_unknown"),
        "coverage": find_bootstrap("overall", "coverage", "D_grounded", "C_plus_unknown"),
    }

    abstention_only = {
        "abstentionRate": find_bootstrap("overall", "abstention_rate", "C_plus_unknown", "C_constrained"),
        "accuracy": find_bootstrap("overall", "accuracy", "C_plus_unknown", "C_constrained"),
        "semanticUnsupportedRate": find_bootstrap("overall", "semantic_unsupported_rate_over_evaluable", "C_plus_unknown", "C_constrained"),
        "semanticSupportedAccuracy": find_bootstrap("overall", "semantic_supported_accuracy_over_evaluable", "C_plus_unknown", "C_constrained"),
    }

    m0 = {
        "goldM0Fraction": next(float(r["fraction"]) for r in gold_rows if r["target"] == "M" and r["value"] == "M0"),
        "goldM1Fraction": next(float(r["fraction"]) for r in gold_rows if r["target"] == "M" and r["value"] == "M1"),
        "byArm": [
            {
                "id": arm,
                "label": ARM_LABELS[arm],
                "nM0Predictions": int(_num(m0_rows[arm], "n_m0_predictions")),
                "m0Accuracy": _num(m0_rows[arm], "m0_accuracy"),
                "m0ProportionOfAssertedM": _num(m0_rows[arm], "m0_proportion_of_asserted_m"),
                "m0SemanticSupportedRate": _num(m0_rows[arm], "m0_semantic_supported_rate"),
                "m0SpanGroundedRate": _num(m0_rows[arm], "m0_span_grounded_rate"),
            }
            for arm in ARMS
        ],
    }

    return {
        "sources": SOURCES,
        "metadata": metadata,
        "arms": arms,
        "byTarget": by_target,
        "evidenceBinding": evidence_binding,
        "abstentionOnly": abstention_only,
        "m0": m0,
    }


def main() -> None:
    data = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
