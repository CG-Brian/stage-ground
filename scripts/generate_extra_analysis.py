"""Thin CLI: generate the experiment-specific analyses not already produced
by `stageground.evaluate.run_evaluation` -- M0 deep-dive, gold/predicted
label distributions, coverage-vs-accuracy, error-taxonomy rates (all from
`stageground.evaluation.experiment_analysis`), plus the two additional
figures (`plot_semantic_unsupported_by_target`, `plot_m_accuracy_vs_semantic_supported`)
that aren't wired into the main pipeline.

    uv run python scripts/generate_extra_analysis.py --results-dir results/<experiment_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stageground.evaluation.experiment_analysis import (
    coverage_vs_accuracy_table,
    error_taxonomy_rates,
    gold_label_distribution,
    m0_prediction_analysis,
    predicted_label_distribution,
)
from stageground.evaluation.plots import (
    plot_m_accuracy_vs_semantic_supported,
    plot_semantic_unsupported_by_target,
)
from stageground.evaluation.records import from_jsonl

TARGETS = ("T", "N", "M")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    records = from_jsonl(results_dir / "predictions.jsonl")

    analysis_dir = results_dir / "extra_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    m0 = m0_prediction_analysis(records)
    m0.to_csv(analysis_dir / "m0_analysis.csv", index=False)
    print("\n=== M0 prediction analysis ===")
    print(m0.to_string(index=False))

    gold_dist = gold_label_distribution(records)
    gold_dist.to_csv(analysis_dir / "gold_label_distribution.csv", index=False)
    print("\n=== Gold label distribution ===")
    print(gold_dist.to_string(index=False))

    pred_dist = predicted_label_distribution(records)
    pred_dist.to_csv(analysis_dir / "predicted_label_distribution.csv", index=False)
    print("\n=== Predicted label distribution (head) ===")
    print(pred_dist.head(40).to_string(index=False))

    print("\n=== Coverage vs accuracy ===")
    for label, target in [("overall", None), ("T", "T"), ("N", "N"), ("M", "M")]:
        table = coverage_vs_accuracy_table(records, target=target)
        table.to_csv(analysis_dir / f"coverage_vs_accuracy_{label}.csv", index=False)
        print(f"\n-- {label} --")
        print(table.to_string(index=False))

    print("\n=== Error taxonomy rates ===")
    for label, target in [("overall", None), ("T", "T"), ("N", "N"), ("M", "M")]:
        table = error_taxonomy_rates(records, target=target)
        table.to_csv(analysis_dir / f"error_taxonomy_{label}.csv", index=False)

    figures_dir = results_dir / "figures"
    plot_semantic_unsupported_by_target(records, figures_dir / "fig4_semantic_unsupported_by_target.png")
    plot_m_accuracy_vs_semantic_supported(records, figures_dir / "fig5_m_accuracy_vs_semantic_supported.png")
    print(f"\nwrote figures -> {figures_dir}/fig4_semantic_unsupported_by_target.png, fig5_m_accuracy_vs_semantic_supported.png")
    print(f"wrote tables -> {analysis_dir}/")


if __name__ == "__main__":
    main()
