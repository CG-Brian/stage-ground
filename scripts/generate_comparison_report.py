"""Thin CLI: read an experiment's bootstrap.json and write a human-readable
primary-comparison report (effect size + 95% CI first, properly-formatted
p-values -- see stageground.evaluation.comparison_report).

    uv run python scripts/generate_comparison_report.py --results-dir results/<experiment_id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stageground.evaluation.comparison_report import generate_comparison_report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    bootstrap_json = json.loads((results_dir / "bootstrap.json").read_text())
    report = generate_comparison_report(bootstrap_json)
    print(report)

    output_path = Path(args.output) if args.output else results_dir / "primary_comparisons.md"
    output_path.write_text(report)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
