"""Thin CLI: compare a pilot experiment's metrics against the main
experiment's, checking whether the pilot's qualitative findings replicate at
scale (spec item 16) -- NOT a hypothesis test between sample sizes.

    uv run python scripts/generate_pilot_vs_main.py \
        --pilot-dir results/20260907T214735_gpt-4o_seed42_n200 \
        --main-dir results/<experiment_id> \
        --output results/<experiment_id>/pilot_vs_main.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stageground.evaluation.pilot_comparison import (
    build_pilot_vs_main_table,
    check_replication_claims,
    render_pilot_vs_main_markdown,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot-dir", required=True)
    ap.add_argument("--main-dir", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    pilot_dir, main_dir = Path(args.pilot_dir), Path(args.main_dir)

    # Prefer the RESCORED pilot metrics (current semantic-support heuristic)
    # over the pilot's original metrics.json, which was computed with
    # whatever heuristic existed when the pilot ran. Comparing stale-heuristic
    # pilot numbers against current-heuristic main numbers would conflate a
    # heuristic change with a sample-size effect -- see
    # scripts/rescore_semantic_support.py --emit-metrics-json.
    pilot_overall_path = pilot_dir / "metrics_rescored.json"
    pilot_by_target_path = pilot_dir / "metrics_by_target_rescored.json"
    if not pilot_overall_path.exists():
        print(
            f"WARNING: {pilot_overall_path} not found -- falling back to metrics.json, which may "
            "have been computed with an older semantic-support heuristic than the main run. Run "
            "scripts/rescore_semantic_support.py --emit-metrics-json on the pilot's predictions.jsonl first."
        )
        pilot_overall_path = pilot_dir / "metrics.json"
        pilot_by_target_path = pilot_dir / "metrics_by_target.json"

    pilot_overall = json.loads(pilot_overall_path.read_text())
    main_overall = json.loads((main_dir / "metrics.json").read_text())

    pilot_by_target = json.loads(pilot_by_target_path.read_text())
    main_by_target = json.loads((main_dir / "metrics_by_target.json").read_text())
    pilot_m = {arm: t.get("M", {}) for arm, t in pilot_by_target.items()}
    main_m = {arm: t.get("M", {}) for arm, t in main_by_target.items()}

    rows = build_pilot_vs_main_table(pilot_overall, main_overall)
    claims = check_replication_claims(pilot_overall, main_overall, pilot_m=pilot_m, main_m=main_m)
    report = render_pilot_vs_main_markdown(
        rows, claims, pilot_id=pilot_dir.name, main_id=main_dir.name
    )
    print(report)

    output_path = Path(args.output) if args.output else main_dir / "pilot_vs_main.md"
    output_path.write_text(report)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
