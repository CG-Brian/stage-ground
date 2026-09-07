"""Comparison tables (spec §9): one row per arm, CSV + Markdown.

`tabulate` isn't a project dependency and adding it just for Markdown table
rendering would violate "avoid unnecessary frameworks" (spec §14), so
Markdown output is hand-rolled here instead of via `DataFrame.to_markdown`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stageground.evaluation.metrics import compute_all_metrics
from stageground.evaluation.records import PredictionRecord

TARGETS = ("T", "N", "M")

# Precise (snake_case) column name -> human-readable Markdown header. CSV
# output always keeps the precise names (spec §9/§13: readable labels in
# Markdown/figures, precise internal metric names everywhere else).
PRETTY_LABELS = {
    "arm": "Arm",
    "n_evaluable": "N (evaluable)",
    "accuracy": "Accuracy",
    "semantic_supported_accuracy": "Semantic Supported Accuracy",
    "span_unsupported_rate": "Span Unsupported Rate",
    "semantic_unsupported_rate": "Semantic Unsupported Rate",
    "abstention_rate": "Abstention",
    "coverage": "Coverage",
}


def build_comparison_table(records: list[PredictionRecord], *, target: str | None = None) -> pd.DataFrame:
    """One row per arm present in `records`. `target=None` pools across all
    targets present; otherwise filters to `record.target == target`. Always
    includes `n_evaluable` alongside every metric so the sample size backing
    each number is explicit (spec §9). `semantic_supported_accuracy` is
    reported as the headline grounding-accuracy column per spec §13 -- it is
    still an automated heuristic (see `metrics.py` module docstring), not
    confirmed semantic judgment."""
    arms = sorted({r.arm for r in records})
    rows = []
    for arm in arms:
        arm_records = [r for r in records if r.arm == arm]
        m = compute_all_metrics(arm_records, target=target)
        rows.append({
            "arm": arm,
            "n_evaluable": m["n_evaluable"],
            "accuracy": m["accuracy"],
            "semantic_supported_accuracy": m["semantic_supported_accuracy_over_evaluable"],
            "span_unsupported_rate": m["span_unsupported_rate_over_evaluable"],
            "semantic_unsupported_rate": m["semantic_unsupported_rate_over_evaluable"],
            "abstention_rate": m["abstention_rate"],
            "coverage": m["coverage"],
        })
    return pd.DataFrame(rows, columns=[
        "arm", "n_evaluable", "accuracy", "semantic_supported_accuracy",
        "span_unsupported_rate", "semantic_unsupported_rate", "abstention_rate", "coverage",
    ])


def _to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    headers = [PRETTY_LABELS.get(c, c) for c in cols]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("n/a" if v != v else f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_one(df: pd.DataFrame, path_stem: Path) -> None:
    df.to_csv(path_stem.with_suffix(".csv"), index=False)
    path_stem.with_suffix(".md").write_text(_to_markdown(df))


def write_tables(records: list[PredictionRecord], outdir: str | Path) -> Path:
    """Writes overall.csv/.md + one .csv/.md per T/N/M target into
    `outdir/tables/`. Returns the tables directory path."""
    tables_dir = Path(outdir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    _write_one(build_comparison_table(records, target=None), tables_dir / "overall")
    for target in TARGETS:
        _write_one(build_comparison_table(records, target=target), tables_dir / target)

    return tables_dir
