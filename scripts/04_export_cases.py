"""04: export per-case JSON (the shared UI/API contract) + Track C aggregate.

Writes results/cases/<patient_id>.json — the single data contract the review UI
(static import now, FastAPI later) reads. Also prints the Track-C tag distribution
and source-boundary success rate per arm (DESIGN §6.2b).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from stageground.evaluation.normalize import grounded, norm_pred
from stageground.evaluation.source_boundary import source_boundary_success_rate, tag_case

PRED_DIR = Path("results/predictions")
EVAL = Path("data/processed/eval_subset.parquet")
CASES_DIR = Path("results/cases")
METRICS_DIR = Path("results/metrics")

STAGES = ["T", "N", "M"]
ARMS = ["A", "B", "C", "D"]


def _clean(v):
    """pandas NaN -> None so the value is JSON-serializable (valid null)."""
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def build_prediction(pred_row: pd.Series, gold: dict, text: str) -> dict:
    """One arm's prediction block for one case, with grounding + Track-C tag."""
    out = {}
    for s in STAGES:
        raw = _clean(pred_row[f"pred_{s}_value"])
        evidence = _clean(pred_row[f"pred_{s}_evidence"])
        norm = norm_pred(raw)
        is_grounded = grounded(evidence, text)
        g = gold[s]
        out[s] = {
            "value": raw,
            "value_norm": norm,
            "evidence": evidence,
            "confidence": _clean(pred_row[f"pred_{s}_confidence"]),
            "reason": _clean(pred_row[f"pred_{s}_reason"]),
            "grounded": bool(is_grounded),
            "boundary_tag": tag_case(g, norm, is_grounded, text) if g is not None else None,
        }
    return out


def main() -> None:
    ev = pd.read_parquet(EVAL)[
        ["patient_filename", "text", "gold_T", "gold_N", "gold_M"]
    ]
    preds = {a: pd.read_parquet(PRED_DIR / f"{a}.parquet").set_index("patient_filename")
             for a in ARMS}

    CASES_DIR.mkdir(parents=True, exist_ok=True)
    tag_counts = {a: {s: Counter() for s in STAGES} for a in ARMS}

    for row in ev.itertuples():
        pid = row.patient_filename
        gold = {s: (None if pd.isna(getattr(row, f"gold_{s}")) else getattr(row, f"gold_{s}"))
                for s in STAGES}
        case = {
            "patient_id": pid,
            "text": row.text,
            "gold": gold,
            "predictions": {
                a: build_prediction(preds[a].loc[pid], gold, row.text) for a in ARMS
            },
        }
        (CASES_DIR / f"{pid}.json").write_text(
            json.dumps(case, indent=2, ensure_ascii=False, allow_nan=False)
        )

        for a in ARMS:
            for s in STAGES:
                tag = case["predictions"][a][s]["boundary_tag"]
                if tag is not None:
                    tag_counts[a][s][tag] += 1

    # --- Track C aggregate ---
    rows = []
    for a in ARMS:
        for s in STAGES:
            c = tag_counts[a][s]
            tags = list(c.elements())
            rows.append({
                "arm": a, "stage": s, "n": sum(c.values()),
                **{t: c.get(t, 0) for t in
                   ["correct", "source_boundary_abstention", "over_abstention",
                    "unsupported_error", "report_gold_discordance"]},
                "source_boundary_success_rate": source_boundary_success_rate(tags),
            })
    trackc = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    trackc.to_parquet(METRICS_DIR / "track_c.parquet", index=False)

    print(f"wrote {len(ev)} case JSONs -> {CASES_DIR}/")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 100)
    print("\n===== Track C — source-boundary tags =====")
    print(trackc.to_string(index=False))


if __name__ == "__main__":
    main()
