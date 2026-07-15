"""03: score Track A (accuracy vs gold) + Track B (groundedness vs text) -> results/metrics/.

DESIGN §6. Two independent tracks:
  A — does the prediction match the gold label?
  B — is the prediction actually supported by the report text? (the signature)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from stageground.evaluation.normalize import DOMAIN, grounded, norm_pred

PRED_DIR = Path("results/predictions")
DATASET = Path("data/processed/dataset.parquet")
OUTDIR = Path("results/metrics")

STAGES = ["T", "N", "M"]
ARMS = ["A", "B", "C", "D"]


def score_arm(arm: str, data: pd.DataFrame) -> dict:
    pred = pd.read_parquet(PRED_DIR / f"{arm}.parquet")
    df = pred.merge(data, on="patient_filename", how="left")

    row = {"arm": arm, "n": len(df), "invalid_output_rate": df["invalid"].mean()}

    for s in STAGES:
        gold = df[f"gold_{s}"]
        pv = df[f"pred_{s}_value"].map(norm_pred)
        ev = df[f"pred_{s}_evidence"]

        # --- Track A: accuracy vs gold (only where gold exists) ---
        mask = gold.notna()
        g, p = gold[mask], pv[mask]
        row[f"{s}_accuracy"] = accuracy_score(g, p)
        row[f"{s}_macro_f1"] = f1_score(g, p, average="macro", zero_division=0)
        # bonus: did the RAW value already sit in the allowed set (no canon needed)?
        # format compliance: raw value is a legal domain value (incl. 'unknown')
        row[f"{s}_allowed_value_rate"] = df.loc[mask, f"pred_{s}_value"].isin(DOMAIN).mean()

        # --- Track B: groundedness vs text (all rows, gold-independent) ---
        asserted = pv != "unknown"
        is_grounded = pd.Series(
            [grounded(e, t) for e, t in zip(ev, df["text"])], index=df.index
        )
        row[f"{s}_abstention_rate"] = (~asserted).mean()
        # of the values it asserted, how many are backed by verbatim evidence?
        row[f"{s}_evidence_grounding_rate"] = (
            is_grounded[asserted].mean() if asserted.any() else float("nan")
        )
        # asserts a value with NO textual basis (the dangerous failure)
        row[f"{s}_unsupported_assertion_rate"] = (asserted & ~is_grounded).mean()
        # proxy for "appropriate abstention": on HARD rows (no explicit token),
        # how often did it abstain instead of guessing?
        hard = ~df[f"easy_{s}"].fillna(False)
        row[f"{s}_hard_abstention_rate"] = (
            (~asserted)[hard].mean() if hard.any() else float("nan")
        )

    return row


def main() -> None:
    data = pd.read_parquet(DATASET)[
        ["patient_filename", "text", "gold_T", "gold_N", "gold_M",
         "easy_T", "easy_N", "easy_M"]
    ]

    rows = [score_arm(arm, data) for arm in ARMS]
    metrics = pd.DataFrame(rows)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    metrics.to_parquet(OUTDIR / "summary.parquet", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 100)
    for s in STAGES:
        cols = ["arm"] + [c for c in metrics.columns if c.startswith(f"{s}_")]
        print(f"\n===== {s}-stage =====")
        print(metrics[cols].to_string(index=False))


if __name__ == "__main__":
    main()
