"""02: run an arm -> results/predictions/<arm>.parquet"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from concurrent.futures import ThreadPoolExecutor

from stageground.extraction.extractors import run_one

IN = Path("data/processed/eval_subset.parquet")
OUTDIR = Path("results/predictions")

STAGES = ["T", "N", "M"]


def run_arm(arm: str, limit: int | None, workers: int = 9) -> pd.DataFrame:
    df = pd.read_parquet(IN)
    if limit:
        df = df.head(limit)

    def one(row) -> dict:
        r = run_one(arm, row.text)
        rec = {"patient_filename": row.patient_filename, "arm": arm, "invalid": r.invalid, "raw": r.raw}
        for s in STAGES:
            f = getattr(r.extraction, f"{s}_stage") if r.extraction else None
            rec[f"pred_{s}_value"] = f.value if f else None
            rec[f"pred_{s}_evidence"] = f.evidence if f else None
            rec[f"pred_{s}_confidence"] = f.confidence if f else None
            rec[f"pred_{s}_reason"] = f.reason if f else None
        return rec

    rows = list(df.itertuples())

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = []

        for i, rec in enumerate(ex.map(one, rows), 1):
            results.append(rec)
            if i % 20 == 0:
                print(f" [{arm}] {i}/{len(rows)}")

    return pd.DataFrame(results)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "C", "D"], required=True)
    ap.add_argument("--limit", type=int, default=None) # e.g. limit for 5 smoke
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = run_arm(args.arm, args.limit)

    invalid_rate = out["invalid"].mean()
    dst = OUTDIR / f"{args.arm}.parquet"
    out.to_parquet(dst, index=False)
    print(f"[{args.arm}] {len(out)} rows -> {dst}  | invalid-rate={invalid_rate:.1%}")


if __name__ == "__main__":
    main()