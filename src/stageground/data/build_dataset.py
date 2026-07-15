"""Build the eval dataset: load, join, canonicalize, easy/hard tag (DESIGN §7)."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from stageground.data.clean_labels import canonicalize

RAW = Path("data/raw")
OUT = Path("data/processed/dataset.parquet")

# explicit stage token in text -> "easy"; else hard
_EASY = {
    "T": re.compile(r"\bp?T[0-4X]", re.IGNORECASE),
    "N": re.compile(r"\bp?N[0-3X]", re.IGNORECASE),
    "M": re.compile(r"\bp?M[01X]", re.IGNORECASE),
}


def _safe_canon(v: object) -> str | None:
    """canonicalize, but pass through missing gold as None."""
    if pd.isna(v):
        return None
    return canonicalize(str(v))


def build() -> pd.DataFrame:
    reports = pd.read_csv(RAW / "TCGA_Reports.csv")
    # 'TCGA-BP-5195.25c0b433-...' -> 'TCGA-BP-5195'
    reports["submitter_id"] = reports["patient_filename"].str.split(".").str[0]

    df = reports
    for stage, fname in [
        ("t", "TCGA_T14_patients.csv"),
        ("n", "TCGA_N03_patients.csv"),
        ("m", "TCGA_M01_patients.csv"),
    ]:
        meta = pd.read_csv(RAW / fname)
        col = f"ajcc_pathologic_{stage}"
        keep = meta[["case_submitter_id", col, "project_id"]].drop_duplicates("case_submitter_id")
        df = df.merge(
            keep,
            left_on="submitter_id",
            right_on="case_submitter_id",
            how="left",
            suffixes=("", f"_{stage}"),
        )
        # canonicalized gold
        df[f"gold_{stage.upper()}"] = df[col].map(_safe_canon)

    # cancer type (project_id duplicated across merges: coalesce)
    proj_cols = [c for c in df.columns if c.startswith("project_id")]
    df["cancer_type"] = df[proj_cols].bfill(axis=1).iloc[:, 0]

    # easy/hard per stage from report text
    for s in ["T", "N", "M"]:
        df[f"easy_{s}"] = df["text"].str.contains(_EASY[s])

    return df


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    # sanity check
    n = len(df)
    cov = {s: df[f"gold_{s}"].notna().sum() for s in ["T", "N", "M"]}
    all3 = df[["gold_T", "gold_N", "gold_M"]].notna().all(axis=1).sum()
    print(f"reports: {n}")
    print(f"coverage T={cov['T']}  N={cov['N']}  M={cov['M']}  all3={all3}")
    print("expected T=6966. N=5678. M=4608. all3=3907")


if __name__ == "__main__":
    main()