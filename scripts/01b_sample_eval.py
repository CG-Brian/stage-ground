"""Sample a fixed, reproducible eval subset from fully-labeled patients (DESIGN §7 step 5)."""

from pathlib import Path

import pandas as pd

IN = Path("data/processed/dataset.parquet")
OUT = Path("data/processed/eval_subset.parquet")

N = 200          # v0: 100-300
SEED = 42        # fixed -> reproducible


def main() -> None:
    df = pd.read_parquet(IN)

    full = df[df[["gold_T", "gold_N", "gold_M"]].notna().all(axis=1)].copy()

    per = N // full["easy_M"].nunique()
    parts = [
        g.sample(min(len(g), per), random_state=SEED)
        for _, g in full.groupby("easy_M")
    ]
    sample = pd.concat(parts).reset_index(drop=True)

    sample.to_parquet(OUT, index=False)
    print(f"eval subset: {len(sample)} rows")
    print("M easy/hard:\n", sample["easy_M"].value_counts())
    print("cancer types:", sample["cancer_type"].nunique())


if __name__ == "__main__":
    main()