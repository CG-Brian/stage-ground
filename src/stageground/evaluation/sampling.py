"""Deterministic, stratified, scalable evaluation sampling (spec §2).

`stratified_sample` never hardcodes a sample size -- callers (the CLI in
`evaluate.py`) always pass `n` explicitly, so the same function serves a
200-report pilot, a 1,000-report run, a 2,000+ report run, or "everything"
(`n >= len(df)`).

Stratification key is the T/N/M gold-*availability* pattern (8 possible
strata: which subset of the three targets has a non-null gold label for that
row), not the gold *values* themselves -- this keeps every downstream metric
(which is itself conditioned on gold availability, see `metrics.py`)
represented in proportion to how it naturally occurs in the pool, rather than
accidentally over- or under-sampling, say, M-labeled rows relative to T/N.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DEFAULT_GOLD_COLS = ("gold_T", "gold_N", "gold_M")


@dataclass(frozen=True)
class SamplingSummary:
    total_reports: int  # size of the eligible pool sampling drew from
    n_sampled: int
    n_with_T: int
    n_with_N: int
    n_with_M: int
    n_with_all_TNM: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def _build_summary(
    pool_df: pd.DataFrame, sample_df: pd.DataFrame, gold_cols: tuple[str, str, str], seed: int
) -> SamplingSummary:
    t_col, n_col, m_col = gold_cols
    return SamplingSummary(
        total_reports=len(pool_df),
        n_sampled=len(sample_df),
        n_with_T=int(sample_df[t_col].notna().sum()),
        n_with_N=int(sample_df[n_col].notna().sum()),
        n_with_M=int(sample_df[m_col].notna().sum()),
        n_with_all_TNM=int(sample_df[list(gold_cols)].notna().all(axis=1).sum()),
        seed=seed,
    )


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int,
    *,
    gold_cols: tuple[str, str, str] = DEFAULT_GOLD_COLS,
) -> tuple[pd.DataFrame, SamplingSummary]:
    """Deterministic sample of `min(n, len(df))` rows from `df`, stratified by
    T/N/M gold-availability pattern, using `seed` for both stratum allocation
    and within-stratum draws. Calling this twice with the same `(df, n, seed)`
    returns bit-identical output (same rows, same order).
    """
    if n >= len(df):
        sample = df.reset_index(drop=True)
        return sample, _build_summary(df, sample, gold_cols, seed)

    t_col, n_col, m_col = gold_cols
    strata_key = df[[t_col, n_col, m_col]].notna().apply(tuple, axis=1)
    rng = np.random.default_rng(seed)

    sizes = strata_key.value_counts()
    proportions = sizes / sizes.sum() * n
    base = np.floor(proportions).astype(int)
    remainder = n - int(base.sum())

    # Largest-remainder method for proportional allocation; ties broken by
    # pandas' stable sort over `sizes`' (deterministic) original order.
    fracs = (proportions - base).sort_values(ascending=False)
    alloc = base.copy()
    for key in fracs.index[:remainder]:
        alloc[key] += 1
    alloc = pd.Series({k: min(int(alloc[k]), int(sizes[k])) for k in alloc.index})

    chosen_idx: list = []
    for key, count in alloc.items():
        if count <= 0:
            continue
        pool_idx = df.index[strata_key == key].to_numpy()
        picked = rng.choice(pool_idx, size=count, replace=False)
        chosen_idx.extend(picked.tolist())

    # Per-stratum caps (a stratum smaller than its proportional allocation)
    # can under-fill by a few rows; top up from whatever's left in the pool.
    shortfall = n - len(chosen_idx)
    if shortfall > 0:
        remaining = df.index.difference(pd.Index(chosen_idx))
        extra = rng.choice(remaining.to_numpy(), size=min(shortfall, len(remaining)), replace=False)
        chosen_idx.extend(extra.tolist())

    chosen_idx = list(rng.permutation(chosen_idx))  # deterministic row-order shuffle
    sample = df.loc[chosen_idx].reset_index(drop=True)
    return sample, _build_summary(df, sample, gold_cols, seed)
