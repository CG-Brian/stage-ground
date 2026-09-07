import numpy as np
import pandas as pd

from stageground.evaluation.sampling import stratified_sample


def _synthetic_pool(n=500, seed=0):
    rng = np.random.default_rng(seed)
    gold_T = np.where(rng.random(n) < 0.9, "T2", None)
    gold_N = np.where(rng.random(n) < 0.7, "N0", None)
    gold_M = np.where(rng.random(n) < 0.4, "M0", None)  # M is the rare/low-coverage stage
    return pd.DataFrame({
        "patient_filename": [f"p{i}" for i in range(n)],
        "text": [f"report {i}" for i in range(n)],
        "gold_T": gold_T,
        "gold_N": gold_N,
        "gold_M": gold_M,
    })


def test_determinism_same_seed_same_sample():
    df = _synthetic_pool()
    sample1, summary1 = stratified_sample(df, n=100, seed=42)
    sample2, summary2 = stratified_sample(df, n=100, seed=42)
    pd.testing.assert_frame_equal(sample1, sample2)
    assert summary1 == summary2


def test_different_seeds_give_different_samples():
    df = _synthetic_pool()
    sample1, _ = stratified_sample(df, n=100, seed=1)
    sample2, _ = stratified_sample(df, n=100, seed=2)
    assert not sample1["patient_filename"].tolist() == sample2["patient_filename"].tolist()


def test_n_greater_than_pool_returns_everything():
    df = _synthetic_pool(n=50)
    sample, summary = stratified_sample(df, n=1000, seed=7)
    assert len(sample) == 50
    assert summary.n_sampled == 50
    assert summary.total_reports == 50


def test_sample_size_matches_requested_n():
    df = _synthetic_pool(n=500)
    sample, summary = stratified_sample(df, n=150, seed=3)
    assert len(sample) == 150
    assert summary.n_sampled == 150


def test_rare_stratum_still_represented_when_n_is_reasonable():
    df = _synthetic_pool(n=500)
    sample, summary = stratified_sample(df, n=200, seed=5)
    assert summary.n_with_M > 0  # M is rare (~40%) but shouldn't vanish at n=200


def test_summary_counts_match_manual_notna_counts():
    df = _synthetic_pool(n=500)
    sample, summary = stratified_sample(df, n=120, seed=9)
    assert summary.n_with_T == int(sample["gold_T"].notna().sum())
    assert summary.n_with_N == int(sample["gold_N"].notna().sum())
    assert summary.n_with_M == int(sample["gold_M"].notna().sum())
    assert summary.n_with_all_TNM == int(
        sample[["gold_T", "gold_N", "gold_M"]].notna().all(axis=1).sum()
    )


def test_tiny_n_does_not_crash_across_many_strata():
    df = _synthetic_pool(n=500)
    sample, summary = stratified_sample(df, n=1, seed=11)
    assert len(sample) == 1
    assert summary.n_sampled == 1


def test_no_duplicate_rows_in_sample():
    df = _synthetic_pool(n=500)
    sample, _ = stratified_sample(df, n=137, seed=13)
    assert sample["patient_filename"].is_unique
