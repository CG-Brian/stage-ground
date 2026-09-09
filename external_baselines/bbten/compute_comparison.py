"""Compute the apples-to-apples label-extraction comparison between
StageGround's A/C/C+/D arms and the external BB-TEN baseline, restricted to
the exact leak-free, label-compatible overlap cases (see ../README.md).

Reads (never writes to any of these):
  - results/20260908T022659_gpt-4o_seed42_n1000/predictions.jsonl (committed
    StageGround experiment -- NOT regenerated, only filtered/read)
  - external_baselines/bbten/data/overlap_{T,N,M}.csv
  - external_baselines/bbten/bbten_predictions.csv (from run_bbten.py)

Writes (all under external_baselines/bbten/):
  - comparison_metrics.csv       accuracy/balanced-accuracy/macro-F1/coverage per (target, system)
  - class_metrics.csv            per-class recall/precision/support per (target, system, class)
  - confusion_matrices/*.csv     one confusion matrix per (target, system)
  - truncation_analysis.csv      BB-TEN accuracy stratified by truncated vs not
  - paired_comparisons.json      paired bootstrap BB-TEN vs each StageGround arm, per target

Only standard-library + pandas/numpy/scikit-learn needed (no torch) -- can be
run in the main project's `uv` environment.

    python compute_comparison.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# balanced_accuracy_score warns whenever y_pred contains a value absent from
# y_true -- expected and harmless here: StageGround arms can predict
# ABSTAIN_OR_INVALID, which is never a gold value by construction.
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

ROOT = Path(__file__).resolve().parent
SG_ROOT = ROOT.parent.parent
MAIN_PREDICTIONS = SG_ROOT / "results" / "20260908T022659_gpt-4o_seed42_n1000" / "predictions.jsonl"

ARMS = ["A_zero_shot", "C_constrained", "C_plus_unknown", "D_grounded"]
SYSTEMS = ARMS + ["BBTEN"]
CLASS_SPACE = {"T": ["T1", "T2", "T3", "T4"], "N": ["N0", "N1", "N2", "N3"], "M": ["M0", "M1"]}
ABSTAIN_LABEL = "ABSTAIN_OR_INVALID"  # bucket for StageGround's "unknown"/"INVALID" outputs -- BB-TEN never abstains
N_BOOT = 5000
SEED = 42


def load_overlap() -> dict[str, dict[str, str]]:
    """{target: {case_id: gold}}"""
    out = {}
    for t in CLASS_SPACE:
        df = pd.read_csv(ROOT / "data" / f"overlap_{t}.csv")
        out[t] = dict(zip(df["case_id"], df["gold"]))
    return out


def load_sg_long(overlap: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows = []
    with MAIN_PREDICTIONS.open() as f:
        for line in f:
            r = json.loads(line)
            t, cid, arm = r["target"], r["case_id"], r["arm"]
            if t not in overlap or cid not in overlap[t] or arm not in ARMS:
                continue
            # Cross-check: the main experiment's own ground_truth must match
            # the gold pulled from the static overlap CSV (same ultimate
            # source, dataset.parquet gold_{T,N,M} -- this assert is the
            # "verify label mappings" check, not a fallback).
            assert r["ground_truth"] == overlap[t][cid], (
                f"gold mismatch for {cid}/{t}: predictions.jsonl says "
                f"{r['ground_truth']!r}, overlap CSV says {overlap[t][cid]!r}"
            )
            gold = r["ground_truth"]
            pred_raw = r["prediction"]
            pred = pred_raw if pred_raw in CLASS_SPACE[t] else ABSTAIN_LABEL
            rows.append({
                "case_id": cid, "target": t, "system": arm,
                "gold": gold, "prediction": pred, "prediction_raw": pred_raw,
                "correct": pred_raw == gold,
                "abstained": bool(r["abstained"]),
            })
    return pd.DataFrame(rows)


def load_bbten_long() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "bbten_predictions.csv")
    df = df.rename(columns={"bbten_prediction": "prediction", "bbten_correct": "correct"})
    df["prediction_raw"] = df["prediction"]
    df["system"] = "BBTEN"
    df["abstained"] = False
    return df[["case_id", "target", "system", "gold", "prediction", "prediction_raw", "correct",
               "abstained", "bbten_confidence", "input_token_count", "was_truncated"]]


def paired_bootstrap_accuracy(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int, seed: int) -> dict:
    """correct_a - correct_b, resampling case indices WITH replacement
    (both arrays are the same length and index-aligned -- the paired design).
    """
    assert len(correct_a) == len(correct_b)
    n = len(correct_a)
    rng = np.random.default_rng(seed)
    observed_diff = correct_a.mean() - correct_b.mean()
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    # two-sided empirical p-value: fraction of resamples on the opposite side of zero, doubled
    n_cross = min((diffs <= 0).sum(), (diffs >= 0).sum())
    p_value = min(1.0, 2 * n_cross / n_boot)
    return {
        "diff": float(observed_diff),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(p_value),
        "n_boot": n_boot,
        "n_paired_cases": n,
        "significant": not (ci_low <= 0 <= ci_high),
    }


def format_p(p_value: float, n_boot: int) -> str:
    if p_value <= 1 / n_boot:
        return f"< {1 / n_boot:.4f}"
    return f"{p_value:.4f}"


def compute_metrics_table(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in CLASS_SPACE:
        classes = CLASS_SPACE[target]
        for system in SYSTEMS:
            sub = long_df[(long_df["target"] == target) & (long_df["system"] == system)]
            if sub.empty:
                continue
            n = len(sub)
            accuracy = sub["correct"].mean()
            coverage = 1 - sub["abstained"].mean()
            bal_acc = balanced_accuracy_score(sub["gold"], sub["prediction"])
            macro_f1 = f1_score(sub["gold"], sub["prediction"], labels=classes, average="macro", zero_division=0)
            rows.append({
                "target": target, "system": system, "n": n,
                "accuracy": round(accuracy, 4), "coverage": round(coverage, 4),
                "balanced_accuracy": round(bal_acc, 4), "macro_f1": round(macro_f1, 4),
            })
    return pd.DataFrame(rows)


def compute_class_metrics(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in CLASS_SPACE:
        classes = CLASS_SPACE[target]
        for system in SYSTEMS:
            sub = long_df[(long_df["target"] == target) & (long_df["system"] == system)]
            if sub.empty:
                continue
            recalls = recall_score(sub["gold"], sub["prediction"], labels=classes, average=None, zero_division=0)
            precisions = precision_score(sub["gold"], sub["prediction"], labels=classes, average=None, zero_division=0)
            supports = sub["gold"].value_counts().reindex(classes, fill_value=0)
            for cls, rec, prec in zip(classes, recalls, precisions):
                rows.append({
                    "target": target, "system": system, "class": cls,
                    "recall": round(float(rec), 4), "precision": round(float(prec), 4),
                    "support": int(supports[cls]),
                })
    return pd.DataFrame(rows)


def write_confusion_matrices(long_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    for target in CLASS_SPACE:
        classes = CLASS_SPACE[target]
        for system in SYSTEMS:
            sub = long_df[(long_df["target"] == target) & (long_df["system"] == system)]
            if sub.empty:
                continue
            labels = classes + [ABSTAIN_LABEL] if (sub["prediction"] == ABSTAIN_LABEL).any() else classes
            cm = confusion_matrix(sub["gold"], sub["prediction"], labels=labels)
            cm_df = pd.DataFrame(cm, index=[f"gold_{c}" for c in labels], columns=[f"pred_{c}" for c in labels])
            cm_df.to_csv(out_dir / f"{target}_{system}.csv")


def compute_truncation_analysis(bbten_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in CLASS_SPACE:
        sub = bbten_df[bbten_df["target"] == target]
        for truncated in [False, True]:
            grp = sub[sub["was_truncated"] == truncated]
            if grp.empty:
                continue
            row = {
                "target": target, "truncated": truncated, "n": len(grp),
                "accuracy": round(grp["correct"].mean(), 4),
            }
            if target == "M":
                m1 = grp[grp["gold"] == "M1"]
                row["n_m1"] = len(m1)
                row["m1_recall"] = round(m1["correct"].mean(), 4) if len(m1) else float("nan")
                if len(m1) < 5:
                    row["m1_recall_note"] = "n<5, too small to interpret"
            rows.append(row)
    return pd.DataFrame(rows)


def compute_paired_comparisons(long_df: pd.DataFrame) -> dict:
    results = {}
    for target in CLASS_SPACE:
        results[target] = {}
        bbten_sub = long_df[(long_df["target"] == target) & (long_df["system"] == "BBTEN")].set_index("case_id")
        for arm in ARMS:
            arm_sub = long_df[(long_df["target"] == target) & (long_df["system"] == arm)].set_index("case_id")
            shared_ids = sorted(set(bbten_sub.index) & set(arm_sub.index))
            assert len(shared_ids) == len(bbten_sub) == len(arm_sub), (
                f"case-set mismatch between BBTEN and {arm} on target {target}: "
                f"bbten n={len(bbten_sub)} arm n={len(arm_sub)} shared={len(shared_ids)}"
            )
            correct_bbten = bbten_sub.loc[shared_ids, "correct"].to_numpy(dtype=float)
            correct_arm = arm_sub.loc[shared_ids, "correct"].to_numpy(dtype=float)
            cmp = paired_bootstrap_accuracy(correct_bbten, correct_arm, N_BOOT, SEED)
            cmp["p_display"] = format_p(cmp["p_value"], cmp["n_boot"])
            results[target][f"BBTEN_vs_{arm}"] = cmp
    return results


def main() -> None:
    overlap = load_overlap()
    sg_long = load_sg_long(overlap)
    bbten_long = load_bbten_long()

    # sanity: every overlap case for every target must have exactly 4 SG rows (one per arm)
    for target in CLASS_SPACE:
        n_expected = len(overlap[target]) * len(ARMS)
        n_actual = len(sg_long[sg_long["target"] == target])
        assert n_actual == n_expected, f"{target}: expected {n_expected} SG rows, got {n_actual}"
        n_bbten = len(bbten_long[bbten_long["target"] == target])
        assert n_bbten == len(overlap[target]), f"{target}: expected {len(overlap[target])} BBTEN rows, got {n_bbten}"

    long_df = pd.concat([
        sg_long[["case_id", "target", "system", "gold", "prediction", "prediction_raw", "correct", "abstained"]],
        bbten_long[["case_id", "target", "system", "gold", "prediction", "prediction_raw", "correct", "abstained"]],
    ], ignore_index=True)

    metrics_df = compute_metrics_table(long_df)
    metrics_df.to_csv(ROOT / "comparison_metrics.csv", index=False)
    print("=== comparison_metrics.csv ===")
    print(metrics_df.to_string(index=False))

    class_df = compute_class_metrics(long_df)
    class_df.to_csv(ROOT / "class_metrics.csv", index=False)
    print("\n=== class_metrics.csv (M target) ===")
    print(class_df[class_df["target"] == "M"].to_string(index=False))

    write_confusion_matrices(long_df, ROOT / "confusion_matrices")
    print("\nwrote confusion_matrices/*.csv")

    trunc_df = compute_truncation_analysis(bbten_long)
    trunc_df.to_csv(ROOT / "truncation_analysis.csv", index=False)
    print("\n=== truncation_analysis.csv ===")
    print(trunc_df.to_string(index=False))

    paired = compute_paired_comparisons(long_df)
    with (ROOT / "paired_comparisons.json").open("w") as f:
        json.dump(paired, f, indent=2)
    print("\n=== paired_comparisons.json (accuracy, BBTEN - arm) ===")
    for target, cmps in paired.items():
        print(f"\n{target}:")
        for name, c in cmps.items():
            print(f"  {name}: diff={c['diff']:+.4f} CI=[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
                  f"p {c['p_display']} n={c['n_paired_cases']} sig={c['significant']}")

    print("\nAll artifacts written under", ROOT)


if __name__ == "__main__":
    main()
