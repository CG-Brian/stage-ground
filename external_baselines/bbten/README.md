# External supervised baseline: BB-TEN

**BB-TEN is a separate, published, task-specific classifier baseline — not a fifth StageGround arm.** It is evaluated here purely on label-extraction accuracy, on the exact subset of cases where a fair comparison is possible, and kept fully separate from the main A/C/C+/D ablation.

## What this is

[BB-TEN](https://github.com/tatonetti-lab/tnm-stage-classifier) (Tatonetti lab) is a set of three Clinical-BigBird classifiers (`jkefeli/CancerStage_Classifier_{T,N,M}`) trained on TCGA pathology reports to directly predict T/N/M stage as a supervised classification task — no evidence, no free text, just a label. This directory evaluates it against StageGround's committed A/C/C+/D results **on the same cases**, to see how a purpose-built supervised classifier compares to prompting a general-purpose LLM for the same label-extraction task.

## Why not just run BB-TEN on all 1,000 StageGround cases?

Because StageGround's corpus and BB-TEN's training corpus are (as it turns out) **the same underlying TCGA pathology-report dataset**. Running BB-TEN across the full 1,000 cases and calling it a fair test would mean scoring it partly on data it was trained on. We instead:

1. Reconstructed BB-TEN's own official train/test split by replaying the exact code in its `dataset_split_T14_N03_M01.ipynb` (fixed `random_state=0`, stratified 85/15 split) — verified byte-exact against the repo's own shipped `Demo/T14_test.csv` (1034/1034 IDs and labels match).
2. Intersected BB-TEN's official **held-out test** IDs (never train) with the cases already sampled into StageGround's 1,000-case experiment.
3. Confirmed zero overlap with BB-TEN's *training* set (checked explicitly; see `data/overlap_{T,N,M}.csv` provenance below).

That intersection is the ONLY subset used here:

| Target | n | Source |
|---|---:|---|
| T | 143 | `data/overlap_T.csv` |
| N | 145 | `data/overlap_N.csv` |
| M | 162 | `data/overlap_M.csv` |

Every case in these files is simultaneously: already part of StageGround's 1,000-case run, part of BB-TEN's official held-out test split, and label-compatible with BB-TEN's class space (BB-TEN's own test-set construction already excludes T0/TX, NX-variants, and MX, so every remaining gold label is automatically in-class). Full derivation and cross-checks are documented in the feasibility investigation this evaluation builds on (see the project's own conversation history / prior feasibility report).

## Metrics scope

BB-TEN is a plain sequence classifier — it does not cite evidence. So for BB-TEN:

```text
Evidence: N/A
Span grounding: Not measurable
Semantic support: Not measurable
```

These are never treated as zero or false. Only label-extraction metrics (accuracy, balanced accuracy, macro F1, confusion matrices, per-class precision/recall) are computed for BB-TEN. StageGround's A/C/C+/D are scored on the exact same case sets using their existing, already-committed predictions — **no GPT-4o calls were made for this task.**

## How to reproduce

BB-TEN's dependencies (`torch`, `transformers`) are deliberately kept out of the main StageGround `uv` environment.

```bash
# 1. Inference (needs torch + transformers; separate venv recommended)
cd external_baselines/bbten
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 run_bbten.py            # writes bbten_predictions.csv (~3 min CPU, ~1.5GB model download)

# 2. Comparison (pandas/scikit-learn only -- fine in the main project's uv env)
deactivate
cd /path/to/stage-ground
uv run python external_baselines/bbten/compute_comparison.py
```

## Files

| File | Contents |
|---|---|
| `data/overlap_{T,N,M}.csv` | The exact leak-free, label-compatible case IDs + gold labels used for every metric in this directory |
| `run_bbten.py` | Runs the 3 BB-TEN models on the overlap cases |
| `bbten_predictions.csv` | Raw BB-TEN output: prediction, confidence, correctness, token count, truncation flag, per case |
| `compute_comparison.py` | Restricts committed StageGround predictions to the same cases, computes all metrics |
| `comparison_metrics.csv` | accuracy / coverage / balanced accuracy / macro F1, per (target, system) |
| `class_metrics.csv` | per-class recall / precision / support, per (target, system, class) |
| `confusion_matrices/*.csv` | one confusion matrix per (target, system) |
| `truncation_analysis.csv` | BB-TEN accuracy split by whether the report exceeded its truncation limit |
| `paired_comparisons.json` | paired bootstrap (n_boot=5000) accuracy comparisons: BB-TEN vs each of A/C/C+/D, per target |
| `bbten_summary.md` | The conservative, written interpretation of all of the above |

## What this does NOT do

- Does not modify `results/20260908T022659_gpt-4o_seed42_n1000/` in any way (read-only).
- Does not modify the StageGround frontend.
- Does not add BB-TEN as a sandbox arm or into the main aggregate-results charts.
- Does not compute or imply any grounding/semantic-support score for BB-TEN.
