# BB-TEN external baseline — results and interpretation

Paired comparison on the exact leak-free, label-compatible overlap between StageGround's committed 1,000-case run and BB-TEN's official TCGA held-out test split: **T n=143, N n=145, M n=162**. All five systems (A_zero_shot, C_constrained, C_plus_unknown, D_grounded, BB-TEN) are scored on identical case IDs and identical gold labels per target. No new GPT-4o calls were made; StageGround's numbers here are the existing committed predictions, filtered.

## Headline numbers

| Target | System | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|
| T | A_zero_shot | 79.0% | 0.785 | 0.818 |
| T | C_constrained | 78.3% | 0.769 | 0.786 |
| T | C_plus_unknown | 74.8% | 0.739 | 0.758 |
| T | D_grounded | 76.2% | 0.744 | 0.768 |
| T | **BB-TEN** | 79.7% | 0.795 | 0.801 |
| N | A_zero_shot | 80.7% | 0.773 | 0.802 |
| N | C_constrained | 81.4% | 0.780 | 0.803 |
| N | C_plus_unknown | 79.3% | 0.739 | 0.765 |
| N | D_grounded | 80.7% | 0.761 | 0.780 |
| N | **BB-TEN** | 80.7% | **0.679** | **0.670** |
| M | A_zero_shot | 46.3% | 0.480 | 0.647 |
| M | C_constrained | 53.1% | **0.555** | 0.712 |
| M | C_plus_unknown | 30.9% | 0.435 | 0.572 |
| M | D_grounded | 26.5% | 0.412 | 0.561 |
| M | **BB-TEN** | **90.7%** | 0.528 | 0.534 |

## 1. Does BB-TEN outperform the GPT-based approaches on label accuracy?

**Only for M, and that result needs immediate qualification (see Q3).** For T and N, BB-TEN's raw accuracy is statistically indistinguishable from every StageGround arm — every paired bootstrap CI crosses zero (T: diffs +0.007 to +0.049, all n.s.; N: diffs −0.007 to +0.014, all n.s.). For M, BB-TEN's raw accuracy (90.7%) is significantly higher than all four arms (+37.6pp to +64.2pp, all CIs exclude zero, p < 1/5000) — but balanced accuracy tells a different story (Q3).

## 2. Is the result consistent across T, N, and M?

**No.** T is a tie. N is a tie on raw accuracy but BB-TEN's balanced accuracy (0.679) and macro F1 (0.670) are the *lowest* of all five systems — meaning BB-TEN's N accuracy leans more heavily on the majority N0 class than the GPT arms do (class-level detail in `class_metrics.csv`: BB-TEN's N1 recall 0.694 and N2 recall 0.692 both trail every GPT arm's 0.85–0.92). M shows the largest and only statistically significant gap, but in the direction of a raw-accuracy illusion, not a genuine capability edge (Q3).

## 3. Does class imbalance materially affect the M conclusion?

**Yes — this is the central finding of this evaluation.** The M overlap subset is 150 gold-M0 vs. 12 gold-M1 (92.6% M0 prevalence). BB-TEN's M0 recall is 97.3%, but its **M1 recall is 8.3%** (1 of 12 correct) — the worst M1 detection of any system evaluated, GPT arms included (A: 50.0%, C: 58.3%, C+: 58.3%, D: 58.3% M1 recall). BB-TEN's headline 90.7% accuracy is almost entirely a base-rate effect of defaulting to the majority class; once corrected for imbalance, its balanced accuracy (0.528) sits below C_constrained's (0.555) and is not meaningfully different from A_zero_shot's (0.480). **This is the same prior-driven M0 pattern StageGround's own main results already document for the GPT arms — BB-TEN exhibits a more extreme version of it, not an escape from it.**

## 4. Does truncation appear to hurt BB-TEN, especially for M?

**Inconclusive on this sample — the M1 cells are too small to interpret.** 61/162 M cases (37.7%) exceeded the 1024-token limit and were truncated. Truncated-M accuracy (91.8%) was not lower than non-truncated (90.1%) — but this comparison is dominated by the majority M0 class, which is easy regardless of truncation. Looking specifically at M1: only 8 M1 cases were non-truncated (recall 0.125, i.e. 1/8) and only 4 were truncated (recall 0.000, i.e. 0/4) — both cells are far too small (n<5 in the truncated case, explicitly flagged in `truncation_analysis.csv`) to support any claim about whether truncation specifically drives the M1 failure versus the class-imbalance effect in Q3 doing so on its own. One illustrative anecdote from the earlier smoke test (not part of this n=162 statistic): a genuine M1 case was truncated at exactly the 1024-token boundary and misclassified as M0 — consistent with, but not proof of, a truncation effect.

## 5. Does a supervised classifier provide a useful external reference point?

**Yes, specifically for T and N**, where it performs comparably to prompting a general-purpose LLM without any prompt engineering, structured-output constraints, or evidence requirements — a useful sanity check that StageGround's GPT-based accuracy numbers aren't unusually low relative to a purpose-built classifier. For M, BB-TEN is a useful *negative* reference: it demonstrates that even a model trained specifically on this task, on thousands of TCGA reports, still collapses to the majority class on the minority label — reinforcing that M-stage extraction is a hard, base-rate-dominated problem independent of whether the underlying model is a fine-tuned classifier or a prompted LLM.

## 6. Does anything about BB-TEN change StageGround's main grounding conclusions?

**No — and it cannot, by construction.** BB-TEN is a pure sequence classifier with no evidence output. It can be compared on label extraction, but it does not emit evidence, so it cannot answer the source-grounding question StageGround is designed to study. Higher (or lower) BB-TEN label accuracy says nothing about whether a prediction is grounded in the source report — grounding is not a metric BB-TEN can be scored on, and none was computed (`Evidence: N/A`, `Semantic support: Not measurable`, per every row of `comparison_metrics.csv`). StageGround's central thesis — that label accuracy and source grounding are separate axes — is unaffected either way.

## Runtime

Full inference (3 models, 450 cases total) on this machine, CPU only: **188 seconds (~3.1 minutes)**, ~500MB peak memory per model (one loaded at a time). See `run_bbten.py` output / `bbten_predictions.csv` for per-target timing.

## Limitations

- n=143/145/162 is a fixed ceiling from the leak-free overlap with BB-TEN's official test split combined with StageGround's already-sampled 1,000 cases — it is not a limitation of method, but it does mean per-target confidence intervals are wide (visible directly in `paired_comparisons.json`).
- The M1 truncation breakdown (Q4) is underpowered (n=8 and n=4) and should not be treated as evidence either way.
- BB-TEN's HF model cards carry no explicit license field; treat as research-use only pending clarification from the authors.
- This evaluation used StageGround's own gold labels throughout (verified 99.9% consistent with BB-TEN's independently-derived AJCC labels on the full T14 test set, per the earlier feasibility check) — a shared labeling error in the underlying TCGA metadata would affect both systems identically and would not appear as a discrepancy here.

## Recommendation

**Add a small external-baseline section** (Option A) — the T/N results are a clean, informative "we're in the right ballpark versus a purpose-built classifier" sanity check, and the M result is a genuinely interesting, well-supported illustration of the same base-rate/imbalance phenomenon the M0-paradox section already highlights, from an independent model family. It should be a small secondary block under aggregate results, explicitly labeled "External supervised baseline — BB-TEN," never presented as a fifth sandbox arm, and should lead with the accuracy-vs-balanced-accuracy distinction for M so the imbalance point isn't lost. See the main chat response for the suggested frontend copy.
