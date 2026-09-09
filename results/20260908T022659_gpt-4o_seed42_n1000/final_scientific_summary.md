# Final Scientific Summary — 1,000-case Ablation (`20260908T022659_gpt-4o_seed42_n1000`)

Paired design: same 1,000 cases (seed=42, stratified on gold T/N/M, `all-three` eligible pool of 3,907), same model config (`gpt-4o`, temperature=None, max_tokens=None, retries=3, `response_format=json_object`; requested config == effective config, confirmed in `config.json`) across all four arms: `A_zero_shot`, `C_constrained`, `C_plus_unknown` ("C+"), `D_grounded`. 12,000 PredictionRecords total (1000 cases × 3 targets × 4 arms), 0 invalid/malformed extractions, 0 duplicate (case, arm, target) triples, 4/4 arms present for every case, `allowed_value_compliance` ≥ 0.999 for C/C+/D (0.378 for `A_zero_shot`, which has no output schema constraint — expected, not a bug).

Throughout: **accuracy** = does the predicted label match gold; **span-grounded** = the cited evidence is verbatim text from the report; **semantically supported** = the cited evidence, read by a conservative heuristic, actually implies the predicted label; **abstention/coverage** = how often the arm declines to assert a label at all. These are four separate axes — an arm can be accurate without being grounded, or grounded without being correct.

---

## 1. Does the 200-case pilot's pattern replicate at n=1,000?

Yes, on all five qualitative claims checked in [`pilot_vs_main.md`](pilot_vs_main.md), and the estimates barely moved (largest overall-metric drift was +0.025 on `A_zero_shot` coverage):

| Pilot claim | 200-case | 1000-case |
|---|---|---|
| `C_constrained` has the highest (or near-highest) raw accuracy | HOLDS | HOLDS |
| `C_constrained`'s grounding is worse than both `C_plus_unknown`'s and `D_grounded`'s | HOLDS | HOLDS |
| `C_plus_unknown` has a lower semantic-unsupported rate than `C_constrained` | HOLDS | HOLDS |
| `D_grounded` has the highest semantic-supported accuracy | HOLDS | HOLDS |
| M-stage raw accuracy is much higher than M-stage semantic-supported accuracy, for every arm | HOLDS | HOLDS (gap actually widened slightly, e.g. C: 0.450 → 0.454) |

This is not a new experiment confirming an independent hypothesis — the 1,000-case run is a larger, more precise re-estimate of the same underlying quantities the pilot measured. What it adds is tight bootstrap CIs that the 200-case pilot could not support.

## 2. Which arm has the highest raw accuracy?

`C_constrained`, overall accuracy 0.702 vs `A_zero_shot` 0.689, `C_plus_unknown` 0.629, `D_grounded` 0.623 (`tables/overall.md`). The overall-pooled paired bootstrap confirms `C_constrained` > `C_plus_unknown` (diff −0.0727, 95% CI [−0.0840, −0.0617], p < 1/5000) and > `D_grounded` (diff −0.0790, 95% CI [−0.0910, −0.0673], p < 1/5000). By target, this accuracy edge is driven almost entirely by M (see §8) — T accuracy differences among C/C+/D are **not** statistically distinguishable (all three pairwise 95% CIs cross zero).

## 3. Which arm has the lowest semantic-unsupported rate (best grounding)?

`D_grounded`: 0.404 overall vs `C_plus_unknown` 0.440, `C_constrained` 0.517, `A_zero_shot` 0.594. Overall bootstrap: D < C+ (diff −0.0353, CI [−0.0497, −0.0213], p < 1/5000) and D < C (diff −0.1123, CI [−0.1277, −0.0973], p < 1/5000). This holds by target too, except N, where D vs C+ is not distinguishable (diff −0.0090, CI [−0.0350, +0.0180]).

## 4. Does abstention alone (without requiring evidence-binding) improve reliability? (C → C+)

Partially, and at a real accuracy cost. `C_plus_unknown` adds only the option to answer "unknown" on top of `C_constrained`'s prompt — no evidence-binding requirement. Abstention rises sharply (+0.1173 overall, CI [0.1053, 0.1293], p < 1/5000; driven mostly by M, +0.292), and semantic-unsupported rate among what's left drops (−0.0770 overall, p < 1/5000). But this comes with a drop in both raw accuracy (−0.0727 overall, p < 1/5000) and, overall, semantic-supported accuracy (−0.0227, CI [−0.0330, −0.0123], p < 1/5000 — C+ is *worse* than C on this joint correct-and-grounded metric, because it loses more correct-but-unsupported answers to abstention than it gains in genuinely-supported ones). By target this is uneven: for T, C+ is significantly *worse* on both semantic-unsupported rate (+0.024, p=0.046) and semantic-supported accuracy (−0.051, p < 1/5000) than C; for N, none of these C-vs-C+ differences reach significance. So "give the model an unknown option" is not a uniformly positive intervention on its own — its main measurable effect is trading coverage for a lower unsupported rate, and that trade is not obviously favorable outside of M.

## 5. Does adding evidence-binding on top of abstention help further? (C+ → D)

Yes, and this is the clearest positive result in the run. `D_grounded` requires the model to bind a claim to evidence (not just permit abstention). Relative to `C_plus_unknown`, `D_grounded` has significantly lower semantic-unsupported rate (−0.0353 overall, p < 1/5000; driven by T, −0.085, p < 1/5000) and significantly higher semantic-supported accuracy (+0.0363 overall, CI [0.0253, 0.0470], p < 1/5000; also higher for T, +0.085, p < 1/5000, and for M, +0.008, p=0.011). Coverage is *also* slightly higher for D than C+ (+0.0253, p < 1/5000), meaning D is not simply hiding behind more abstention to get there. Evidence-binding, not abstention alone, is what moves semantic-supported accuracy.

## 6. What does D cost in raw accuracy, relative to C?

−0.079 overall (95% CI [−0.091, −0.067], p < 1/5000) — driven almost entirely by M (−0.226, p < 1/5000); T shows no significant accuracy difference (−0.006, CI [−0.021, +0.010]) and N shows none either (−0.005, CI [−0.015, +0.005]). So the "cost" of requiring grounded evidence is concentrated in M-stage accuracy, not spread evenly across targets — and per §9, a large share of that M "accuracy" being displaced is M0 predictions that were never source-supported to begin with.

## 7. Is the T/N/M pattern consistent, or does one target dominate?

Not consistent — M is qualitatively different from T and N:
- **T**: accuracy indistinguishable across C/C+/D; D's advantage is entirely in grounding (semantic-supported accuracy, semantic-unsupported rate).
- **N**: mostly flat — small or non-significant differences across arms on nearly every metric except abstention/coverage (which move mechanically with the prompt design) and D's modestly lower span-unsupported rate vs C.
- **M**: the only target where accuracy differences between arms are large and highly significant, and where semantic-supported accuracy is uniformly near zero (0.018–0.025) for every arm — see §8–9.

Any headline "D trades accuracy for grounding" narrative is really an M-driven story riding on top of near-flat T/N accuracy.

## 8. M-stage, arm by arm

| Arm | Accuracy (over evaluable) | Accuracy (over asserted) | Abstention | Semantic-supported accuracy | Semantic-unsupported rate | Span-unsupported rate |
|---|---:|---:|---:|---:|---:|---:|
| A_zero_shot | 0.453 | 0.513 | 0.117 | 0.018 | 0.712 | 0.477 |
| C_constrained | 0.476 | 0.642 | 0.258 | 0.022 | 0.563 | 0.287 |
| C_plus_unknown | 0.276 | 0.613 | 0.550 | 0.017 | 0.328 | 0.161 |
| D_grounded | 0.250 | 0.497 | 0.497 | 0.025 | 0.316 | 0.089 |

Two things move together but are not the same thing: accuracy-over-evaluable collapses for C+/D almost entirely because abstention rises to ~50%, not because the arms are much worse when they do assert (accuracy-over-asserted only drops from 0.642 (C) to 0.613 (C+) / 0.497 (D)). Separately, semantic-supported accuracy stays uniformly low (0.017–0.025) across *all four* arms — none of them achieve M predictions that are both correct and genuinely evidence-backed more than about 1 in 40 times. D does have the lowest span-unsupported rate (0.089) by a wide margin, meaning when D asserts an M-stage it is far more likely to cite real, verbatim report text — but citing real text is not the same as that text actually supporting the specific M-stage claimed (see §9).

## 9. Does high M0 accuracy reflect genuine source support, or something else?

This is the strongest and most consistent finding in the whole run, and it sharpens (not weakens) at n=1,000.

| Arm | M0 predictions (of asserted M) | M0 accuracy | M0 semantic-supported rate | M0 span-grounded rate |
|---|---:|---:|---:|---:|
| A_zero_shot | 431 (48.8% of asserted M) | 0.988 | 0.021 | 0.325 |
| C_constrained | 450 (60.6%) | 0.991 | 0.020 | 0.469 |
| C_plus_unknown | 248 (55.1%) | 0.992 | 0.032 | 0.484 |
| D_grounded | 222 (44.1%) | 0.991 | 0.041 | 0.779 |

Gold M is 94.7% M0 / 5.3% M1 in this eligible pool (`extra_analysis/gold_label_distribution.csv`) — so an M0 prediction is correct roughly 94.7% of the time by base rate alone, and every arm's ~99% M0 accuracy is fully consistent with that base rate rather than requiring any report-specific reasoning. Critically, semantic-supported rate for these M0 predictions is only 2.0–4.1% even though they are "correct" ~99% of the time — i.e., near-ceiling M0 accuracy coexists with the model's own cited evidence *failing* to actually assert M0 in 96–98% of cases. This pattern is **consistent with prior-driven prediction** (the model defaulting to the majority label, which happens to be right most of the time) **rather than with label-agreement reflecting source support** — we do not have evidence to call it "guessing" in a stronger sense than that, since the model may be doing something more structured that this heuristic simply doesn't recognize as M0-specific evidence. D_grounded's span-grounded rate (0.779, far above the other arms) shows its prompt does successfully push the model to cite verbatim text for M0 far more often — but the semantic-supported rate barely moves (0.041 vs 0.020–0.032), so binding *a* span of evidence is not the same as that evidence *sufficiently justifying* M0 specifically. This is the same pattern the pilot found; at n=1,000 the M0 count (roughly 220–450 per arm) is now large enough that these percentages are not noise.

## 10. What conclusions are justified by this run

- The pilot's five qualitative findings replicate at 5x the sample size with materially unchanged effect sizes.
- Evidence-binding (D vs C+) produces a real, statistically significant improvement in semantic-supported accuracy and semantic-unsupported rate, *without* trading away coverage relative to C+ — this is not just "D abstains more."
- The C→C+ abstention-only change is a mixed intervention: it lowers the unsupported rate among asserted answers but also lowers accuracy and (overall) semantic-supported accuracy; it is not a free win.
- The T/N/M pattern is not homogeneous: M drives essentially all of the cross-arm accuracy differences, while T and N accuracy are largely arm-invariant.
- M0 predictions have near-ceiling label accuracy across all four arms (overall M-stage accuracy is not near ceiling — see §8), and that near-ceiling M0 accuracy is not accompanied by meaningful source support under this heuristic. Gold M0 prevalence (94.7%) is high enough that base-rate prediction alone would produce similarly high M0 accuracy — this is a genuine reliability caveat about M-stage evaluation in this dataset, not an artifact of one arm's prompt.

## 11. What is NOT justified by this run

- We cannot claim the model is literally "guessing" on M0 — we only know its cited evidence does not, per this heuristic, semantically support the M0 label it gave. It may be relying on real but uncited priors, partial reasoning, or evidence categories the heuristic doesn't recognize as M-relevant.
- We cannot generalize these arm rankings beyond `gpt-4o` at this prompt/schema version — no other model was tested here.
- Semantic-supported accuracy and semantic-unsupported rate are both derived from an automated regex/keyword heuristic (`semantic_support.py`), not from human adjudication of every case; the blinded human audit (80 cases, separate artifact) is the only ground truth available for how well this heuristic tracks human judgment, and it should not be treated as interchangeable with it.
- "D costs accuracy" should not be reported as a target-agnostic statement — it is true overall and for M, but not demonstrated for T or N, where the accuracy difference is not statistically distinguishable from zero.
- Nothing here establishes causality about *why* M0 predictions lack semantic support (e.g., whether it's training-distribution bias toward M0, prompt design, or the underlying reports genuinely lacking explicit M-stage language) — this run only documents that the pattern exists and is robust.

## 12. Suggested next experiment

The single most informative follow-up is a targeted audit of the M-target specifically: pull a stratified sample of M predictions (oversampling M1 and any M0 predictions with `evidence_span_found=True` but `evidence_semantically_supports_prediction=False`, since those are exactly the "cited real text that doesn't say M0" cases) for blinded human review, to test whether the semantic-support heuristic is too conservative for M-stage evidence specifically (which is often stated implicitly, e.g. via absence of metastasis-related findings, that a regex heuristic may not catch) or whether the model genuinely lacks grounding there. A secondary useful comparison would be re-running the same 1,000-case paired design on a second model to see whether the T/N-flat, M-collapsing pattern is model-specific or a property of this staging task and dataset.
