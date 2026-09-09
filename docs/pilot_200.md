# Pilot study (v0, n=200)

This is the original pilot that motivated the main 1,000-case ablation. It used a different arm set (A/B/C/D, including a few-shot arm not carried into the main study), a different metric vocabulary (Track A/B/C), and a 200-case sample. Its qualitative conclusions replicated at n=1,000 — see the main [README](../README.md) and [`results/20260908T022659_gpt-4o_seed42_n1000/pilot_vs_main.md`](../results/20260908T022659_gpt-4o_seed42_n1000/pilot_vs_main.md) — but the numbers below are from the pilot itself and shouldn't be quoted as current results.

## Arms

| Arm | Constraint |
|---|---|
| A | zero-shot free-form (none) |
| B | few-shot — worked examples including one abstention, no explicit rule |
| C | allowed-values only — value domain listed, no evidence/abstention rule |
| D | schema-guided — allowed values + verbatim-evidence + abstain-with-reason rule |

`B_few_shot` was not carried into the main ablation. The main study's `C_constrained` / `C_plus_unknown` / `D_grounded` isolate what B and C were each hinting at (structure, abstention, evidence) more cleanly.

## M-stage decomposition (the discriminating target even then)

| Metric | A | B | C | D |
|---|---:|---:|---:|---:|
| Format compliance (allowed-value rate) ↑ | .21 | .88 | **1.00** | **1.00** |
| Evidence-grounding rate ↑ | .69 | **1.00** | .70 | **1.00** |
| Unsupported-assertion rate ↓ | .31 | **.00** | .30 | **.00** |
| Abstention rate | .00 | .51 | .01 | .47 |
| Track C source-boundary success ↑ | .07 | **.53** | .01 | .47 |
| Unsupported errors (count) | 33 | 0 | **56** | 0 |

The finding that carried forward: **C (allowed-values only) breaks the intuition that a schema alone makes output safe.** C maximizes format compliance but its groundedness is no better than free-form A — its unsupported-error count actually rises (56 vs. A's 33), because forced to emit *some* allowed value with no permission to abstain, the model hedges with `MX` (52 of 56 have no M descriptor anywhere in the report). Constraining the value set alone produces confidently unsupported assertions. This is the observation that later became the main study's four-way ablation: is it structure, abstention, or evidence-binding that actually fixes this?

B (few-shot) induced strong abstention from a single worked example — unsupported-assertion dropped to 0 and it briefly matched D's source-boundary success rate. That result is suggestive, not decisive (small n, one example), which is part of why the main study replaced B with the cleaner `C_plus_unknown` / `D_grounded` split.

## Track A — accuracy, roughly flat except by design on M

| Stage | A | B | C | D |
|---|---:|---:|---:|---:|
| T | .89 | .90 | .91 | .90 |
| N | .80 | .80 | .79 | .79 |
| M | .26 | .05 | .17 | .035 |

Arms that abstain (B, D) score low on Track A's M row *because they refuse to guess the dominant M0 label* when the report gives no basis — not because they're worse at the task. Read alone, this table rewards guessing; it's the reason a second, grounding-aware track exists at all.

## Track B / Track C — grounding and source-boundary agreement

Track B classifies every prediction by whether the report text supports it (asserted + grounded, asserted + unsupported, abstained + over-cautious, abstained + appropriate). Track C crosses that against gold agreement, so a wrong-by-accuracy prediction that correctly abstained on genuinely unsupportable gold isn't scored the same as a confidently wrong guess:

| Prediction vs. gold | Text condition | Tag |
|---|---|---|
| matches | — | `correct` |
| abstains | no textual basis for gold | `source_boundary_abstention` (success) |
| abstains | text supports gold | `over_abstention` (a real miss) |
| asserts other value | no verbatim evidence | `unsupported_error` |
| asserts other value | evidence supports it | `report_gold_discordance` (registry may reflect out-of-document info) |

| Arm | correct | boundary abstention ✓ | over-abstention | unsupported error ✗ | discordance | success rate |
|---|---:|---:|---:|---:|---:|---:|
| A | 52 | 10 | 0 | 33 | 105 | 0.07 |
| B | 10 | 101 | 1 | 0 | 88 | **0.53** |
| C | 33 | 2 | 0 | 56 | 109 | 0.01 |
| D | 7 | 91 | 2 | 0 | 100 | 0.47 |

Only an *abstention* against unsupportable gold counts as a source-boundary success — an asserted wrong value is an error regardless of whether gold happened to be unsupportable too. That guardrail is what stops the metric from laundering lucky abstentions into virtue.

## Manual audit (25 cases, LLM-judge)

Before trusting `grounded()` — the verbatim-substring proxy behind Track B/C — a stratified, blinded 25-case sample (arm and existing tag hidden) was scored by an independent LLM judge (a different model family than the extraction models) against a 4-question rubric.

Overall agreement: **23/25 (92%)**. `source_boundary_abstention` and `report_gold_discordance` were 100% (9/9, 8/8) — the tagging *logic* was correct on every sampled case. All disagreement concentrated in `unsupported_error` (6/8, 75%).

Both disagreements traced to the same mechanism: TCGA reports are OCR'd PDFs where line-wraps insert a stray period mid-phrase (`"under. investigation"`). The model's evidence was substantively correct — it quoted the right passage — but a strict verbatim-substring check failed on the punctuation noise, mislabeling a grounded prediction as unsupported. Fixed by normalizing periods/commas before matching (`stageground.evaluation.normalize.grounded`); the numbers above already reflect the fix. A larger reviewer study (n=100, per the original design) was never run — see `docs/archive/original_design.md` for that plan.
