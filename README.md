# StageGround

**When a clinical LLM is "correct," is it actually grounded in the source?**

StageGround studies TNM stage extraction from TCGA pathology reports by separating registry-label accuracy from whether a prediction is actually supported by the document the model was given.

**[Interactive research replay →](https://web-plum-one-77.vercel.app)** · [Main results](results/20260908T022659_gpt-4o_seed42_n1000/final_scientific_summary.md) · [Methods](docs/methods.md)

```
1,000 reports · 12,000 paired predictions · 4 prompting strategies · T/N/M
```

This is a research evaluation, not a staging product. Nothing here is validated for, or intended for, clinical use.

Dataset label match is not the same thing as source support. A correct label can still be unsupported by the report it was supposedly read from. High accuracy is useful; high accuracy for the wrong reason is less useful.

The [research replay](https://web-plum-one-77.vercel.app) linked above lets you step through real committed predictions from this experiment — same report, four strategies, side by side — without running anything.

## Why this matters

Extraction pipelines get evaluated against a registry label and stop there. That's fine when the label and the source document actually agree, and quietly wrong when they don't — a pipeline can look highly accurate while routinely asserting things the document in front of it never said. For anything downstream that treats "the model said X" as evidence that X is written somewhere, that gap matters more than the accuracy number does.

## Research question

TCGA registry labels are richer than what a pathology report can tell you — M stage in particular is often confirmed by whole-body imaging the model never sees. The model only gets the report. So a model can hit the right label by matching a registry value it has no textual basis for, and standard accuracy has no way to tell the difference between that and genuine extraction.

> How do structured output, abstention, and mandatory evidence binding affect both label accuracy and source-grounded reliability in TNM extraction?

## What gets measured

Four separate axes, kept separate on purpose — collapsing them into one reliability score is exactly what would hide the effects below.

| Axis | Question |
|---|---|
| Accuracy | Does the predicted label match the registry gold value? |
| Span grounding | Does the cited evidence actually appear in the report? |
| Semantic support | Does that evidence actually justify the predicted label? |
| Abstention | Did the model decline to answer rather than guess? |

Full definitions and denominators: [`docs/evaluation.md`](docs/evaluation.md).

## Data

9,523 TCGA pathology reports (public, MIT-licensed corpus originally assembled by the [Tatonetti lab](https://github.com/tatonetti-lab/tnm-stage-classifier)), joined to GDC clinical metadata for T/N/M gold labels. Label coverage falls off target by target — T 73.1%, N 59.6%, M 48.4%, all three together 41.0% (3,907 reports) — and that drop isn't noise; M in particular often depends on imaging a pathology report doesn't contain. The main study samples 1,000 of those 3,907 fully-labeled reports. Details: [`docs/methods.md`](docs/methods.md).

## Experimental design

Four prompting strategies, same model, same 1,000 reports, same targets. Only the intervention changes.

| Strategy | Intervention |
|---|---|
| A — Zero-shot | Minimal constraint |
| C — Constrained | Allowed TNM values enforced |
| C+ — Constrained + unknown | Adds explicit permission to abstain |
| D — Grounded | Adds mandatory evidence binding |

Read A → C → C+ → D as one progressive intervention rather than four unrelated prompts: C adds structure to A, C+ adds permission to abstain on top of C, D adds a requirement — every asserted label must come with a verbatim quote from the report — on top of C+. It's a paired design: every arm sees the identical 1,000 cases, so a difference between arms is attributable to the prompting change, not to which reports happened to get sampled.

## Main findings

**1. Accuracy and grounding rank the strategies differently.**

| Arm | Accuracy | Coverage | Semantic-unsupported rate | Semantic-supported accuracy |
|---|---:|---:|---:|---:|
| A — Zero-shot | 68.9% | 93.5% | 59.4% | 26.6% |
| C — Constrained | **70.2%** | 88.9% | 51.7% | 29.8% |
| C+ — Constrained + unknown | 62.9% | 77.2% | 44.0% | 27.5% |
| D — Grounded | 62.3% | 79.7% | **40.4%** | **31.1%** |

C_constrained wins on raw accuracy; D_grounded wins on both grounding metrics. Neither arm wins on everything, and that's the point — a schema can stop malformed outputs, but it can't by itself stop a model from being confidently wrong about what the report supports.

**2. Evidence binding helps beyond abstention alone.** Going from C+ to D — adding a mandatory evidence requirement on top of an abstention option that already exists — moves the semantic-unsupported rate by −3.53 points and semantic-supported accuracy by +3.63 points. Coverage went *up* slightly (+2.53 points), not down, so D isn't just refusing to answer more often and calling that progress. This is the cleanest positive result in the study: the win comes from requiring evidence, not from permission to abstain.

**3. M stage shows how headline accuracy can mislead.** Gold M0 prevalence in this cohort is 94.7%. Across all four arms, M0 predictions land at roughly 99% accuracy — but the semantic-supported rate for those same M0 predictions is only 2–4%. That 99% is accuracy *among M0 predictions*, not overall M-stage accuracy, which is much lower and varies substantially by arm. On M0, being right was easy. Showing why was not.

The three findings above are pooled across T/N/M. Split apart, the targets don't behave alike: T and N accuracy are essentially flat across arms, and nearly all of the cross-arm movement comes from M. M stage turned out to be where accuracy gets especially deceptive.

Full numbers, confidence intervals, and the target-by-target breakdown: [`final_scientific_summary.md`](results/20260908T022659_gpt-4o_seed42_n1000/final_scientific_summary.md).

## External supervised baseline — BB-TEN

[BB-TEN](https://github.com/tatonetti-lab/tnm-stage-classifier) is a published Clinical-BigBird classifier trained directly on TCGA pathology reports for T/N/M staging. It is not a fifth StageGround arm — it's a different kind of system (fine-tuned classifier vs. prompted LLM) evaluated only where the comparison is fair: the leak-free intersection between StageGround's sampled cases and BB-TEN's own official held-out TCGA test split (T n=143, N n=145, M n=162 — same exact cases, no new GPT calls).

```
T:  BB-TEN 79.7%           StageGround arms 74.8–79.0%     no significant pairwise difference
N:  BB-TEN 80.7%           StageGround arms 79.3–81.4%     no significant pairwise difference
M:  BB-TEN raw accuracy 90.7%, balanced accuracy 0.528, M1 recall 8.3%
    C_constrained balanced accuracy 0.555, GPT-arm M1 recall 50–58%
```

T and N are a wash. M looks like a landslide for BB-TEN until you correct for the fact that this M subset is heavily M0-skewed (150 M0 vs. 12 M1) — its raw accuracy is mostly the majority class showing through, and its M1 recall is the worst of any system compared here, GPT arms included. BB-TEN doesn't emit evidence, so it's a label-extraction baseline, not a grounding baseline; nothing about its accuracy speaks to whether a prediction was supported by the report. It's a useful sanity check on T/N and a useful illustration, from an entirely separate model family, that the M0-prior problem above isn't specific to prompting GPT-4o. Full breakdown: [`external_baselines/bbten/bbten_summary.md`](external_baselines/bbten/bbten_summary.md).

## Limitations

- The main ablation uses one model (`gpt-4o`). Whether the pattern holds for other models is untested here.
- Semantic support is a conservative rule-based heuristic, not a clinical adjudicator or a trained classifier.
- The current 80-case audit is model-assisted provisional — a different LLM checked the heuristic's calls, not an independent human reviewer. Read it as corroborating signal, not final validation.
- M-stage registry labels can depend on information (imaging) that isn't in the pathology report at all, which caps how well any text-only method can do on M regardless of prompting.
- The BB-TEN comparison covers a few hundred cases per target, not the full 1,000, and BB-TEN has no evidence output to compare against StageGround's grounding metrics.
- Everything here is TCGA pathology reports from one corpus. Report style, formatting, and OCR artifacts from other institutions or EHR systems could behave differently.

## Reproducing the study

```bash
uv sync --extra dev
cp .env.example .env               # add your OpenAI API key
uv run python scripts/01_build_dataset.py
uv run python scripts/run_large_experiment.py \
  --arms A_zero_shot C_constrained C_plus_unknown D_grounded \
  --sample-size 1000 --seed 42 --model gpt-4o
```

Full commands, output layout, checkpointing behavior, and the audit/BB-TEN workflows: [`docs/reproducibility.md`](docs/reproducibility.md).

## Layout

```
src/stageground/       library code (data, extraction, evaluation)
scripts/                pipeline entry points and analysis generators
results/                one directory per experiment run, self-contained
external_baselines/     BB-TEN, kept separate from the main ablation
web/                    the research replay frontend
docs/                   methods, evaluation, pilot study, reproducibility
tests/                  unit tests
```

Deeper documentation lives in [`docs/`](docs/): [`methods.md`](docs/methods.md) (dataset, canonicalization, arm definitions), [`evaluation.md`](docs/evaluation.md) (metric definitions, bootstrap, audit process), [`pilot_200.md`](docs/pilot_200.md) (the earlier 200-case study this one grew out of), [`reproducibility.md`](docs/reproducibility.md) (exact commands and output layout).
