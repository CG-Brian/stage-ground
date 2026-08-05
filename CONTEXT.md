# StageGround Context

Last updated: 2026-08-05

## What this project is

StageGround is a clinical-AI validation project for TNM extraction from TCGA pathology reports. The core question is not just whether an LLM matches registry TNM labels, but whether each staging claim is actually supported by the source pathology report.

The important framing:

- Track A measures normal accuracy against registry gold labels.
- Track B measures groundedness: whether predicted evidence appears in the report text.
- Track C crosses gold disagreement with source support, separating principled abstention from unsupported errors.
- The main insight is that allowed values alone improve format compliance but do not make extraction safer. The prompt must also require evidence and permit `unknown` when the source report cannot support a value.

## Current local state before machine format

As of 2026-08-05, `main` had no committed divergence from `origin/main` after `git fetch origin`, but there were local uncommitted changes. The root repository should be committed and pushed with this file before formatting.

Root repo changes ready to preserve:

- README results updated after the OCR-noise grounding fix.
- `src/stageground/evaluation/normalize.py` now normalizes periods and commas away before evidence matching, because TCGA OCR line-wrap noise can split phrases like `under. investigation` and `cannot be. assessed`.
- `src/stageground/evaluation/track_b.py` delegates evidence matching to `grounded()`.
- `scripts/05_audit_sample.py` creates a stratified blinded 25-case Track-C audit sample.
- `scripts/06_audit_score.py` scores blind judge results against pipeline Track-C tags.
- `results/audit/` contains the blinded sheet, answer key, judgments, and scored audit output.
- `results/metrics/summary.parquet`, `results/metrics/track_c.parquet`, and selected `results/cases/*.json` were regenerated after the grounding fix.
- `pyproject.toml` adds optional `arm-e` dependencies for a future supervised reference arm.

Validation run before commit:

- `uv --cache-dir .uv-cache run pytest` passed: 35 tests.
- `uv --cache-dir .uv-cache run python scripts/06_audit_score.py` produced 23/25 audit agreement.
- `cd web && npm run lint` passed.

Audit result to remember:

- Overall agreement: 23/25 = 92%.
- `source_boundary_abstention`: 9/9 agreement.
- `report_gold_discordance`: 8/8 agreement.
- `unsupported_error`: 6/8 agreement.
- Both disagreements were cases where OCR punctuation noise made strict substring evidence matching too brittle; the current `grounded()` fix addresses this.

## Web app warning

`web/` is currently a nested git repository recorded by the root repo as a gitlink (`mode 160000`) but there is no `.gitmodules` file and `web` has no configured remote. That means root commits do not preserve the actual `web/` file contents unless this structure is fixed later.

Current uncommitted `web/` changes at the time this context was written:

- Modified: `web/app/globals.css`
- Modified: `web/app/page.tsx`
- Untracked: `web/app/case/`
- Untracked: `web/components/`
- Untracked: `web/lib/`

Recommendation after restoring from GitHub:

1. Decide whether `web/` should be a normal tracked folder or a proper submodule.
2. If normal folder, remove the nested `web/.git` and add the web files from the root repo.
3. If submodule, create/push a separate GitHub repo for `web` and add a valid `.gitmodules`.
4. Update `web/app/layout.tsx` metadata from the default Create Next App values.
5. Avoid `next/font/google` if deploying in restricted/offline build environments, or configure builds with network access.

## Useful commands

```bash
uv sync --extra dev
uv --cache-dir .uv-cache run pytest
uv --cache-dir .uv-cache run python scripts/06_audit_score.py

uv sync --extra serve
uv run uvicorn stageground.api.app:app --reload

cd web
npm install
npm run lint
npm run dev
```

## Portfolio positioning

Short version:

> StageGround evaluates LLM TNM extraction from pathology reports by checking not only registry-label accuracy, but whether every staging claim is grounded in the source document.

Plain-English version:

> This project catches when an AI guesses cancer stage values that are not actually stated or supported by the pathology report.

