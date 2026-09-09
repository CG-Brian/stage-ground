# Reproducing the study

Exact commands, output layout, and the checkpointing/retry behavior that made the 1,000-case run survive rate limits and a machine sleep cycle without corrupting results. For what the four arms mean, see [`methods.md`](methods.md); for what the metrics mean, see [`evaluation.md`](evaluation.md).

## Setup

```bash
uv sync --extra dev              # core deps + pytest/ruff
cp .env.example .env             # add your OpenAI API key
```

## Building the dataset

```bash
uv run python scripts/01_build_dataset.py    # load, join, canonicalize -> data/processed/*.parquet
```

## Running the main 1,000-case experiment

This is the command that produced `results/20260908T022659_gpt-4o_seed42_n1000/`. It uses a checkpointed, concurrent runner rather than the plain `stageground.evaluate` CLI, because a run this size is long enough to hit OpenAI rate limits or survive a laptop sleep cycle, and neither should force a restart from zero:

```bash
uv run python scripts/run_large_experiment.py \
  --arms A_zero_shot C_constrained C_plus_unknown D_grounded \
  --sample-size 1000 --seed 42 --model gpt-4o \
  --max-workers 3 --n-boot 5000 \
  --experiment-id <your-experiment-id>
```

Re-running the identical command resumes from `results/_checkpoints/<experiment-id>.jsonl` — every `(case, arm)` pair already completed is skipped, not re-requested. A request that fails with a rate-limit error is retried in a later round rather than permanently recorded as invalid; a request that fails because the account has no remaining quota aborts the run immediately with a clear message instead of retrying forever (OpenAI surfaces both conditions as the same HTTP 429, so the runner checks the error's `.code` to tell them apart). Pass `--dry-run` to print case/arm/request counts and the requested-vs-effective model config without making any API calls.

## Running a smaller experiment

The plain evaluation CLI is enough for anything that doesn't need checkpointing:

```bash
uv run python -m stageground.evaluate \
  --arms A_zero_shot C_constrained C_plus_unknown D_grounded \
  --sample-size 200 --seed 42 --model gpt-4o
```

`--sample-size` is a required argument, plumbed straight into `stratified_sample` — never hardcoded. `--pool all-three` (default) restricts to reports with all of T/N/M gold; `--pool any` uses reports with at least one target's gold.

## Output layout

Each run writes a self-contained, timestamped directory and never touches another run's:

```
results/<experiment_id>/
  config.json             # arms, model, seed, sample size, sampling summary, timestamp
  predictions.jsonl       # one record per (case, arm, target); raw model output preserved
  metrics.json            # per-arm metrics, pooled across T/N/M
  metrics_by_target.json  # per-arm metrics, split by T/N/M
  error_breakdown.json    # per-arm, per-target error-taxonomy counts
  bootstrap.json          # paired-bootstrap arm comparisons
  tables/                 # overall/T/N/M comparison tables, .csv + .md
  figures/                # accuracy-vs-grounding and error-breakdown plots
```

`config.json` records both the *requested* model config and the *effective* one — see [`methods.md`](methods.md#model-configuration) for why those can differ.

## Regenerating analysis without re-running extraction

```python
from stageground.evaluation.records import from_jsonl
from stageground.evaluation.tables import write_tables
from stageground.evaluation.plots import plot_error_breakdown

records = from_jsonl("results/<experiment_id>/predictions.jsonl")
write_tables(records, "results/<experiment_id>")
plot_error_breakdown(records, "results/<experiment_id>/figures/fig3_M.png", target="M")
```

The extra analysis behind the M0 deep-dive and the pilot-vs-main comparison are their own scripts:

```bash
uv run python scripts/generate_extra_analysis.py --results-dir results/<experiment_id>
uv run python scripts/generate_pilot_vs_main.py \
  --pilot-dir results/20260907T214735_gpt-4o_seed42_n200 \
  --main-dir results/<experiment_id>
uv run python scripts/generate_comparison_report.py --results-dir results/<experiment_id>
```

## Running a blinded audit

```bash
uv run python -m stageground.evaluation.audit build \
  --predictions results/<experiment_id>/predictions.jsonl \
  --dataset data/processed/dataset.parquet \
  --target-count T=20 N=20 M=40 \
  --seed 42 --mode grounding-blind \
  --output audit/audit_00N

# ... a human reviewer fills in reviewer.jsonl ...

uv run python -m stageground.evaluation.audit score --audit-dir audit/audit_00N
```

See [`evaluation.md`](evaluation.md#blinded-human-audit) for what the reviewer fields mean and how blinding works.

## Running the external BB-TEN baseline

BB-TEN's dependencies (`torch`, `transformers`) are deliberately kept out of the main project environment. See [`external_baselines/bbten/README.md`](../external_baselines/bbten/README.md) for the full, separate reproduction steps.

## Running the frontend

```bash
# regenerate the frontend's data (only needed after a new experiment run)
uv run python scripts/export_frontend_data.py
uv run python scripts/export_case_examples.py

cd web
npm install
npm run dev      # http://localhost:3000
npm run build    # production build
```

`web/src/data/stageground.ts` documents the source result file for every metric it exposes — nothing in the frontend is hand-typed.

## Tests

```bash
uv run pytest -q
uv run ruff check src/ tests/ scripts/
```
