# StageGround — research landing page

A static Next.js page presenting StageGround's source-grounded evaluation of
clinical LLM extraction (TNM cancer staging as the test case). No backend, no
live LLM calls — every number comes from the committed 1,000-case experiment
under `../results/20260908T022659_gpt-4o_seed42_n1000/`.

## Data flow

```
results/<experiment>/*.json, *.csv, predictions.jsonl
        │  (uv run python ../scripts/export_frontend_data.py)
        │  (uv run python ../scripts/export_case_examples.py)
        ▼
src/data/generated/*.json
        │  (typed accessors, no new numbers)
        ▼
src/data/stageground.ts
        ▼
components/*
```

Re-run the two export scripts from the repo root after any change to the
committed results, then restart `npm run dev` / rebuild.

## Commands

```bash
npm install
npm run dev      # http://localhost:3000
npm run build    # production build
npm run lint
```

See the repo root [README.md](../README.md#research-landing-page-web) for
how this fits into the rest of the project.
