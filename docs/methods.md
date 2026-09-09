# Methods

How the data was built, what the four arms actually differ on, and how cases were sampled and configured for the main 1,000-case study. For metric definitions and the audit process, see [`evaluation.md`](evaluation.md). For exact commands, see [`reproducibility.md`](reproducibility.md).

## Dataset

Reports and gold labels both come from the public TCGA pathology-report corpus originally assembled by the [Tatonetti lab](https://github.com/tatonetti-lab/tnm-stage-classifier) (MIT-licensed): 9,523 machine-readable pathology reports, joined to GDC clinical metadata on `case_submitter_id`.

| | patients with gold | % of 9,523 reports |
|---|---:|---:|
| T | 6,966 | 73.1% |
| N | 5,678 | 59.6% |
| M | 4,608 | 48.4% |
| **all of T, N, M** | **3,907** | **41.0%** |

Coverage falls T > N > M, and that ordering isn't noise — M stage often depends on whole-body imaging that a pathology report alone doesn't contain. That's exactly why M is the target where the accuracy/grounding gap shows up most.

The main 1,000-case experiment draws a deterministic, stratified sample (seed 42) from the 3,907 reports with all three gold labels present, so every case has a T/N/M answer to score against.

## Canonicalization

Registry labels include sparse subtypes (`T3a`, `N2c`, `M1a`, ...) that would make per-subtype scoring mostly noise — some subtypes have single-digit case counts. Both gold labels and model predictions are folded to the major category (`T3a` → `T3`, `N2c` → `N2`) before scoring, applied identically to both sides so the comparison stays fair.

```
T: {T0, T1, T2, T3, T4, TX, unknown}
N: {N0, N1, N2, N3, NX, unknown}
M: {M0, M1, MX, unknown}
```

`TX`/`NX`/`MX` and `unknown` are not the same thing, and the whole abstention analysis depends on keeping them separate. `TX` is a *gold* value — the registry itself says "assessed, indeterminate." `unknown` is a *model behavior* — the model declining to assert anything because the report gives it no basis. Collapsing those into one "don't know" bucket would erase the distinction the D arm is designed to test.

## The four arms

All arms share one output schema (a `value` / `evidence` / `confidence` JSON shape) and one model config — the only thing that varies is the constraint applied to the prompt. Arms are defined once in [`src/stageground/config.py`](../src/stageground/config.py) (`ExperimentArm` + `ARM_CONFIGS`), so prompt text and evidence requirements read from one source instead of branching on arm-name strings scattered through the codebase.

| Arm | Allowed values enforced | Evidence required | `unknown` permitted |
|---|:-:|:-:|:-:|
| `A_zero_shot` | – | – | – |
| `C_constrained` | ✓ | – | – |
| `C_plus_unknown` | ✓ | – | ✓ |
| `D_grounded` | ✓ | ✓ (every asserted value) | ✓ |

Read A → C → C+ → D as one progressive intervention, not four unrelated prompting styles: C adds structure to A, C+ adds permission to abstain on top of C, D adds mandatory evidence binding on top of C+. A fifth historical arm, `B_few_shot`, existed in an earlier pilot and is not part of the main study — see [`pilot_200.md`](pilot_200.md).

## Model configuration

Every experiment records both the *requested* model config and the *effective* one in its `config.json`. They're usually identical, but some providers silently reject settings — an OpenAI reasoning model, for instance, rejects any non-default temperature — and `resolve_effective_model_config` normalizes that up front rather than either sending a doomed request or quietly diverging from what `config.json` claims happened. The main study used `gpt-4o` with no temperature override, `retries=3`, JSON-object output mode.

## Paired design

All four arms run on the *same* 1,000 reports. This matters for two reasons: it lets every comparison use paired statistics (a same-case bootstrap, not two independent samples), and it means any difference between arms is attributable to the prompting intervention, not to sampling variance from evaluating different cases.

## Sampling

`stageground.evaluation.sampling.stratified_sample` is deterministic — same dataframe, same `n`, same seed always returns the identical sample — and stratifies proportionally on which of T/N/M gold labels a report has available (`--pool all-three` restricts to reports with all three, matching the main study; `--pool any` would include partial-label reports for a different kind of run).
