/**
 * Typed accessors over the generated result JSON. This file adds no numbers
 * of its own -- every value it exposes was computed by a Python script
 * reading the committed experiment artifacts under `results/` at the repo
 * root:
 *
 *   - scripts/export_frontend_data.py  -> generated/stageground-data.json
 *   - scripts/export_case_examples.py  -> generated/case-examples.json
 *
 * Re-run those two scripts (`uv run python scripts/export_...py` from the
 * repo root) after any change to the committed results; this module and the
 * UI never hand-edit a metric.
 */
import rawData from "./generated/stageground-data.json";
import rawCases from "./generated/case-examples.json";

export type ArmId =
  | "A_zero_shot"
  | "C_constrained"
  | "C_plus_unknown"
  | "D_grounded";

export const ARM_ORDER: ArmId[] = [
  "A_zero_shot",
  "C_constrained",
  "C_plus_unknown",
  "D_grounded",
];

export const ARM_SHORT_LABEL: Record<ArmId, string> = {
  A_zero_shot: "A",
  C_constrained: "C",
  C_plus_unknown: "C+",
  D_grounded: "D",
};

export const ARM_NAME: Record<ArmId, string> = {
  A_zero_shot: "Zero-shot",
  C_constrained: "Constrained",
  C_plus_unknown: "Constrained + Unknown",
  D_grounded: "Grounded",
};

/** One-line subtitle for each arm's tab in the sandbox -- makes the ablation
 * (what changed relative to the previous rung) visible without a separate
 * educational section. */
export const ARM_SUBTITLE: Record<ArmId, string> = {
  A_zero_shot: "Zero-shot",
  C_constrained: "+ structured output",
  C_plus_unknown: "+ abstention",
  D_grounded: "+ evidence binding",
};

export const ARM_COLOR_VAR: Record<ArmId, string> = {
  A_zero_shot: "var(--arm-a)",
  C_constrained: "var(--arm-c)",
  C_plus_unknown: "var(--arm-cplus)",
  D_grounded: "var(--arm-d)",
};

interface ArmOverallMetrics {
  id: ArmId;
  label: string;
  accuracy: number;
  coverage: number;
  abstentionRate: number;
  spanUnsupportedRate: number;
  semanticUnsupportedRate: number;
  semanticSupportedAccuracy: number;
  allowedValueCompliance: number;
}

interface ArmTargetMetrics {
  id: ArmId;
  label: string;
  accuracy: number;
  accuracyOverAsserted: number;
  coverage: number;
  abstentionRate: number;
  spanUnsupportedRate: number;
  semanticUnsupportedRate: number;
  semanticSupportedAccuracy: number;
}

interface BootstrapComparison {
  diff: number;
  ciLow: number;
  ciHigh: number;
  pValue: number;
  nBoot: number;
  significant: boolean;
}

interface M0ArmRow {
  id: ArmId;
  label: string;
  nM0Predictions: number;
  m0Accuracy: number;
  m0ProportionOfAssertedM: number;
  m0SemanticSupportedRate: number;
  m0SpanGroundedRate: number;
}

interface StagegroundData {
  sources: Record<string, string>;
  metadata: {
    n_cases: number;
    n_arms: number;
    n_targets: number;
    n_predictions: number;
    model: string;
    seed: number;
    experiment_id: string;
    eligible_pool: number;
  };
  arms: ArmOverallMetrics[];
  byTarget: Record<"T" | "N" | "M", ArmTargetMetrics[]>;
  evidenceBinding: {
    semanticUnsupportedRate: BootstrapComparison;
    semanticSupportedAccuracy: BootstrapComparison;
    coverage: BootstrapComparison;
  };
  abstentionOnly: {
    abstentionRate: BootstrapComparison;
    accuracy: BootstrapComparison;
    semanticUnsupportedRate: BootstrapComparison;
    semanticSupportedAccuracy: BootstrapComparison;
  };
  m0: {
    goldM0Fraction: number;
    goldM1Fraction: number;
    byArm: M0ArmRow[];
  };
}

export const data = rawData as StagegroundData;
export const armById = (id: ArmId) => data.arms.find((a) => a.id === id)!;

/** Overall metrics for all 4 arms, in canonical A/C/C+/D order. */
export const overallByArmOrder = ARM_ORDER.map(
  (id) => data.arms.find((a) => a.id === id)!
);

export function targetByArmOrder(target: "T" | "N" | "M") {
  const rows = data.byTarget[target];
  return ARM_ORDER.map((id) => rows.find((r) => r.id === id)!);
}

// --- Sandbox case data --------------------------------------------------

export type SandboxCategory =
  | "correct_grounded"
  | "correct_unsupported"
  | "wrong_grounded"
  | "abstained"
  | "m0_unsupported"
  | "span_not_supporting"
  | "evidence_binding_change"
  | "c_asserts_d_abstains";

/**
 * Every field here comes directly from a committed `PredictionRecord`
 * (`results/20260908T022659_gpt-4o_seed42_n1000/predictions.jsonl`) with no
 * re-derivation: `evidenceSpanFound`/`semanticSupport` are `null` exactly
 * when the backend recorded them as `None` (abstained predictions only --
 * see `stageground.evaluation.records` docstring), and `correct` is the
 * backend's own `prediction == ground_truth` comparison (still meaningful
 * when abstained: "unknown" never equals a real TNM value, so an abstained
 * prediction is simply `correct: false`, not undefined).
 */
export interface PredictionView {
  prediction: string;
  evidence: string | null;
  correct: boolean;
  abstained: boolean;
  evidenceSpanFound: boolean | null;
  semanticSupport: boolean | null;
}

export interface SandboxCase {
  id: string;
  /** Sequential, human-readable label ("Case 01") -- never the raw TCGA
   * barcode, which stays in `caseId` for provenance only. */
  displayId: string;
  /** Short, scientifically-neutral, category-derived title, e.g.
   * "M0 base-rate pattern" -- see scripts/export_case_examples.py's
   * TITLE_PRIORITY for the deterministic selection rule when a case
   * matches more than one category. */
  title: string;
  caseId: string;
  target: "T" | "N" | "M";
  gold: string | null;
  report: string;
  categories: SandboxCategory[];
  arms: Record<ArmId, PredictionView>;
}

interface CaseExamplesFile {
  cases: SandboxCase[];
  defaultCaseId: string;
}

const caseExamplesFile = rawCases as CaseExamplesFile;
export const sandboxCases: SandboxCase[] = caseExamplesFile.cases;
export const DEFAULT_SANDBOX_CASE_ID = caseExamplesFile.defaultCaseId;

export const sandboxCaseById = (id: string) =>
  sandboxCases.find((c) => c.id === id);

export const CATEGORY_LABEL: Record<SandboxCategory, string> = {
  correct_grounded: "Correct + grounded",
  correct_unsupported: "Correct + unsupported",
  wrong_grounded: "Wrong + grounded",
  abstained: "Abstained",
  m0_unsupported: "M0: correct but unsupported",
  span_not_supporting: "Evidence exists but doesn't support label",
  evidence_binding_change: "Evidence binding changes behavior",
  c_asserts_d_abstains: "C asserts, D abstains",
};

export const CATEGORY_ORDER: SandboxCategory[] = [
  "correct_grounded",
  "correct_unsupported",
  "wrong_grounded",
  "abstained",
  "m0_unsupported",
  "span_not_supporting",
  "evidence_binding_change",
  "c_asserts_d_abstains",
];

// --- Inline metric definitions (used in the sandbox, not a standalone
// marketing section) -----------------------------------------------------

export const METRIC_HELP = {
  labelMatch: {
    title: "Label match",
    body: "Did the model's predicted TNM value equal the gold (registry) label for this target?",
  },
  spanGrounding: {
    title: "Span grounding",
    body: "Does the model's cited evidence appear verbatim (OCR-noise-tolerant) in the source report? A syntactic check only -- it says nothing about whether the text actually supports the prediction.",
  },
  semanticSupport: {
    title: "Semantic support",
    body: "Does that evidence actually justify the predicted label? Scored by a conservative, rule-based heuristic -- not a clinical adjudicator. Always false when the span itself isn't found.",
  },
  abstention: {
    title: "Abstention",
    body: "Did the model decline to assert a label (\"unknown\") instead of guessing without support?",
  },
} as const;
