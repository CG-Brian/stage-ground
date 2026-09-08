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

export type CaseCategory =
  | "correct_grounded"
  | "correct_unsupported"
  | "wrong_grounded"
  | "abstained"
  | "m0_unsupported";

export interface CaseExample {
  id: string;
  category: CaseCategory;
  caseId: string;
  arm: ArmId;
  target: "T" | "N" | "M";
  groundTruth: string | null;
  prediction: string;
  evidence: string | null;
  reportExcerpt: string;
  excerptTruncated: boolean;
  evidenceSpanFound: boolean | null;
  semanticSupport: boolean | null;
  correct: boolean;
  abstained: boolean;
  errors: string[];
}

export const data = rawData as StagegroundData;
export const caseExamples = (rawCases as { cases: CaseExample[] }).cases;

export const armById = (id: ArmId) => data.arms.find((a) => a.id === id)!;

/** Convenience: overall metrics for all 4 arms, in canonical A/C/C+/D order. */
export const overallByArmOrder = ARM_ORDER.map(
  (id) => data.arms.find((a) => a.id === id)!
);

export function targetByArmOrder(target: "T" | "N" | "M") {
  const rows = data.byTarget[target];
  return ARM_ORDER.map((id) => rows.find((r) => r.id === id)!);
}

export const CASE_CATEGORY_LABEL: Record<CaseCategory, string> = {
  correct_grounded: "Correct + grounded",
  correct_unsupported: "Correct + unsupported",
  wrong_grounded: "Wrong + grounded",
  abstained: "Abstained",
  m0_unsupported: "M0: correct, unsupported",
};

export const CASE_CATEGORY_BLURB: Record<CaseCategory, string> = {
  correct_grounded:
    "The predicted label matches gold, and the model's cited evidence both appears in the report and actually implies that label.",
  correct_unsupported:
    "The predicted label matches gold, but the model's own cited evidence doesn't actually justify it.",
  wrong_grounded:
    "The model cited real, on-point evidence — but the predicted label still doesn't match gold.",
  abstained:
    "The model declined to assert a label rather than guess without support.",
  m0_unsupported:
    "An M0 prediction that is technically correct against gold, but not backed by evidence that specifically implies M0.",
};
