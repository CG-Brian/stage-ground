import { Container } from "../ui/Container";
import { Card } from "../ui/Card";
import { SectionHeading } from "../ui/SectionHeading";
import { data } from "@/data/stageground";

const FACTS = [
  `${data.metadata.n_cases.toLocaleString()} pathology reports, same cases across all arms (paired design)`,
  `Model: ${data.metadata.model}, deterministic seed = ${data.metadata.seed} sampling`,
  `${data.metadata.n_predictions.toLocaleString()} PredictionRecords (${data.metadata.n_cases} cases × ${data.metadata.n_arms} arms × ${data.metadata.n_targets} TNM targets)`,
  `Sampled from an eligible pool of ${data.metadata.eligible_pool.toLocaleString()} reports with complete gold T/N/M labels`,
  "Structured TNM targets, scored against registry gold labels",
  "Semantic support is currently a conservative, rule-based heuristic — not a clinical staging engine",
];

export function Methodology() {
  return (
    <section id="methodology" className="border-b border-border">
      <Container className="py-16 sm:py-20">
        <SectionHeading
          eyebrow="Methodology & limitations"
          title="How this was run, and what it doesn't claim."
        />

        <div className="mt-8 grid gap-8 lg:grid-cols-2">
          <Card>
            <h3 className="font-serif-display text-lg text-foreground mb-4">
              Setup
            </h3>
            <ul className="space-y-2.5 text-sm text-muted">
              {FACTS.map((f) => (
                <li key={f} className="flex gap-2.5">
                  <span aria-hidden className="text-accent shrink-0">
                    ·
                  </span>
                  {f}
                </li>
              ))}
            </ul>
          </Card>

          <Card className="border-accent/25 bg-accent-soft/30">
            <h3 className="font-serif-display text-lg text-foreground mb-4">
              Important limitations
            </h3>
            <ul className="space-y-3 text-sm text-muted leading-relaxed">
              <li>
                <strong className="text-foreground font-medium">
                  Not a clinical staging engine.
                </strong>{" "}
                Semantic support is not equivalent to full human adjudication
                — it is validated against a blinded 80-case human audit, but
                that audit is not a source of clinical ground truth either.
              </li>
              <li>
                <strong className="text-foreground font-medium">
                  Single model, single dataset.
                </strong>{" "}
                Results should not be generalized beyond gpt-4o at this
                prompt/schema version, or beyond this TCGA-derived cohort.
              </li>
              <li>
                <strong className="text-foreground font-medium">
                  Not a diagnostic tool.
                </strong>{" "}
                StageGround is an evaluation methodology, not a TNM staging
                product or decision-support system.
              </li>
              <li>
                <strong className="text-foreground font-medium">
                  Correlational, not causal beyond this ablation.
                </strong>{" "}
                Findings describe this paired prompt intervention; they do
                not establish why a given failure mode occurs.
              </li>
            </ul>
          </Card>
        </div>

        <p className="mt-8 text-sm text-muted-2 max-w-2xl">
          Full write-up, all metric definitions, and the human-audit
          methodology are in the repository README and{" "}
          <code className="font-mono-data text-xs">
            results/20260908T022659_gpt-4o_seed42_n1000/final_scientific_summary.md
          </code>
          .
        </p>
      </Container>
    </section>
  );
}
