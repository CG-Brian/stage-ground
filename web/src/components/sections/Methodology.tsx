import { Container } from "../ui/Container";
import { SectionHeading } from "../ui/SectionHeading";
import { data } from "@/data/stageground";

const GITHUB_METHODOLOGY_URL =
  "https://github.com/CG-Brian/stage-ground/blob/main/results/20260908T022659_gpt-4o_seed42_n1000/final_scientific_summary.md";

const FACTS: [string, string][] = [
  ["Model", data.metadata.model],
  ["Cases", `${data.metadata.n_cases.toLocaleString()} paired reports`],
  ["Predictions", data.metadata.n_predictions.toLocaleString()],
  ["Targets", "T / N / M"],
  ["Sampling", `seed ${data.metadata.seed}, deterministic`],
  ["Evaluation", "label accuracy, span grounding, semantic support, abstention"],
];

export function Methodology() {
  return (
    <section id="methodology" className="border-t border-border">
      <Container wide className="py-12 sm:py-14">
        <SectionHeading compact eyebrow="Methodology" title="How this was run" />

        <dl className="mt-5 grid sm:grid-cols-2 gap-x-8 gap-y-2 max-w-2xl">
          {FACTS.map(([k, v]) => (
            <div key={k} className="flex justify-between border-b border-border py-1.5 text-sm">
              <dt className="text-muted">{k}</dt>
              <dd className="font-mono-data text-foreground text-right">{v}</dd>
            </div>
          ))}
        </dl>

        <p className="mt-4 text-xs text-muted-2 max-w-2xl">
          Semantic support is a conservative, rule-based heuristic — not a
          clinical staging engine or a substitute for human adjudication.
          Results describe gpt-4o on this dataset only and should not be
          generalized to other models.
        </p>

        <a
          href={GITHUB_METHODOLOGY_URL}
          target="_blank"
          rel="noreferrer"
          className="mt-3 inline-block text-sm font-medium text-accent hover:underline underline-offset-4"
        >
          Read full methodology on GitHub →
        </a>
      </Container>
    </section>
  );
}
