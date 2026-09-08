import { Container } from "../ui/Container";
import { SectionHeading } from "../ui/SectionHeading";
import { ARM_COLOR_VAR } from "@/data/stageground";

const RUNGS = [
  {
    id: "A_zero_shot" as const,
    letter: "A",
    name: "Zero-shot",
    desc: "Minimal constraints — the model returns whatever labels and evidence it produces unprompted.",
  },
  {
    id: "C_constrained" as const,
    letter: "C",
    name: "Constrained",
    desc: "Adds allowed-value / structured-output constraints — the model must choose from valid TNM values.",
    intervention: "+ structured allowed outputs",
  },
  {
    id: "C_plus_unknown" as const,
    letter: "C+",
    name: "Constrained + Unknown",
    desc: "Same structured constraint, plus explicit permission to answer “unknown” instead of guessing.",
    intervention: "+ allow unknown / abstention",
  },
  {
    id: "D_grounded" as const,
    letter: "D",
    name: "Grounded",
    desc: "Structured output + abstention + mandatory evidence binding for every non-abstained prediction.",
    intervention: "+ require source evidence",
  },
];

export function InterventionLadder() {
  return (
    <section id="ladder" className="border-b border-border">
      <Container className="py-16 sm:py-20">
        <SectionHeading
          eyebrow="Experimental design"
          title="One model, four prompting strategies, one ablation."
          lede="The same gpt-4o model was evaluated under each strategy on the same paired 1,000-case dataset. Read A → C → C+ → D as a progressive intervention ladder, not four unrelated models."
        />

        <div className="mt-10 flex flex-col">
          {RUNGS.map((r, i) => (
            <div key={r.id}>
              {i > 0 && (
                <div className="flex items-center gap-3 pl-6 py-2 text-xs text-muted font-mono-data">
                  <span aria-hidden className="text-accent">
                    ↓
                  </span>
                  {r.intervention}
                </div>
              )}
              <div className="flex items-start gap-4 rounded-xl border border-border bg-surface p-5">
                <div
                  className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full font-mono-data text-sm font-semibold text-white"
                  style={{ background: ARM_COLOR_VAR[r.id] }}
                  aria-hidden
                >
                  {r.letter}
                </div>
                <div>
                  <h3 className="font-serif-display text-lg text-foreground">
                    {r.name}
                    <span className="ml-2 font-mono-data text-xs text-muted-2 align-middle">
                      {r.letter !== r.name ? `(${r.letter})` : null}
                    </span>
                  </h3>
                  <p className="mt-1 text-sm text-muted leading-relaxed max-w-2xl">
                    {r.desc}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>

        <p className="mt-8 text-sm text-muted-2 max-w-2xl">
          The experiment asks three questions: do structured outputs improve
          apparent accuracy? Does allowing abstention reduce unsupported
          predictions? Does requiring evidence binding improve grounded
          reliability?
        </p>
      </Container>
    </section>
  );
}
