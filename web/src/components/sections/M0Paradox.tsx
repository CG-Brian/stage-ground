import { Container } from "../ui/Container";
import { Card } from "../ui/Card";
import { SectionHeading } from "../ui/SectionHeading";
import { M0ParadoxChart } from "../charts/M0ParadoxChart";
import { data } from "@/data/stageground";

export function M0Paradox() {
  const m0 = data.m0;

  return (
    <section id="m0-paradox" className="border-b border-border bg-black/[0.015] dark:bg-white/[0.02]">
      <Container className="py-16 sm:py-20">
        <SectionHeading
          eyebrow="Centerpiece finding · M-stage"
          title="99% correct does not mean 99% grounded."
          lede="Across all four prompting strategies, M0 predictions were nearly always label-correct, while only a small fraction were supported by evidence that actually implied M0."
        />

        <div className="mt-10 grid gap-4 sm:grid-cols-3">
          <Card className="text-center">
            <p className="text-xs uppercase tracking-wide text-muted">
              M0 prediction accuracy
            </p>
            <p className="font-mono-data text-4xl sm:text-5xl text-foreground mt-2">
              ~99%
            </p>
            <p className="text-xs text-muted-2 mt-2">across all four arms</p>
          </Card>
          <Card className="text-center border-accent/30">
            <p className="text-xs uppercase tracking-wide text-muted">
              Semantic support
            </p>
            <p className="font-mono-data text-4xl sm:text-5xl text-accent mt-2">
              2–4%
            </p>
            <p className="text-xs text-muted-2 mt-2">
              of those same M0 predictions
            </p>
          </Card>
          <Card className="text-center">
            <p className="text-xs uppercase tracking-wide text-muted">
              Gold M0 prevalence
            </p>
            <p className="font-mono-data text-4xl sm:text-5xl text-foreground mt-2">
              {(m0.goldM0Fraction * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-2 mt-2">
              of gold M labels in this cohort ({(m0.goldM1Fraction * 100).toFixed(1)}% M1)
            </p>
          </Card>
        </div>

        <Card className="mt-8">
          <h3 className="font-serif-display text-lg text-foreground mb-1">
            M0 accuracy vs. semantic-supported M0 rate, by arm
          </h3>
          <p className="text-xs text-muted mb-4">
            Gray bars: how often an M0 prediction matches gold. Accent bars:
            how often the model&rsquo;s own cited evidence actually implies M0.
          </p>
          <M0ParadoxChart />
        </Card>

        <div className="mt-8 max-w-2xl">
          <p className="text-sm text-muted leading-relaxed">
            Because {(m0.goldM0Fraction * 100).toFixed(1)}% of gold M labels
            are already M0, predicting M0 by default is correct most of the
            time regardless of what the report says. This pattern is{" "}
            <strong className="text-foreground font-medium">
              consistent with prior-driven prediction
            </strong>{" "}
            — label agreement without source support — rather than evidence
            that the model reasoned from the report to reach M0.
          </p>
          <p className="mt-3 text-xs text-muted-2">
            We do not call this &ldquo;guessing&rdquo; as a factual claim: the
            model may rely on real but uncited priors, partial reasoning, or
            evidence categories this heuristic does not recognize as
            M-relevant. What the data show is the gap between label agreement
            and demonstrated source support.
          </p>
        </div>

        <p className="mt-6 text-xs text-muted-2">
          Source:{" "}
          <code className="font-mono-data">
            results/20260908T022659_gpt-4o_seed42_n1000/extra_analysis/m0_analysis.csv
          </code>{" "}
          and{" "}
          <code className="font-mono-data">gold_label_distribution.csv</code>
        </p>
      </Container>
    </section>
  );
}
