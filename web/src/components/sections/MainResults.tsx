import { Container } from "../ui/Container";
import { Card } from "../ui/Card";
import { SectionHeading } from "../ui/SectionHeading";
import { ArmLegend } from "../ui/ArmLegend";
import { MetricBarChart } from "../charts/MetricBarChart";
import { overallByArmOrder } from "@/data/stageground";

export function MainResults() {
  const arms = overallByArmOrder;

  return (
    <section id="results" className="border-b border-border">
      <Container className="py-16 sm:py-20">
        <SectionHeading
          eyebrow="Main results · overall, pooled across T / N / M"
          title="Accuracy and grounding do not rank the arms the same way."
          lede="Highest raw accuracy: C_constrained. Best grounding: D_grounded. These are two different results, not one."
        />

        <div className="mt-6">
          <ArmLegend />
        </div>

        <div className="mt-8 grid gap-x-8 gap-y-10 sm:grid-cols-2">
          <MetricBarChart
            title="Raw accuracy"
            description="Predicted label matches gold, over all evaluable cases."
            goodDirection="higher"
            data={arms.map((a) => ({ id: a.id, value: a.accuracy }))}
          />
          <MetricBarChart
            title="Semantic unsupported rate"
            description="Asserted predictions whose own cited evidence does not justify the label."
            goodDirection="lower"
            data={arms.map((a) => ({ id: a.id, value: a.semanticUnsupportedRate }))}
          />
          <MetricBarChart
            title="Semantic-supported accuracy"
            description="Correct AND grounded AND semantically justified — the strictest joint metric."
            goodDirection="higher"
            data={arms.map((a) => ({ id: a.id, value: a.semanticSupportedAccuracy }))}
          />
          <MetricBarChart
            title="Coverage"
            description="1 − abstention rate: how often the arm asserts a label at all."
            goodDirection="higher"
            domainMax={1}
            data={arms.map((a) => ({ id: a.id, value: a.coverage }))}
          />
        </div>

        <Card className="mt-10 bg-accent-soft/40 border-accent/25">
          <p className="font-serif-display text-lg sm:text-xl text-foreground text-balance">
            C_constrained reaches the highest raw accuracy (
            {(arms[1].accuracy * 100).toFixed(1)}%), but D_grounded has the
            lowest semantic unsupported rate (
            {(arms[3].semanticUnsupportedRate * 100).toFixed(1)}%) and the
            highest semantic-supported accuracy (
            {(arms[3].semanticSupportedAccuracy * 100).toFixed(1)}%).
          </p>
          <p className="mt-2 text-sm text-muted">
            Accuracy and grounding are separate axes here — optimizing for one
            does not automatically optimize the other.
          </p>
        </Card>

        <p className="mt-6 text-xs text-muted-2">
          Source:{" "}
          <code className="font-mono-data">
            results/20260908T022659_gpt-4o_seed42_n1000/metrics.json
          </code>
        </p>
      </Container>
    </section>
  );
}
