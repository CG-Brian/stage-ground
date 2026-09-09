import { Container } from "../ui/Container";
import { Card } from "../ui/Card";
import { SectionHeading } from "../ui/SectionHeading";
import { ArmLegend } from "../ui/ArmLegend";
import { ComparisonScatter } from "../charts/ComparisonScatter";
import { TargetGroupedChart } from "../charts/TargetGroupedChart";
import { EvidenceBindingResult } from "./EvidenceBindingResult";
import { armById } from "@/data/stageground";

export function AggregateResults() {
  const c = armById("C_constrained");
  const d = armById("D_grounded");

  return (
    <section id="results" className="border-t border-border">
      <Container wide className="py-12 sm:py-14">
        <SectionHeading
          compact
          eyebrow="Across 1,000 reports"
          title="What happens across 1,000 reports?"
          lede="Accuracy and grounding rank the strategies differently."
        />

        <div className="mt-6">
          <ArmLegend />
        </div>

        <div className="mt-4 grid lg:grid-cols-[1fr_auto] gap-6 items-start">
          <Card>
            <ComparisonScatter />
          </Card>
          <div className="grid grid-cols-2 lg:grid-cols-1 gap-3 lg:w-48">
            <div className="rounded border border-border p-3">
              <p className="text-[11px] text-muted-2">Highest raw accuracy</p>
              <p className="font-mono-data text-lg text-foreground">C</p>
              <p className="font-mono-data text-sm text-muted">
                {(c.accuracy * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded border border-accent/30 bg-accent-soft/40 p-3">
              <p className="text-[11px] text-muted-2">Best grounding</p>
              <p className="font-mono-data text-lg text-accent">D</p>
              <p className="font-mono-data text-sm text-muted">
                {((1 - d.semanticUnsupportedRate) * 100).toFixed(1)}% grounded
              </p>
            </div>
          </div>
        </div>

        <div className="mt-8 grid lg:grid-cols-2 gap-8">
          <EvidenceBindingResult />
          <div>
            <h3 className="text-sm font-medium text-foreground mb-1">
              Semantic unsupported rate, by target
            </h3>
            <p className="text-xs text-muted-2 mb-3">
              M drives most of the cross-arm difference; T and N are flatter.
            </p>
            <TargetGroupedChart metric="semanticUnsupportedRate" />
          </div>
        </div>

        <p className="mt-6 text-xs text-muted-2 max-w-2xl">
          M-stage semantic-supported accuracy stays at 1.7–2.5% across{" "}
          <em>all</em> M predictions, not just M0 (see M0 paradox above for
          the M0-specific breakdown). Full tables:{" "}
          <code className="font-mono-data">
            results/20260908T022659_gpt-4o_seed42_n1000/metrics_by_target.json
          </code>
        </p>
      </Container>
    </section>
  );
}
