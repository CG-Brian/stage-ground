import { Container } from "../ui/Container";
import { Card } from "../ui/Card";
import { SectionHeading } from "../ui/SectionHeading";
import { ArmLegend } from "../ui/ArmLegend";
import { TargetGroupedChart } from "../charts/TargetGroupedChart";
import { targetByArmOrder } from "@/data/stageground";
import { pct } from "@/lib/format";

const TARGET_SUMMARIES = [
  {
    id: "T",
    title: "T — mostly unchanged accuracy",
    body: "Accuracy is roughly flat across C / C+ / D (differences are not statistically distinguishable); D's main advantage here is grounding, not correctness.",
  },
  {
    id: "N",
    title: "N — mostly flat",
    body: "Small or non-significant differences across arms on nearly every metric except abstention/coverage, which move mechanically with each prompt's design.",
  },
  {
    id: "M",
    title: "M — large accuracy swings, uniformly weak grounding",
    body: "The only target with large, significant cross-arm accuracy differences — and semantic-supported accuracy stays near zero for every arm (see the M0 section above).",
  },
];

export function TNMDifferences() {
  return (
    <section id="tnm" className="border-b border-border">
      <Container className="py-16 sm:py-20">
        <SectionHeading
          eyebrow="Target breakdown"
          title="Reliability failures are not uniform across targets."
          lede="The intervention effects above are pooled across T, N, and M — but the three targets behave differently underneath that average."
        />

        <div className="mt-8 grid gap-4 sm:grid-cols-3">
          {TARGET_SUMMARIES.map((t) => (
            <Card key={t.id}>
              <h3 className="font-serif-display text-base text-foreground">
                {t.title}
              </h3>
              <p className="mt-2 text-sm text-muted leading-relaxed">
                {t.body}
              </p>
            </Card>
          ))}
        </div>

        <div className="mt-6">
          <ArmLegend />
        </div>

        <div className="mt-6 grid gap-x-8 gap-y-8 sm:grid-cols-2">
          <Card>
            <h4 className="text-sm font-medium text-foreground mb-1">
              Accuracy by target
            </h4>
            <TargetGroupedChart metric="accuracy" />
          </Card>
          <Card>
            <h4 className="text-sm font-medium text-foreground mb-1">
              Semantic unsupported rate by target
            </h4>
            <TargetGroupedChart metric="semanticUnsupportedRate" />
          </Card>
        </div>

        <div className="mt-10">
          <h3 className="font-serif-display text-lg text-foreground mb-1">
            Overall M-stage metrics, by arm
          </h3>
          <p className="text-xs text-muted mb-4 max-w-2xl">
            These are M-stage performance across <em>all</em> M predictions —
            not the M0-only conditional numbers shown above. Kept separate
            deliberately: mixing the two would misleadingly imply M-stage
            accuracy overall is ~99%, when that figure only ever describes M0
            predictions specifically.
          </p>
          <div className="overflow-x-auto rounded-xl border border-border">
            <table className="w-full text-sm border-collapse min-w-[520px]">
              <thead>
                <tr className="border-b border-border bg-black/[0.02] dark:bg-white/[0.03] text-left text-xs uppercase tracking-wide text-muted">
                  <th className="px-4 py-3 font-medium">Arm</th>
                  <th className="px-4 py-3 font-medium text-right">Accuracy</th>
                  <th className="px-4 py-3 font-medium text-right">Abstention</th>
                  <th className="px-4 py-3 font-medium text-right">
                    Semantic-supported accuracy
                  </th>
                  <th className="px-4 py-3 font-medium text-right">
                    Semantic unsupported
                  </th>
                </tr>
              </thead>
              <tbody className="font-mono-data">
                {targetByArmOrder("M").map((row) => (
                  <tr key={row.id} className="border-b border-border last:border-0">
                    <td className="px-4 py-3 font-sans font-medium text-foreground">
                      {row.label}
                    </td>
                    <td className="px-4 py-3 text-right">{pct(row.accuracy)}</td>
                    <td className="px-4 py-3 text-right">{pct(row.abstentionRate)}</td>
                    <td className="px-4 py-3 text-right text-accent">
                      {pct(row.semanticSupportedAccuracy)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {pct(row.semanticUnsupportedRate)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <p className="mt-6 text-xs text-muted-2">
          Source:{" "}
          <code className="font-mono-data">
            results/20260908T022659_gpt-4o_seed42_n1000/metrics_by_target.json
          </code>
        </p>
      </Container>
    </section>
  );
}
