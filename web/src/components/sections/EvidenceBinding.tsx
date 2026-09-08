import { Container } from "../ui/Container";
import { Card } from "../ui/Card";
import { SectionHeading } from "../ui/SectionHeading";
import { data } from "@/data/stageground";
import { pLabel, pp } from "@/lib/format";

function ResultRow({
  label,
  value,
  tone,
  detail,
}: {
  label: string;
  value: string;
  tone: "good" | "bad";
  detail: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-3 border-b border-border last:border-0">
      <div>
        <p className="text-sm text-foreground font-medium">{label}</p>
        <p className="text-xs text-muted mt-0.5">{detail}</p>
      </div>
      <span
        className={`font-mono-data text-lg shrink-0 ${
          tone === "good" ? "text-good" : "text-bad"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

export function EvidenceBinding() {
  const eb = data.evidenceBinding;
  const ab = data.abstentionOnly;

  return (
    <section id="evidence-binding" className="border-b border-border">
      <Container className="py-16 sm:py-20">
        <SectionHeading
          eyebrow="Strongest finding · C+ → D"
          title="Evidence binding improves reliability beyond abstention alone."
          lede="D_grounded is not simply C_plus_unknown abstaining more. Coverage went up, not down, while grounding improved."
        />

        <div className="mt-10 grid gap-6 lg:grid-cols-2">
          <Card>
            <p className="text-xs uppercase tracking-wide text-muted mb-1">
              C_plus_unknown → D_grounded (overall, paired bootstrap, 95% CI)
            </p>
            <ResultRow
              label="Semantic unsupported rate"
              value={pp(eb.semanticUnsupportedRate.diff)}
              tone="good"
              detail={`95% CI ${(eb.semanticUnsupportedRate.ciLow * 100).toFixed(2)} to ${(eb.semanticUnsupportedRate.ciHigh * 100).toFixed(2)} pp, ${pLabel(eb.semanticUnsupportedRate.pValue, eb.semanticUnsupportedRate.nBoot)}`}
            />
            <ResultRow
              label="Semantic-supported accuracy"
              value={pp(eb.semanticSupportedAccuracy.diff)}
              tone="good"
              detail={`95% CI ${(eb.semanticSupportedAccuracy.ciLow * 100).toFixed(2)} to ${(eb.semanticSupportedAccuracy.ciHigh * 100).toFixed(2)} pp, ${pLabel(eb.semanticSupportedAccuracy.pValue, eb.semanticSupportedAccuracy.nBoot)}`}
            />
            <ResultRow
              label="Coverage"
              value={pp(eb.coverage.diff)}
              tone="good"
              detail={`95% CI ${(eb.coverage.ciLow * 100).toFixed(2)} to ${(eb.coverage.ciHigh * 100).toFixed(2)} pp, ${pLabel(eb.coverage.pValue, eb.coverage.nBoot)}`}
            />
            <p className="mt-4 font-serif-display text-base text-foreground text-balance">
              Grounding improved <em className="not-italic text-accent">and</em>{" "}
              coverage rose — D is not hiding behind lower coverage.
            </p>
          </Card>

          <Card className="bg-black/[0.02] dark:bg-white/[0.03]">
            <p className="text-xs uppercase tracking-wide text-muted mb-1">
              C_constrained → C_plus_unknown (abstention alone, overall)
            </p>
            <ResultRow
              label="Abstention rate"
              value={pp(ab.abstentionRate.diff)}
              tone="bad"
              detail="Coverage drops sharply — the model answers far less often."
            />
            <ResultRow
              label="Raw accuracy"
              value={pp(ab.accuracy.diff)}
              tone="bad"
              detail={pLabel(ab.accuracy.pValue, ab.accuracy.nBoot)}
            />
            <ResultRow
              label="Semantic unsupported rate"
              value={pp(ab.semanticUnsupportedRate.diff)}
              tone="good"
              detail="Lower unsupported rate among what's left asserted."
            />
            <ResultRow
              label="Semantic-supported accuracy"
              value={pp(ab.semanticSupportedAccuracy.diff)}
              tone="bad"
              detail={pLabel(ab.semanticSupportedAccuracy.pValue, ab.semanticSupportedAccuracy.nBoot)}
            />
            <p className="mt-4 font-serif-display text-base text-foreground text-balance">
              Abstention alone is not a free reliability win.
            </p>
          </Card>
        </div>

        <p className="mt-6 text-xs text-muted-2 max-w-2xl">
          This is a paired prompt-intervention comparison on one model
          (gpt-4o) and one dataset — it shows that evidence binding, not
          abstention by itself, moved semantic-supported accuracy here. It
          does not establish a general causal claim beyond this setting.
        </p>
        <p className="mt-2 text-xs text-muted-2">
          Source:{" "}
          <code className="font-mono-data">
            results/20260908T022659_gpt-4o_seed42_n1000/bootstrap.json
          </code>{" "}
          (n_boot = {eb.coverage.nBoot.toLocaleString()})
        </p>
      </Container>
    </section>
  );
}
