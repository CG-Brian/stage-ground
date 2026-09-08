import { Container } from "../ui/Container";
import { Card } from "../ui/Card";
import { SectionHeading } from "../ui/SectionHeading";

export function CoreProblem() {
  return (
    <section id="problem" className="border-b border-border">
      <Container className="py-16 sm:py-20">
        <SectionHeading
          eyebrow="The core problem"
          title="A prediction can match the label and still not be grounded."
          lede="Below is a simplified, illustrative walk-through of the failure mode this project measures — see the case explorer further down for real predictions from the committed experiment."
        />

        <div className="mt-10 grid gap-4 sm:grid-cols-2">
          <Card className="font-mono-data text-sm leading-relaxed">
            <p className="text-xs uppercase tracking-wide text-muted mb-2">
              Pathology report (excerpt)
            </p>
            <p className="text-foreground">&ldquo;…pT3, pN2…&rdquo;</p>
            <p className="text-xs uppercase tracking-wide text-muted mt-5 mb-2">
              Gold label
            </p>
            <p className="text-foreground">M0</p>
          </Card>

          <Card className="font-mono-data text-sm leading-relaxed">
            <p className="text-xs uppercase tracking-wide text-muted mb-2">
              Model prediction
            </p>
            <p className="text-good">M0 ✓ correct</p>
            <p className="text-xs uppercase tracking-wide text-muted mt-5 mb-2">
              Model&rsquo;s cited evidence
            </p>
            <p className="text-foreground">
              &ldquo;No evidence of distant metastasis.&rdquo;
            </p>
            <p className="text-xs uppercase tracking-wide text-bad mt-5 mb-1">
              Problem
            </p>
            <p className="text-bad">
              That sentence does not appear anywhere in the report.
            </p>
          </Card>
        </div>

        <p className="mt-8 font-serif-display text-xl sm:text-2xl text-foreground text-balance">
          Correct label ≠ grounded prediction.
        </p>
        <p className="mt-2 text-sm text-muted-2 italic">
          Illustrative example, not a specific committed case — see the case
          explorer for real, verified examples of this exact pattern.
        </p>
      </Container>
    </section>
  );
}
