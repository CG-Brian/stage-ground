import { Container } from "../ui/Container";
import { SectionHeading } from "../ui/SectionHeading";
import { CaseExplorer } from "./CaseExplorer";

export function CaseExplorerSection() {
  return (
    <section id="explorer" className="border-b border-border">
      <Container className="py-16 sm:py-20">
        <SectionHeading
          eyebrow="Real case explorer"
          title="Inspect real predictions from the committed experiment."
          lede="A curated set of examples from the 12,000-prediction run — filter by category, then compare the model's prediction and cited evidence against the actual report excerpt."
        />
        <div className="mt-10">
          <CaseExplorer />
        </div>
        <p className="mt-6 text-xs text-muted-2">
          Source reports are from TCGA (public, de-identified pathology
          reports keyed only by TCGA case barcodes). Source:{" "}
          <code className="font-mono-data">
            results/20260908T022659_gpt-4o_seed42_n1000/predictions.jsonl
          </code>
        </p>
      </Container>
    </section>
  );
}
