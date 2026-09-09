import { Container } from "../ui/Container";
import { SectionHeading } from "../ui/SectionHeading";

export function WhyThisMatters() {
  return (
    <section id="why" className="border-t border-border">
      <Container wide className="py-10 sm:py-12">
        <SectionHeading
          compact
          eyebrow="Why this matters"
          title="A prediction can match the label and still not be grounded."
        />
        <p className="mt-3 max-w-2xl text-sm text-muted leading-relaxed">
          Every axis above — label accuracy, span grounding, semantic
          support, abstention — can move independently. The sandbox above
          shows real committed examples of each failure mode; the sections
          below summarize what that pattern looks like across all 1,000
          reports.
        </p>
      </Container>
    </section>
  );
}
