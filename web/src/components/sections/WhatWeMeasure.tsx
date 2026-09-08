import { Container } from "../ui/Container";
import { Card } from "../ui/Card";
import { SectionHeading } from "../ui/SectionHeading";

const MEASURES = [
  {
    tag: "01",
    title: "Accuracy",
    question: "Does the predicted label match gold?",
    detail:
      "The traditional metric. On its own, it says nothing about whether the model actually read the right part of the report.",
  },
  {
    tag: "02",
    title: "Span grounding",
    question: "Does the cited evidence actually appear in the report?",
    detail:
      "A syntactic check — is the model's quoted evidence real, verbatim text from the source document, or fabricated?",
  },
  {
    tag: "03",
    title: "Semantic support",
    question: "Does that evidence actually justify the predicted label?",
    detail:
      "Real text can still fail to support the specific claim made. This is scored by a conservative, rule-based heuristic — not a clinical adjudicator.",
  },
  {
    tag: "04",
    title: "Abstention / coverage",
    question: "Does the model decline to answer when support is insufficient?",
    detail:
      "A model that says 'unknown' instead of guessing trades coverage for reliability. Whether that trade is worthwhile is itself a question this project tests.",
  },
];

export function WhatWeMeasure() {
  return (
    <section id="measures" className="border-b border-border">
      <Container className="py-16 sm:py-20">
        <SectionHeading
          eyebrow="Framework"
          title="Four separate axes, not four names for the same thing."
          lede="StageGround keeps these distinct throughout — an arm can be accurate without being grounded, or grounded without being correct."
        />
        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {MEASURES.map((m) => (
            <Card key={m.title} className="flex flex-col">
              <span className="font-mono-data text-xs text-accent">
                {m.tag}
              </span>
              <h3 className="font-serif-display text-xl text-foreground mt-2">
                {m.title}
              </h3>
              <p className="mt-2 text-sm font-medium text-foreground/90">
                {m.question}
              </p>
              <p className="mt-3 text-sm text-muted leading-relaxed">
                {m.detail}
              </p>
            </Card>
          ))}
        </div>
      </Container>
    </section>
  );
}
