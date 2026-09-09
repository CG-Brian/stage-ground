import { Container } from "../ui/Container";
import { Card } from "../ui/Card";
import { SectionHeading } from "../ui/SectionHeading";
import { M0ParadoxChart } from "../charts/M0ParadoxChart";
import { M0BridgeButton } from "../sandbox/M0BridgeButton";
import { data } from "@/data/stageground";

export function M0Paradox() {
  const m0 = data.m0;

  return (
    <section id="m0-paradox" className="border-t border-border bg-surface-2/40">
      <Container wide className="py-12 sm:py-14">
        <SectionHeading
          compact
          eyebrow="Centerpiece finding · M-stage"
          title="99% correct does not mean 99% grounded."
        />

        <div className="mt-6 grid gap-3 sm:grid-cols-3">
          <Card className="text-center">
            <p className="text-xs text-muted-2">Among M0 predictions</p>
            <p className="font-mono-data text-3xl sm:text-4xl text-foreground mt-1">
              ~99%
            </p>
            <p className="text-xs text-muted-2 mt-1">label accuracy, all 4 arms</p>
          </Card>
          <Card className="text-center border-accent/30">
            <p className="text-xs text-muted-2">Of those same predictions</p>
            <p className="font-mono-data text-3xl sm:text-4xl text-accent mt-1">
              2–4%
            </p>
            <p className="text-xs text-muted-2 mt-1">semantic-supported rate</p>
          </Card>
          <Card className="text-center">
            <p className="text-xs text-muted-2">Gold M0 prevalence</p>
            <p className="font-mono-data text-3xl sm:text-4xl text-foreground mt-1">
              {(m0.goldM0Fraction * 100).toFixed(1)}%
            </p>
            <p className="text-xs text-muted-2 mt-1">
              of gold M labels ({(m0.goldM1Fraction * 100).toFixed(1)}% M1)
            </p>
          </Card>
        </div>

        <Card className="mt-4">
          <h3 className="text-sm font-medium text-foreground mb-1">
            M0 accuracy vs. semantic-supported M0 rate, by arm
          </h3>
          <M0ParadoxChart />
        </Card>

        <p className="mt-4 max-w-2xl text-sm text-muted leading-relaxed">
          With {(m0.goldM0Fraction * 100).toFixed(1)}% of gold M labels
          already M0, predicting M0 by default is usually correct regardless
          of report content — a pattern{" "}
          <strong className="text-foreground font-medium">
            consistent with prior-driven prediction
          </strong>{" "}
          rather than demonstrated source support.
        </p>
        <div className="mt-3">
          <M0BridgeButton />
        </div>
      </Container>
    </section>
  );
}
