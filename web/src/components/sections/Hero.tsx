import { Container } from "../ui/Container";
import { data } from "@/data/stageground";

const GITHUB_URL = "https://github.com/CG-Brian/stage-ground";

export function Hero() {
  return (
    <section id="top" className="pt-8 pb-4 sm:pt-10">
      <Container wide>
        <p className="font-mono-data text-[11px] tracking-wider uppercase text-accent mb-3">
          Evaluation sandbox · not a clinical product
        </p>
        <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight text-foreground text-balance">
          StageGround
        </h1>
        <p className="mt-1.5 text-lg sm:text-xl font-medium text-muted">
          Source-grounded TNM evaluation sandbox
        </p>
        <p className="mt-3 max-w-2xl text-sm sm:text-base text-muted leading-relaxed">
          Compare how structured outputs, abstention, and evidence binding
          change the behavior of the same LLM on pathology reports.{" "}
          <span className="text-foreground font-medium">
            Correct predictions are not necessarily grounded predictions.
          </span>
        </p>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <a
            href="#sandbox"
            className="inline-flex items-center rounded bg-accent px-4 py-2 text-sm font-medium text-accent-foreground hover:opacity-90 transition-opacity focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Explore the experiment
          </a>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center rounded border border-border-strong px-4 py-2 text-sm font-medium text-foreground hover:border-accent hover:text-accent transition-colors focus-visible:outline-2 focus-visible:outline-accent"
          >
            GitHub
          </a>
          <span className="font-mono-data text-xs text-muted-2">
            {data.metadata.n_cases.toLocaleString()} reports ·{" "}
            {data.metadata.n_predictions.toLocaleString()} predictions ·{" "}
            {data.metadata.n_arms} strategies · {data.metadata.model}
          </span>
        </div>
      </Container>
    </section>
  );
}
