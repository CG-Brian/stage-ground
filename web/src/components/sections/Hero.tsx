import { Container } from "../ui/Container";
import { data } from "@/data/stageground";

const GITHUB_URL = "https://github.com/CG-Brian/stage-ground";

const STATS: { label: string; value: string }[] = [
  { label: "pathology reports", value: data.metadata.n_cases.toLocaleString() },
  { label: "predictions", value: data.metadata.n_predictions.toLocaleString() },
  { label: "prompting strategies", value: String(data.metadata.n_arms) },
  { label: "TNM targets", value: String(data.metadata.n_targets) },
];

export function Hero() {
  return (
    <section id="top" className="border-b border-border">
      <Container className="pt-16 pb-14 sm:pt-24 sm:pb-20">
        <p className="font-mono-data text-xs tracking-widest uppercase text-accent mb-5">
          Research evaluation · not a clinical product
        </p>
        <h1 className="font-serif-display text-4xl sm:text-6xl leading-[1.05] tracking-tight text-foreground max-w-3xl text-balance">
          When &ldquo;correct&rdquo; isn&rsquo;t grounded.
        </h1>
        <p className="mt-6 max-w-2xl text-lg sm:text-xl text-muted leading-relaxed">
          StageGround evaluates whether clinical LLM predictions are actually{" "}
          <em className="text-foreground not-italic border-b border-accent/50">
            supported by the source pathology report
          </em>{" "}
          — not just whether they match the label.
        </p>

        <div className="mt-10 flex flex-wrap items-center gap-3">
          <a
            href="#results"
            className="inline-flex items-center rounded-full bg-accent px-5 py-2.5 text-sm font-medium text-accent-foreground hover:opacity-90 transition-opacity focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            Explore results
          </a>
          <a
            href="#methodology"
            className="inline-flex items-center rounded-full border border-border-strong px-5 py-2.5 text-sm font-medium text-foreground hover:border-accent hover:text-accent transition-colors focus-visible:outline-2 focus-visible:outline-accent"
          >
            View methodology
          </a>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center px-2 py-2.5 text-sm font-medium text-muted hover:text-foreground transition-colors focus-visible:outline-2 focus-visible:outline-accent rounded-full"
          >
            View on GitHub →
          </a>
        </div>

        <dl className="mt-14 grid grid-cols-2 sm:grid-cols-4 gap-6 sm:gap-8 border-t border-border pt-8">
          {STATS.map((s) => (
            <div key={s.label}>
              <dt className="text-xs text-muted uppercase tracking-wide">
                {s.label}
              </dt>
              <dd className="font-mono-data text-2xl sm:text-3xl text-foreground mt-1">
                {s.value}
              </dd>
            </div>
          ))}
        </dl>
        <p className="mt-4 text-xs text-muted-2">
          Paired design — same {data.metadata.n_cases.toLocaleString()} reports,
          same model ({data.metadata.model}), across all four strategies. Source:{" "}
          <code className="font-mono-data">{data.sources.config}</code>
        </p>
      </Container>
    </section>
  );
}
