"use client";

import { useMemo, useState } from "react";
import {
  ARM_NAME,
  CASE_CATEGORY_BLURB,
  CASE_CATEGORY_LABEL,
  caseExamples,
  type CaseCategory,
} from "@/data/stageground";
import { Pill } from "../ui/Card";
import { highlightSpan } from "@/lib/highlight";

const FILTERS: { id: CaseCategory | "all"; label: string }[] = [
  { id: "all", label: "All" },
  { id: "correct_grounded", label: "Correct + grounded" },
  { id: "correct_unsupported", label: "Correct + unsupported" },
  { id: "wrong_grounded", label: "Wrong + grounded" },
  { id: "abstained", label: "Abstained" },
  { id: "m0_unsupported", label: "M0 unsupported" },
];

export function CaseExplorer() {
  const [filter, setFilter] = useState<CaseCategory | "all">("all");
  const filtered = useMemo(
    () =>
      filter === "all"
        ? caseExamples
        : caseExamples.filter((c) => c.category === filter),
    [filter]
  );
  const [selectedId, setSelectedId] = useState(caseExamples[0]?.id);
  const selected =
    filtered.find((c) => c.id === selectedId) ?? filtered[0] ?? null;

  return (
    <div>
      <div
        role="group"
        aria-label="Filter cases by category"
        className="flex flex-wrap gap-2 mb-6"
      >
        {FILTERS.map((f) => (
          <button
            key={f.id}
            type="button"
            aria-pressed={filter === f.id}
            onClick={() => {
              setFilter(f.id);
              const next =
                f.id === "all"
                  ? caseExamples
                  : caseExamples.filter((c) => c.category === f.id);
              if (next.length) setSelectedId(next[0].id);
            }}
            className={`rounded-full border px-3.5 py-1.5 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-accent ${
              filter === f.id
                ? "bg-foreground text-background border-foreground"
                : "border-border-strong text-muted hover:text-foreground hover:border-foreground/40"
            }`}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <div
          role="list"
          aria-label="Cases"
          className="flex lg:flex-col gap-2 overflow-x-auto lg:overflow-visible pb-2 lg:pb-0 -mx-1 px-1 lg:mx-0 lg:px-0"
        >
          {filtered.map((c) => (
            <button
              key={c.id}
              role="listitem"
              aria-current={selected?.id === c.id}
              onClick={() => setSelectedId(c.id)}
              className={`shrink-0 lg:shrink text-left rounded-lg border px-3.5 py-2.5 w-56 lg:w-auto transition-colors focus-visible:outline-2 focus-visible:outline-accent ${
                selected?.id === c.id
                  ? "border-accent bg-accent-soft/50"
                  : "border-border bg-surface hover:border-border-strong"
              }`}
            >
              <p className="text-xs font-mono-data text-muted-2">
                {c.target} · {ARM_NAME[c.arm]}
              </p>
              <p className="text-sm font-medium text-foreground mt-0.5">
                {CASE_CATEGORY_LABEL[c.category]}
              </p>
            </button>
          ))}
        </div>

        {selected && <CaseDetail case={selected} />}
      </div>
    </div>
  );
}

function CaseDetail({ case: c }: { case: (typeof caseExamples)[number] }) {
  return (
    <div className="rounded-xl border border-border bg-surface p-5 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <p className="font-serif-display text-lg text-foreground">
            {CASE_CATEGORY_LABEL[c.category]}
          </p>
          <p className="text-xs text-muted mt-0.5">
            {CASE_CATEGORY_BLURB[c.category]}
          </p>
        </div>
        <span className="font-mono-data text-[11px] text-muted-2 shrink-0">
          {c.caseId.split(".")[0]}
        </span>
      </div>

      <div className="flex flex-wrap gap-2 mb-5">
        <Pill tone="neutral">Target {c.target}</Pill>
        <Pill tone="neutral">{ARM_NAME[c.arm]}</Pill>
        <Pill tone={c.correct ? "good" : "bad"}>
          {c.correct ? "Correct" : "Incorrect"}
        </Pill>
        <Pill tone={c.abstained ? "accent" : "neutral"}>
          {c.abstained ? "Abstained" : "Asserted"}
        </Pill>
        {!c.abstained && (
          <>
            <Pill tone={c.evidenceSpanFound ? "good" : "bad"}>
              Span {c.evidenceSpanFound ? "found" : "not found"}
            </Pill>
            <Pill tone={c.semanticSupport ? "good" : "bad"}>
              Semantic {c.semanticSupport ? "supported" : "unsupported"}
            </Pill>
          </>
        )}
      </div>

      <dl className="grid grid-cols-2 gap-4 mb-5 text-sm">
        <div>
          <dt className="text-xs uppercase tracking-wide text-muted">
            Gold label
          </dt>
          <dd className="font-mono-data text-foreground mt-0.5">
            {c.groundTruth ?? "—"}
          </dd>
        </div>
        <div>
          <dt className="text-xs uppercase tracking-wide text-muted">
            Model prediction
          </dt>
          <dd className="font-mono-data text-foreground mt-0.5">
            {c.prediction}
          </dd>
        </div>
      </dl>

      {c.evidence && (
        <div className="mb-5">
          <p className="text-xs uppercase tracking-wide text-muted mb-1.5">
            Model&rsquo;s cited evidence
          </p>
          <p className="font-mono-data text-sm text-foreground bg-black/[0.03] dark:bg-white/[0.05] rounded-lg px-3 py-2 leading-relaxed">
            &ldquo;{c.evidence}&rdquo;
          </p>
        </div>
      )}

      <div>
        <p className="text-xs uppercase tracking-wide text-muted mb-1.5">
          Source report excerpt{c.excerptTruncated ? " (windowed around evidence)" : ""}
        </p>
        <div className="max-h-72 overflow-y-auto font-mono-data text-xs leading-relaxed text-muted whitespace-pre-wrap rounded-lg border border-border p-3">
          {highlightSpan(c.reportExcerpt, c.evidence)}
        </div>
      </div>
    </div>
  );
}
