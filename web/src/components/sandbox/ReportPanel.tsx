"use client";

import { useEffect, useRef } from "react";
import { findSpan } from "@/lib/highlight";
import { Tag } from "../ui/Card";
import type { SandboxCase } from "@/data/stageground";

export function ReportPanel({
  sandboxCase,
  highlightEvidence,
}: {
  sandboxCase: SandboxCase;
  /** Evidence string to highlight, or null/undefined to show plain text
   * (compare-all mode, or an abstained arm with no evidence). */
  highlightEvidence?: string | null;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    const mark = markRef.current;
    if (highlightEvidence && mark && container) {
      // Scroll only the report panel itself, not the whole page --
      // `mark.scrollIntoView()` would also scroll ancestor/window scroll
      // containers, yanking the viewport away from the evaluation panel.
      const target = mark.offsetTop - container.clientHeight / 2 + mark.clientHeight / 2;
      container.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
    } else if (container) {
      container.scrollTop = 0;
    }
  }, [sandboxCase.id, highlightEvidence]);

  const span = highlightEvidence ? findSpan(sandboxCase.report, highlightEvidence) : null;
  const evidenceFoundInText = !highlightEvidence || span !== null;

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span
          className="font-mono-data text-xs text-muted-2"
          title={`Source case: ${sandboxCase.caseId}`}
        >
          {sandboxCase.displayId}
        </span>
        <span className="text-sm font-medium text-foreground">
          {sandboxCase.title}
        </span>
        <Tag tone="neutral">Target {sandboxCase.target}</Tag>
        <Tag tone="neutral">Gold {sandboxCase.gold ?? "—"}</Tag>
      </div>

      {highlightEvidence !== undefined && highlightEvidence !== null && !evidenceFoundInText && (
        <p className="mb-2 text-xs font-medium text-warn">
          Evidence span not found in source
        </p>
      )}

      <div
        ref={containerRef}
        className="scroll-thin flex-1 min-h-0 overflow-y-auto rounded border border-border bg-surface-2 p-3.5 font-mono-data text-[12.5px] leading-relaxed text-muted whitespace-pre-wrap"
      >
        {span ? (
          <>
            {span.before}
            <mark
              ref={markRef}
              className="bg-accent-soft text-foreground rounded px-0.5 py-px ring-1 ring-accent/40 not-italic"
            >
              {span.match}
            </mark>
            {span.after}
          </>
        ) : (
          sandboxCase.report
        )}
      </div>
    </div>
  );
}
