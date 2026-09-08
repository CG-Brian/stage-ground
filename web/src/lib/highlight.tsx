import type { ReactNode } from "react";

/** Case-insensitive split of `text` around the first occurrence of `needle`,
 * for rendering the cited evidence highlighted inside a report excerpt.
 * Display-only; not the OCR-noise-tolerant grounding check used for scoring. */
export function highlightSpan(text: string, needle: string | null): ReactNode {
  if (!needle) return text;
  const idx = text.toLowerCase().indexOf(needle.toLowerCase());
  if (idx === -1) return text;
  const before = text.slice(0, idx);
  const match = text.slice(idx, idx + needle.length);
  const after = text.slice(idx + needle.length);
  return (
    <>
      {before}
      <mark className="bg-accent-soft text-foreground rounded px-0.5 py-px ring-1 ring-accent/40 not-italic">
        {match}
      </mark>
      {after}
    </>
  );
}
