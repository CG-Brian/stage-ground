export function Card({
  children,
  className = "",
  as: As = "div",
}: {
  children: React.ReactNode;
  className?: string;
  as?: React.ElementType;
}) {
  return (
    <As
      className={`rounded-md border border-border bg-surface p-5 ${className}`}
    >
      {children}
    </As>
  );
}

export function Tag({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "good" | "bad" | "warn" | "accent";
}) {
  const toneClass = {
    neutral: "bg-surface-2 text-muted border-border",
    good: "text-good border-good/25 bg-good-soft",
    bad: "text-bad border-bad/25 bg-bad-soft",
    warn: "text-warn border-warn/25 bg-warn-soft",
    accent: "text-accent border-accent/25 bg-accent-soft",
  }[tone];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded border px-2 py-0.5 text-xs font-medium font-mono-data ${toneClass}`}
    >
      {children}
    </span>
  );
}

