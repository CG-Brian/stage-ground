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
      className={`rounded-xl border border-border bg-surface p-6 ${className}`}
    >
      {children}
    </As>
  );
}

export function Pill({
  children,
  tone = "neutral",
}: {
  children: React.ReactNode;
  tone?: "neutral" | "good" | "bad" | "accent";
}) {
  const toneClass = {
    neutral: "bg-black/5 dark:bg-white/10 text-muted border-border",
    good: "text-good border-good/30 bg-good/10",
    bad: "text-bad border-bad/30 bg-bad/10",
    accent: "text-accent border-accent/30 bg-accent-soft",
  }[tone];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium font-mono-data ${toneClass}`}
    >
      {children}
    </span>
  );
}
