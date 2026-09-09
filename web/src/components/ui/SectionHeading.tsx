export function SectionHeading({
  eyebrow,
  title,
  lede,
  id,
  compact = false,
}: {
  eyebrow?: string;
  title: string;
  lede?: string;
  id?: string;
  compact?: boolean;
}) {
  return (
    <div className="max-w-2xl">
      {eyebrow && (
        <p className="font-mono-data text-[11px] tracking-wider uppercase text-accent mb-2">
          {eyebrow}
        </p>
      )}
      <h2
        id={id}
        className={`font-semibold tracking-tight text-foreground scroll-mt-20 ${
          compact ? "text-xl sm:text-2xl" : "text-2xl sm:text-3xl"
        }`}
      >
        {title}
      </h2>
      {lede && (
        <p className="mt-2.5 text-sm sm:text-base text-muted leading-relaxed">
          {lede}
        </p>
      )}
    </div>
  );
}
