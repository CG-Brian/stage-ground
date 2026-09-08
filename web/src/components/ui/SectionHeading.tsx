export function SectionHeading({
  eyebrow,
  title,
  lede,
  id,
}: {
  eyebrow?: string;
  title: string;
  lede?: string;
  id?: string;
}) {
  return (
    <div className="max-w-2xl">
      {eyebrow && (
        <p className="font-mono-data text-xs tracking-widest uppercase text-accent mb-3">
          {eyebrow}
        </p>
      )}
      <h2
        id={id}
        className="font-serif-display text-3xl sm:text-4xl leading-tight tracking-tight text-foreground scroll-mt-24"
      >
        {title}
      </h2>
      {lede && (
        <p className="mt-4 text-base sm:text-lg text-muted leading-relaxed">
          {lede}
        </p>
      )}
    </div>
  );
}
