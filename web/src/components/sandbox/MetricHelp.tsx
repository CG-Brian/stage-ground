export function MetricHelp({ title, body }: { title: string; body: string }) {
  return (
    <details className="group inline-block relative align-middle ml-1">
      <summary
        className="inline-flex h-3.5 w-3.5 items-center justify-center rounded-full border border-border-strong text-[9px] leading-none text-muted-2 cursor-pointer select-none list-none hover:border-accent hover:text-accent focus-visible:outline-2 focus-visible:outline-accent [&::-webkit-details-marker]:hidden"
        aria-label={`What is ${title.toLowerCase()}?`}
      >
        i
      </summary>
      <div className="absolute z-20 left-0 top-5 w-56 rounded-md border border-border-strong bg-surface p-3 text-xs leading-relaxed text-muted shadow-md">
        <p className="font-medium text-foreground mb-1">{title}</p>
        {body}
      </div>
    </details>
  );
}
