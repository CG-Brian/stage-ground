import { MetricHelp } from "./MetricHelp";

export type StatusTone = "good" | "bad" | "neutral";

const TONE_CLASS: Record<StatusTone, string> = {
  good: "text-good",
  bad: "text-bad",
  neutral: "text-muted-2",
};

const TONE_SYMBOL: Record<StatusTone, string> = {
  good: "✓",
  bad: "✕",
  neutral: "—",
};

export function StatusRow({
  label,
  tone,
  text,
  help,
}: {
  label: string;
  tone: StatusTone;
  text: string;
  help?: { title: string; body: string };
}) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-muted">
        {label}
        {help && <MetricHelp title={help.title} body={help.body} />}
      </span>
      <span className={`font-mono-data font-medium ${TONE_CLASS[tone]}`}>
        {TONE_SYMBOL[tone]} {text}
      </span>
    </div>
  );
}

export function ValueRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between py-1.5 text-sm">
      <span className="text-muted">{label}</span>
      <span className="font-mono-data font-medium text-foreground">
        {value}
      </span>
    </div>
  );
}
