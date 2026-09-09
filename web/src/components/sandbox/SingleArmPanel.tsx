import {
  ARM_NAME,
  ARM_ORDER,
  ARM_SHORT_LABEL,
  ARM_SUBTITLE,
  METRIC_HELP,
  type ArmId,
  type SandboxCase,
} from "@/data/stageground";
import { StatusRow, ValueRow, type StatusTone } from "./StatusRow";

export function ArmTabs({
  activeArm,
  onSelect,
}: {
  activeArm: ArmId;
  onSelect: (arm: ArmId) => void;
}) {
  return (
    <div role="tablist" aria-label="Prompting strategy" className="grid grid-cols-4 gap-1.5">
      {ARM_ORDER.map((id) => (
        <button
          key={id}
          role="tab"
          aria-selected={activeArm === id}
          onClick={() => onSelect(id)}
          className={`rounded border px-2 py-1.5 text-center transition-colors focus-visible:outline-2 focus-visible:outline-accent ${
            activeArm === id
              ? "border-accent bg-accent-soft"
              : "border-border hover:border-border-strong"
          }`}
        >
          <span
            className={`block font-mono-data text-sm font-semibold ${
              activeArm === id ? "text-accent" : "text-foreground"
            }`}
          >
            {ARM_SHORT_LABEL[id]}
          </span>
          <span className="block text-[10px] text-muted-2 leading-tight mt-0.5">
            {ARM_SUBTITLE[id]}
          </span>
        </button>
      ))}
    </div>
  );
}

export function SingleArmDetail({
  sandboxCase,
  arm,
}: {
  sandboxCase: SandboxCase;
  arm: ArmId;
}) {
  const view = sandboxCase.arms[arm];

  const labelTone: StatusTone = view.abstained ? "neutral" : view.correct ? "good" : "bad";
  const labelText = view.abstained ? "Abstained" : view.correct ? "Correct" : "Incorrect";

  const spanTone: StatusTone = view.abstained
    ? "neutral"
    : view.evidenceSpanFound
      ? "good"
      : "bad";
  const spanText = view.abstained ? "N/A" : view.evidenceSpanFound ? "Found" : "Not found";

  const semanticTone: StatusTone = view.abstained
    ? "neutral"
    : view.semanticSupport
      ? "good"
      : "bad";
  const semanticText = view.abstained
    ? "N/A"
    : view.semanticSupport
      ? "Supported"
      : "Unsupported";

  return (
    <div>
      <p className="text-xs text-muted-2 mb-2">{ARM_NAME[arm]}</p>

      <div className="rounded border border-border divide-y divide-border">
        <div className="p-3">
          <ValueRow label="Prediction" value={view.prediction} />
          <ValueRow label="Gold" value={sandboxCase.gold ?? "—"} />
        </div>
        <div className="p-3">
          <StatusRow label="Label match" tone={labelTone} text={labelText} help={METRIC_HELP.labelMatch} />
          <StatusRow label="Span grounding" tone={spanTone} text={spanText} help={METRIC_HELP.spanGrounding} />
          <StatusRow
            label="Semantic support"
            tone={semanticTone}
            text={semanticText}
            help={METRIC_HELP.semanticSupport}
          />
        </div>
      </div>

      <div className="mt-3">
        <p className="text-xs uppercase tracking-wide text-muted-2 mb-1.5">
          Cited evidence
        </p>
        {view.evidence ? (
          <p className="font-mono-data text-[12.5px] leading-relaxed text-foreground bg-surface-2 rounded border border-border px-3 py-2">
            &ldquo;{view.evidence}&rdquo;
          </p>
        ) : (
          <p className="text-[12.5px] text-muted-2 italic">
            {view.abstained ? "No evidence -- the model abstained." : "No evidence cited."}
          </p>
        )}
      </div>
    </div>
  );
}
