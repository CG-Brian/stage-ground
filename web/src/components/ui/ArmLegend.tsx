import { ARM_ORDER, ARM_COLOR_VAR, ARM_SHORT_LABEL, ARM_NAME } from "@/data/stageground";

export function ArmLegend({ className = "" }: { className?: string }) {
  return (
    <ul className={`flex flex-wrap gap-x-6 gap-y-2 ${className}`}>
      {ARM_ORDER.map((id) => (
        <li key={id} className="flex items-center gap-2 text-sm text-muted">
          <span
            aria-hidden
            className="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-sm text-[9px] font-bold text-white font-mono-data"
            style={{ background: ARM_COLOR_VAR[id] }}
          >
            {ARM_SHORT_LABEL[id]}
          </span>
          <span>
            <span className="font-medium text-foreground">
              {ARM_SHORT_LABEL[id]}
            </span>{" "}
            {ARM_NAME[id]}
          </span>
        </li>
      ))}
    </ul>
  );
}
