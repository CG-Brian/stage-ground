import { Container } from "../ui/Container";
import { ARM_COLOR_VAR, ARM_ORDER, ARM_SHORT_LABEL, ARM_SUBTITLE } from "@/data/stageground";

export function InterventionLadder() {
  return (
    <section className="border-t border-border">
      <Container wide className="py-8">
        <p className="text-xs text-muted-2 mb-3">
          The ablation, for reference — read A → C → C+ → D as one progressive
          intervention, not four unrelated models.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          {ARM_ORDER.map((id, i) => (
            <div key={id} className="flex items-center gap-2">
              <div className="flex items-center gap-2 rounded border border-border px-3 py-1.5">
                <span
                  className="inline-flex h-5 w-5 items-center justify-center rounded-sm text-[10px] font-bold text-white font-mono-data"
                  style={{ background: ARM_COLOR_VAR[id] }}
                >
                  {ARM_SHORT_LABEL[id]}
                </span>
                <span className="text-xs text-muted">{ARM_SUBTITLE[id]}</span>
              </div>
              {i < ARM_ORDER.length - 1 && (
                <span aria-hidden className="text-muted-2 text-sm">
                  →
                </span>
              )}
            </div>
          ))}
        </div>
      </Container>
    </section>
  );
}
