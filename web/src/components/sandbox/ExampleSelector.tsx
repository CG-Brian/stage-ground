import {
  CATEGORY_LABEL,
  CATEGORY_ORDER,
  type SandboxCase,
  type SandboxCategory,
} from "@/data/stageground";

const TARGETS: { id: "all" | "T" | "N" | "M"; label: string }[] = [
  { id: "all", label: "All" },
  { id: "T", label: "T" },
  { id: "N", label: "N" },
  { id: "M", label: "M" },
];

export function ExampleSelector({
  category,
  onCategoryChange,
  target,
  onTargetChange,
  cases,
  selectedCaseId,
  onSelectCase,
}: {
  category: SandboxCategory | "all";
  onCategoryChange: (c: SandboxCategory | "all") => void;
  target: "all" | "T" | "N" | "M";
  onTargetChange: (t: "all" | "T" | "N" | "M") => void;
  cases: SandboxCase[];
  selectedCaseId: string;
  onSelectCase: (id: string) => void;
}) {
  return (
    <div className="space-y-3">
      <div role="group" aria-label="Filter by target" className="inline-flex rounded border border-border p-0.5">
        {TARGETS.map((t) => (
          <button
            key={t.id}
            type="button"
            aria-pressed={target === t.id}
            onClick={() => onTargetChange(t.id)}
            className={`px-2.5 py-1 text-xs font-mono-data rounded-sm transition-colors focus-visible:outline-2 focus-visible:outline-accent ${
              target === t.id
                ? "bg-accent text-accent-foreground"
                : "text-muted hover:text-foreground"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div role="group" aria-label="Filter by category" className="flex flex-wrap gap-1.5">
        <button
          type="button"
          aria-pressed={category === "all"}
          onClick={() => onCategoryChange("all")}
          className={`rounded border px-2.5 py-1 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-accent ${
            category === "all"
              ? "bg-foreground text-background border-foreground"
              : "border-border-strong text-muted hover:text-foreground"
          }`}
        >
          All examples
        </button>
        {CATEGORY_ORDER.map((c) => (
          <button
            key={c}
            type="button"
            aria-pressed={category === c}
            onClick={() => onCategoryChange(c)}
            className={`rounded border px-2.5 py-1 text-xs font-medium transition-colors focus-visible:outline-2 focus-visible:outline-accent ${
              category === c
                ? "bg-foreground text-background border-foreground"
                : "border-border-strong text-muted hover:text-foreground"
            }`}
          >
            {CATEGORY_LABEL[c]}
          </button>
        ))}
      </div>

      <label className="block">
        <span className="block text-xs text-muted-2 mb-1">Example</span>
        <select
          value={selectedCaseId}
          onChange={(e) => onSelectCase(e.target.value)}
          className="w-full rounded border border-border-strong bg-surface px-2.5 py-1.5 text-sm font-mono-data text-foreground focus-visible:outline-2 focus-visible:outline-accent"
        >
          {cases.map((c) => (
            <option key={c.id} value={c.id}>
              {c.displayId} · {c.target} · gold {c.gold} ·{" "}
              {c.categories.map((cat) => CATEGORY_LABEL[cat]).join(", ")}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
