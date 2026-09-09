import {
  ARM_ORDER,
  ARM_SHORT_LABEL,
  ARM_SUBTITLE,
  type SandboxCase,
} from "@/data/stageground";

function cell(tone: "good" | "bad" | "neutral", text: string) {
  const cls = { good: "text-good", bad: "text-bad", neutral: "text-muted-2" }[tone];
  const symbol = { good: "✓", bad: "✕", neutral: "—" }[tone];
  return (
    <span className={`font-mono-data text-xs font-medium ${cls}`}>
      {symbol} {text}
    </span>
  );
}

export function CompareTable({ sandboxCase }: { sandboxCase: SandboxCase }) {
  return (
    <div className="overflow-x-auto scroll-thin rounded border border-border">
      <table className="w-full min-w-[520px] text-sm border-collapse">
        <thead>
          <tr className="bg-surface-2 text-left">
            <th className="px-3 py-2 text-xs font-medium text-muted-2 w-32">
              &nbsp;
            </th>
            {ARM_ORDER.map((id) => (
              <th key={id} className="px-3 py-2 border-l border-border">
                <span className="block font-mono-data text-sm font-semibold text-foreground">
                  {ARM_SHORT_LABEL[id]}
                </span>
                <span className="block text-[10px] font-normal text-muted-2">
                  {ARM_SUBTITLE[id]}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="font-mono-data">
          <tr className="border-t border-border">
            <td className="px-3 py-2 text-xs text-muted">Prediction</td>
            {ARM_ORDER.map((id) => (
              <td key={id} className="px-3 py-2 border-l border-border text-foreground font-medium">
                {sandboxCase.arms[id].prediction}
              </td>
            ))}
          </tr>
          <tr className="border-t border-border">
            <td className="px-3 py-2 text-xs text-muted font-sans">Correct</td>
            {ARM_ORDER.map((id) => {
              const v = sandboxCase.arms[id];
              return (
                <td key={id} className="px-3 py-2 border-l border-border">
                  {v.abstained
                    ? cell("neutral", "Abstained")
                    : v.correct
                      ? cell("good", "Yes")
                      : cell("bad", "No")}
                </td>
              );
            })}
          </tr>
          <tr className="border-t border-border">
            <td className="px-3 py-2 text-xs text-muted font-sans align-top">Evidence</td>
            {ARM_ORDER.map((id) => {
              const v = sandboxCase.arms[id];
              return (
                <td key={id} className="px-3 py-2 border-l border-border text-[11px] text-muted max-w-40 align-top">
                  {v.evidence ? `"${v.evidence}"` : "none"}
                </td>
              );
            })}
          </tr>
          <tr className="border-t border-border">
            <td className="px-3 py-2 text-xs text-muted font-sans">Span grounded</td>
            {ARM_ORDER.map((id) => {
              const v = sandboxCase.arms[id];
              return (
                <td key={id} className="px-3 py-2 border-l border-border">
                  {v.abstained
                    ? cell("neutral", "—")
                    : v.evidenceSpanFound
                      ? cell("good", "Found")
                      : cell("bad", "Not found")}
                </td>
              );
            })}
          </tr>
          <tr className="border-t border-border">
            <td className="px-3 py-2 text-xs text-muted font-sans">Semantic support</td>
            {ARM_ORDER.map((id) => {
              const v = sandboxCase.arms[id];
              return (
                <td key={id} className="px-3 py-2 border-l border-border">
                  {v.abstained
                    ? cell("neutral", "—")
                    : v.semanticSupport
                      ? cell("good", "Supported")
                      : cell("bad", "Unsupported")}
                </td>
              );
            })}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
