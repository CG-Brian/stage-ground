import { armById, data } from "@/data/stageground";
import { pct, pLabel } from "@/lib/format";

function DeltaRow({
  label,
  from,
  to,
  diffPp,
  pValue,
  nBoot,
}: {
  label: string;
  from: number;
  to: number;
  diffPp: number;
  pValue: number;
  nBoot: number;
}) {
  const improved = diffPp > 0;
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-border last:border-0 text-sm">
      <span className="text-muted">{label}</span>
      <span className="font-mono-data flex items-center gap-2">
        <span className="text-muted-2">{pct(from)}</span>
        <span aria-hidden className="text-muted-2">→</span>
        <span className="text-foreground font-medium">{pct(to)}</span>
        <span className={improved ? "text-good" : "text-bad"}>
          ({improved ? "+" : ""}
          {(diffPp * 100).toFixed(2)} pp)
        </span>
        <span className="text-muted-2 text-xs">{pLabel(pValue, nBoot)}</span>
      </span>
    </div>
  );
}

export function EvidenceBindingResult() {
  const cPlus = armById("C_plus_unknown");
  const d = armById("D_grounded");
  const eb = data.evidenceBinding;

  return (
    <div>
      <h3 className="text-sm font-medium text-foreground mb-1">
        C+ → D: requiring evidence, on top of abstention
      </h3>
      <p className="text-xs text-muted-2 mb-3">
        Paired bootstrap, 95% CI, n_boot = {eb.coverage.nBoot.toLocaleString()}
      </p>
      <div className="rounded border border-border px-4">
        <DeltaRow
          label="Semantic unsupported rate"
          from={cPlus.semanticUnsupportedRate}
          to={d.semanticUnsupportedRate}
          diffPp={eb.semanticUnsupportedRate.diff}
          pValue={eb.semanticUnsupportedRate.pValue}
          nBoot={eb.semanticUnsupportedRate.nBoot}
        />
        <DeltaRow
          label="Semantic-supported accuracy"
          from={cPlus.semanticSupportedAccuracy}
          to={d.semanticSupportedAccuracy}
          diffPp={eb.semanticSupportedAccuracy.diff}
          pValue={eb.semanticSupportedAccuracy.pValue}
          nBoot={eb.semanticSupportedAccuracy.nBoot}
        />
        <DeltaRow
          label="Coverage"
          from={cPlus.coverage}
          to={d.coverage}
          diffPp={eb.coverage.diff}
          pValue={eb.coverage.pValue}
          nBoot={eb.coverage.nBoot}
        />
      </div>
      <p className="mt-2 text-xs text-muted-2">
        Coverage rose too — D isn&rsquo;t simply hiding behind more abstention.
      </p>
    </div>
  );
}
