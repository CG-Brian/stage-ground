"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ARM_ORDER,
  ARM_COLOR_VAR,
  ARM_NAME,
  ARM_SHORT_LABEL,
  targetByArmOrder,
} from "@/data/stageground";
import { pct } from "@/lib/format";

type MetricKey = "accuracy" | "semanticUnsupportedRate" | "semanticSupportedAccuracy";

export function TargetGroupedChart({ metric }: { metric: MetricKey }) {
  const targets = ["T", "N", "M"] as const;

  // Pivot so each target is one x-axis category with one bar series per arm
  // -- the shape Recharts needs to actually cluster bars by target.
  const rows = targets.map((t) => {
    const row: Record<string, string | number> = { target: t };
    for (const armRow of targetByArmOrder(t)) {
      row[ARM_SHORT_LABEL[armRow.id]] = armRow[metric];
    }
    return row;
  });

  return (
    <div
      className="h-56"
      role="img"
      aria-label={`${metric} by target and arm: ${targets
        .map((t) =>
          targetByArmOrder(t)
            .map((r) => `${t} ${ARM_NAME[r.id]} ${pct(r[metric])}`)
            .join(", ")
        )
        .join("; ")}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 8, left: -18, bottom: 0 }} barGap={2} barCategoryGap="22%">
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis
            dataKey="target"
            tickLine={false}
            axisLine={{ stroke: "var(--border-strong)" }}
            tick={{ fill: "var(--muted)", fontSize: 12, fontFamily: "var(--font-mono)" }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}`}
            tickLine={false}
            axisLine={false}
            tick={{ fill: "var(--muted-2)", fontSize: 10, fontFamily: "var(--font-mono)" }}
            width={28}
          />
          <Tooltip
            cursor={{ fill: "var(--border)", opacity: 0.4 }}
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border-strong)",
              borderRadius: 8,
              fontSize: 12,
              fontFamily: "var(--font-mono)",
            }}
            formatter={(value) => (typeof value === "number" ? pct(value) : "")}
          />
          {ARM_ORDER.map((id) => (
            <Bar
              key={id}
              dataKey={ARM_SHORT_LABEL[id]}
              fill={ARM_COLOR_VAR[id]}
              radius={[3, 3, 0, 0]}
              maxBarSize={22}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
