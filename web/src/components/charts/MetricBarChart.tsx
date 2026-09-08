"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ARM_COLOR_VAR, ARM_NAME, ARM_SHORT_LABEL, type ArmId } from "@/data/stageground";
import { pct } from "@/lib/format";

export interface MetricBarDatum {
  id: ArmId;
  value: number;
}

export function MetricBarChart({
  data,
  title,
  description,
  domainMax,
  goodDirection,
}: {
  data: MetricBarDatum[];
  title: string;
  description: string;
  /** Y axis max; defaults to a headroom-padded max of the data. */
  domainMax?: number;
  /** For the small "higher/lower is better" caption. */
  goodDirection: "higher" | "lower";
}) {
  const chartData = data.map((d) => ({
    arm: ARM_SHORT_LABEL[d.id],
    armId: d.id,
    fullName: ARM_NAME[d.id],
    value: d.value,
  }));
  const maxVal = domainMax ?? Math.max(...data.map((d) => d.value)) * 1.25;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <h4 className="font-medium text-sm text-foreground">{title}</h4>
        <span className="text-[11px] text-muted-2 font-mono-data whitespace-nowrap">
          {goodDirection === "higher" ? "higher = better" : "lower = better"}
        </span>
      </div>
      <p className="text-xs text-muted mt-0.5 mb-3">{description}</p>
      <div className="h-44" role="img" aria-label={`${title} bar chart: ${chartData.map((d) => `${d.fullName} ${pct(d.value)}`).join(", ")}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 18, right: 8, left: -18, bottom: 0 }}>
            <CartesianGrid vertical={false} stroke="var(--border)" />
            <XAxis
              dataKey="arm"
              tickLine={false}
              axisLine={{ stroke: "var(--border-strong)" }}
              tick={{ fill: "var(--muted)", fontSize: 12, fontFamily: "var(--font-mono)" }}
            />
            <YAxis
              domain={[0, maxVal]}
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
              formatter={(value, _name, item) => [
                typeof value === "number" ? pct(value) : "",
                (item?.payload as { fullName?: string } | undefined)?.fullName ?? "",
              ]}
              labelFormatter={() => ""}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={44}>
              {chartData.map((d) => (
                <Cell key={d.armId} fill={ARM_COLOR_VAR[d.armId]} />
              ))}
              <LabelList
                dataKey="value"
                position="top"
                formatter={(v) => (typeof v === "number" ? pct(v, 0) : "")}
                style={{ fill: "var(--foreground)", fontSize: 11, fontFamily: "var(--font-mono)" }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
