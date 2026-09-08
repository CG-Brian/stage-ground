"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { data } from "@/data/stageground";
import { pct } from "@/lib/format";

const chartData = data.m0.byArm.map((a) => ({
  arm: a.label.split(" · ")[0],
  fullName: a.label,
  "M0 accuracy": a.m0Accuracy,
  "Semantic-supported M0 rate": a.m0SemanticSupportedRate,
}));

export function M0ParadoxChart() {
  return (
    <div
      className="h-72"
      role="img"
      aria-label={`M0 accuracy versus semantic-supported M0 rate by arm: ${data.m0.byArm
        .map((a) => `${a.label} accuracy ${pct(a.m0Accuracy)}, semantic support ${pct(a.m0SemanticSupportedRate)}`)
        .join("; ")}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barGap={4}>
          <CartesianGrid vertical={false} stroke="var(--border)" />
          <XAxis
            dataKey="arm"
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
            width={30}
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
          <Legend
            wrapperStyle={{ fontSize: 12, fontFamily: "var(--font-sans)", color: "var(--muted)" }}
          />
          <Bar
            dataKey="M0 accuracy"
            fill="var(--muted-2)"
            radius={[4, 4, 0, 0]}
            maxBarSize={38}
          />
          <Bar
            dataKey="Semantic-supported M0 rate"
            fill="var(--accent)"
            radius={[4, 4, 0, 0]}
            maxBarSize={38}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
