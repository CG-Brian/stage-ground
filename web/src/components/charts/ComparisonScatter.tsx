"use client";

import {
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ARM_COLOR_VAR, ARM_NAME, ARM_SHORT_LABEL, overallByArmOrder } from "@/data/stageground";
import { pct } from "@/lib/format";

const points = overallByArmOrder.map((a) => ({
  id: a.id,
  name: ARM_NAME[a.id],
  shortLabel: ARM_SHORT_LABEL[a.id],
  accuracy: a.accuracy,
  groundedRate: 1 - a.semanticUnsupportedRate,
}));

const meanAccuracy = points.reduce((s, p) => s + p.accuracy, 0) / points.length;
const meanGrounded = points.reduce((s, p) => s + p.groundedRate, 0) / points.length;

export function ComparisonScatter() {
  return (
    <div
      className="h-72"
      role="img"
      aria-label={`Accuracy vs grounded rate by arm: ${points
        .map((p) => `${p.name} accuracy ${pct(p.accuracy)}, grounded rate ${pct(p.groundedRate)}`)
        .join("; ")}`}
    >
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
          <CartesianGrid stroke="var(--border)" />
          <XAxis
            type="number"
            dataKey="accuracy"
            domain={[0.55, 0.75]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={{ fill: "var(--muted-2)", fontSize: 11, fontFamily: "var(--font-mono)" }}
            axisLine={{ stroke: "var(--border-strong)" }}
            tickLine={false}
            label={{
              value: "Raw accuracy →",
              position: "insideBottomRight",
              offset: -4,
              fill: "var(--muted)",
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            dataKey="groundedRate"
            domain={[0.35, 0.65]}
            tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
            tick={{ fill: "var(--muted-2)", fontSize: 11, fontFamily: "var(--font-mono)" }}
            axisLine={false}
            tickLine={false}
            label={{
              value: "Grounded rate →",
              angle: -90,
              position: "insideLeft",
              fill: "var(--muted)",
              fontSize: 11,
            }}
          />
          <ReferenceLine x={meanAccuracy} stroke="var(--border-strong)" strokeDasharray="3 3" />
          <ReferenceLine y={meanGrounded} stroke="var(--border-strong)" strokeDasharray="3 3" />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            contentStyle={{
              background: "var(--surface)",
              border: "1px solid var(--border-strong)",
              borderRadius: 6,
              fontSize: 12,
              fontFamily: "var(--font-mono)",
            }}
            formatter={(value, name) => [
              typeof value === "number" ? pct(value) : "",
              name === "accuracy" ? "Accuracy" : "Grounded rate",
            ]}
            labelFormatter={() => ""}
          />
          <Scatter data={points} shape="circle">
            {points.map((p) => (
              <Cell key={p.id} fill={ARM_COLOR_VAR[p.id]} r={6} />
            ))}
            <LabelList
              dataKey="shortLabel"
              position="top"
              offset={8}
              style={{ fill: "var(--foreground)", fontSize: 12, fontFamily: "var(--font-mono)", fontWeight: 600 }}
            />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
