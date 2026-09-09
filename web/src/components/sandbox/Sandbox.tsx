"use client";

import { useEffect, useMemo, useState } from "react";
import {
  sandboxCaseById,
  sandboxCases,
  type ArmId,
  type SandboxCategory,
} from "@/data/stageground";
import { Tag } from "../ui/Card";
import { useSandbox } from "./sandbox-context";
import { ExampleSelector } from "./ExampleSelector";
import { ReportPanel } from "./ReportPanel";
import { ArmTabs, SingleArmDetail } from "./SingleArmPanel";
import { CompareTable } from "./CompareTable";

type Mode = "single" | "compare";

export function Sandbox() {
  const { selectedCaseId, setSelectedCaseId } = useSandbox();
  const [category, setCategory] = useState<SandboxCategory | "all">("all");
  const [target, setTarget] = useState<"all" | "T" | "N" | "M">("all");
  const [mode, setMode] = useState<Mode>("compare");
  const [activeArm, setActiveArm] = useState<ArmId>("D_grounded");

  const filteredCases = useMemo(
    () =>
      sandboxCases.filter(
        (c) =>
          (category === "all" || c.categories.includes(category)) &&
          (target === "all" || c.target === target)
      ),
    [category, target]
  );

  useEffect(() => {
    if (filteredCases.length && !filteredCases.some((c) => c.id === selectedCaseId)) {
      setSelectedCaseId(filteredCases[0].id);
    }
  }, [filteredCases, selectedCaseId, setSelectedCaseId]);

  const sandboxCase = sandboxCaseById(selectedCaseId) ?? sandboxCases[0];

  return (
    <div id="sandbox" className="scroll-mt-16">
      <div className="rounded-lg border border-border-strong bg-surface overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-2.5 bg-surface-2">
          <Tag tone="accent">Reproducible demo</Tag>
          <span className="text-xs text-muted-2">
            Using committed 1,000-case experiment outputs — no live model calls
          </span>
        </div>

        <div className="p-4 sm:p-5">
          <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-4">
            <ExampleSelector
              category={category}
              onCategoryChange={setCategory}
              target={target}
              onTargetChange={setTarget}
              cases={filteredCases}
              selectedCaseId={sandboxCase.id}
              onSelectCase={setSelectedCaseId}
            />
            <div
              role="group"
              aria-label="Sandbox mode"
              className="inline-flex rounded border border-border p-0.5 shrink-0"
            >
              {(["single", "compare"] as Mode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  aria-pressed={mode === m}
                  onClick={() => setMode(m)}
                  className={`px-3 py-1.5 text-xs font-medium rounded-sm transition-colors focus-visible:outline-2 focus-visible:outline-accent ${
                    mode === m
                      ? "bg-accent text-accent-foreground"
                      : "text-muted hover:text-foreground"
                  }`}
                >
                  {m === "single" ? "Single arm" : "Compare all"}
                </button>
              ))}
            </div>
          </div>

          <div className="grid lg:grid-cols-2 gap-5 lg:h-[600px]">
            <ReportPanel
              sandboxCase={sandboxCase}
              highlightEvidence={
                mode === "single" ? sandboxCase.arms[activeArm].evidence : undefined
              }
            />

            <div className="flex flex-col min-h-0">
              {mode === "single" ? (
                <div className="flex flex-col gap-4 lg:overflow-y-auto scroll-thin lg:pr-1">
                  <ArmTabs activeArm={activeArm} onSelect={setActiveArm} />
                  <SingleArmDetail sandboxCase={sandboxCase} arm={activeArm} />
                </div>
              ) : (
                <div className="lg:overflow-y-auto scroll-thin lg:pr-1">
                  <CompareTable sandboxCase={sandboxCase} />
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
