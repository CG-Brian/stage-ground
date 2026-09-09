"use client";

import { createContext, useContext, useMemo, useState } from "react";
import { DEFAULT_SANDBOX_CASE_ID } from "@/data/stageground";

export type SandboxMode = "single" | "compare";

interface SandboxContextValue {
  selectedCaseId: string;
  setSelectedCaseId: (caseId: string) => void;
  mode: SandboxMode;
  setMode: (mode: SandboxMode) => void;
  /** Selects a case, forces Compare-all (the mode that actually makes the
   * cross-arm story visible), and smooth-scrolls the sandbox into view --
   * the bridge used by the M0 paradox section's "View a real M0 example". */
  focusCase: (caseId: string) => void;
}

const SandboxContext = createContext<SandboxContextValue | null>(null);

export function SandboxProvider({ children }: { children: React.ReactNode }) {
  const [selectedCaseId, setSelectedCaseId] = useState(DEFAULT_SANDBOX_CASE_ID);
  const [mode, setMode] = useState<SandboxMode>("compare");

  const value = useMemo<SandboxContextValue>(
    () => ({
      selectedCaseId,
      setSelectedCaseId,
      mode,
      setMode,
      focusCase: (caseId: string) => {
        setSelectedCaseId(caseId);
        setMode("compare");
        document.getElementById("sandbox")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      },
    }),
    [selectedCaseId, mode]
  );

  return (
    <SandboxContext.Provider value={value}>{children}</SandboxContext.Provider>
  );
}

export function useSandbox(): SandboxContextValue {
  const ctx = useContext(SandboxContext);
  if (!ctx) throw new Error("useSandbox must be used within a SandboxProvider");
  return ctx;
}
