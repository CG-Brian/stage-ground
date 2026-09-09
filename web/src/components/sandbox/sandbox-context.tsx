"use client";

import { createContext, useContext, useMemo, useState } from "react";
import { DEFAULT_SANDBOX_CASE_ID } from "@/data/stageground";

interface SandboxContextValue {
  selectedCaseId: string;
  /** Selects a case and smooth-scrolls the sandbox into view -- the bridge
   * used by e.g. the M0 paradox section's "View a failure case" link. */
  focusCase: (caseId: string) => void;
  setSelectedCaseId: (caseId: string) => void;
}

const SandboxContext = createContext<SandboxContextValue | null>(null);

export function SandboxProvider({ children }: { children: React.ReactNode }) {
  const [selectedCaseId, setSelectedCaseId] = useState(DEFAULT_SANDBOX_CASE_ID);

  const value = useMemo<SandboxContextValue>(
    () => ({
      selectedCaseId,
      setSelectedCaseId,
      focusCase: (caseId: string) => {
        setSelectedCaseId(caseId);
        document.getElementById("sandbox")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      },
    }),
    [selectedCaseId]
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
