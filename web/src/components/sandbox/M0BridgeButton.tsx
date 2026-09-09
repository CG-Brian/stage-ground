"use client";

import { DEFAULT_SANDBOX_CASE_ID } from "@/data/stageground";
import { useSandbox } from "./sandbox-context";

export function M0BridgeButton() {
  const { focusCase } = useSandbox();
  return (
    <button
      type="button"
      onClick={() => focusCase(DEFAULT_SANDBOX_CASE_ID)}
      className="inline-flex items-center gap-1 text-sm font-medium text-accent hover:underline underline-offset-4 focus-visible:outline-2 focus-visible:outline-accent rounded"
    >
      View a real M0 example →
    </button>
  );
}
