"""FastAPI serving layer for StageGround.

Endpoints (same data contract the Next.js review UI consumes):
  GET  /cases          — triage list (M-stage summary per case)
  GET  /case/{id}       — full per-case JSON (report, gold, A/D predictions, tags)
  GET  /metrics         — Track A/B + Track C aggregates
  POST /extract         — live extraction of arbitrary report text (calls the LLM)

Run:  uvicorn stageground.api.app:app --reload
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from stageground.api import store
from stageground.api.predict import extract

app = FastAPI(title="StageGround API", version="0.1.0")

# allow the Next.js dev server to call the API directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/cases")
def get_cases() -> list[dict]:
    return store.case_summaries()


@app.get("/case/{case_id}")
def get_case(case_id: str) -> dict:
    case = store.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"case not found: {case_id}")
    return case


@app.get("/metrics")
def get_metrics() -> dict:
    return store.metrics()


class ExtractRequest(BaseModel):
    text: str
    arm: Literal["A", "D"] = "D"


@app.post("/extract")
def post_extract(req: ExtractRequest) -> dict:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")
    return extract(req.arm, req.text)
