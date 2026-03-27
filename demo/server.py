"""
Run from repo root:
  pip install -r demo/requirements.txt
  python -m uvicorn demo.server:app --reload --port 8000
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from demo.demo_logic import run_demo

app = FastAPI(title="DAHacks Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


@app.post("/api/demo/run")
def demo_run(body: RunBody):
    return run_demo(body.prompt)


@app.get("/health")
def health():
    return {"ok": True}
