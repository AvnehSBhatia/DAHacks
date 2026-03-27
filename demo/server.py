"""
Run from repo root (use the project venv so ``torch`` / ``sentence-transformers`` resolve):

  pip install -r demo/requirements.txt
  pip install -r backend/requirements.txt   # if not already
  python -m uvicorn demo.server:app --reload --host 127.0.0.1 --port 5005

Frontend (``frontend/``) proxies ``/api`` and ``/health`` to this port via ``vite.config.ts``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from demo.latent_demo import run_latent_demo, stream_latent_demo

app = FastAPI(title="DAHacks Demo", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunBody(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    context: str = Field(default="", max_length=4000)
    num_agents: int = Field(default=10, ge=1, le=32)
    stagger_s: float = Field(default=0.5, ge=0.0, le=30.0)
    cycles: int = Field(default=1, ge=1, le=20)


@app.post("/api/demo/run")
def demo_run(body: RunBody):
    """Runs shared ``LatentSpace`` + agents; returns PCA frames and full 64-D ``latent`` payload."""
    t0 = time.perf_counter()
    print(
        f"[timing] POST /api/demo/run start agents={body.num_agents} "
        f"cycles={body.cycles} stagger_s={body.stagger_s}",
        flush=True,
    )
    out = run_latent_demo(
        body.prompt.strip(),
        context=body.context.strip(),
        num_agents=body.num_agents,
        stagger_s=body.stagger_s,
        cycles=body.cycles,
    )
    print(f"[timing] POST /api/demo/run done: {time.perf_counter() - t0:.3f}s", flush=True)
    return out


@app.get("/health")
def health():
    return {"ok": True}
