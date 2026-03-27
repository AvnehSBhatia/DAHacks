"""
Run from repo root (use the project venv so ``torch`` / ``sentence-transformers`` resolve):

  pip install -r demo/requirements.txt
  pip install -r backend/requirements.txt   # if not already
  python -m uvicorn demo.server:app --reload --host 127.0.0.1 --port 8000

Frontend (``frontend/``) proxies ``/api`` and ``/health`` to ``http://127.0.0.1:${API_PORT}``
(repo-root ``.env`` ``API_PORT``, default ``8000`` in ``vite.config.ts``).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

# Load repo-root .env so AUTH_REQUIRED / Auth0 vars apply when using uvicorn.
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from demo.latent_demo import run_latent_demo, stream_latent_demo
from demo.auth_jwt import get_demo_caller

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
    num_agents: int = Field(default=3, ge=1, le=32)
    stagger_s: float = Field(default=0.5, ge=0.0, le=30.0)
    cycles: int = Field(default=1, ge=1, le=20)


@app.post("/api/demo/run")
def demo_run(
    body: RunBody,
    _caller: dict[str, Any] = Depends(get_demo_caller),
):
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


@app.post("/api/demo/stream")
async def demo_stream(
    body: RunBody,
    _caller: dict[str, Any] = Depends(get_demo_caller),
):
    """Server-Sent Events: ``data: {json}\\n\\n`` chunks from ``stream_latent_demo``."""

    async def event_bytes():
        t0 = time.perf_counter()
        print(
            f"[timing] POST /api/demo/stream start agents={body.num_agents} "
            f"cycles={body.cycles} stagger_s={body.stagger_s}",
            flush=True,
        )
        try:
            async for chunk in stream_latent_demo(
                body.prompt.strip(),
                context=body.context.strip(),
                num_agents=body.num_agents,
                stagger_s=body.stagger_s,
                cycles=body.cycles,
            ):
                if isinstance(chunk, str):
                    yield chunk.encode("utf-8")
                else:
                    yield chunk
        finally:
            print(
                f"[timing] POST /api/demo/stream done: {time.perf_counter() - t0:.3f}s",
                flush=True,
            )

    return StreamingResponse(
        event_bytes(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
def health():
    return {"ok": True}
