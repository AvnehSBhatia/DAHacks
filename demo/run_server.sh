#!/usr/bin/env bash
# API for the Vite frontend (must match API_PORT in repo-root .env; default 8000).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${API_PORT:-8000}"
exec "${PYTHON:-.venv/bin/python}" -m uvicorn demo.server:app --reload --host 127.0.0.1 --port "$PORT"
