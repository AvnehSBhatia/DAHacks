#!/usr/bin/env bash
# API for the Vite frontend (proxies /api and /health to port 5005).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec "${PYTHON:-.venv/bin/python}" -m uvicorn demo.server:app --reload --host 127.0.0.1 --port 5005
