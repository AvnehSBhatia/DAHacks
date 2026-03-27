#!/usr/bin/env bash
# Run cohesive multi-agent system (repo root = DAHacks).
# Optional: first arg overrides path to JSON config.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${1:-$ROOT/backend/system_config.json}"
exec "$ROOT/.venv/bin/python" "$ROOT/backend/cohesive_system.py" --config "$CONFIG"
