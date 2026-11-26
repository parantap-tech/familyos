#!/usr/bin/env bash
# Live mutual-fund demo without MF Engine: seeds backend/.demo/ (consent states + identities, tracked data untouched),
# starts the mock MF Engine MCP server on :9300, then runs ./dev.sh pointed at the demo data.
# Needs ANTHROPIC_API_KEY (or ~/Desktop/claude.env, as dev.sh) for the agentic path; without it the deterministic
# pipeline answers and never calls MF Engine tools.
set -e
cd "$(dirname "$0")"
backend/.venv/bin/python backend/tools/seed_live_demo.py
# shellcheck disable=SC1091
source backend/.demo/demo.env
(cd backend && exec .venv/bin/python tools/mock_mf_engine_mcp.py 9300) &
echo "Mock MF Engine on http://127.0.0.1:9300/mcp"
exec ./dev.sh
