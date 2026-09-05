#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.npm-global/bin:$PATH"
export GLM_API_KEY="$(grep '^GLM_API_KEY=' ~/.hermes/.env | cut -d= -f2-)"
DISPATCH=~/.hermes/skills/autonomous-ai-agents/orch/scripts/dispatch.py
PY="${PYTHON:-/opt/anaconda3/bin/python3}"
"$PY" "$DISPATCH" ~/projects/forgeharness/.agent/tasks/FH-001.yaml --run --timeout-min 90
