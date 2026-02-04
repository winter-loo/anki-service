#!/usr/bin/env bash
set -euo pipefail

# Convenience runner for local dev.
# Defaults to Supabase auth mode if credentials are available.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# change work directory to main source directory
cd "$ROOT_DIR"/..

export PYTHONPATH="$ROOT_DIR/out/pylib:$ROOT_DIR/pylib"
source "$ROOT_DIR/out/pyenv/bin/activate"

# Best-effort: load local credentials if present (not committed).
# This file may contain secrets; keep permissions tight (chmod 600).
if [[ -f "$HOME/credentials/supabase.txt" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/credentials/supabase.txt"
  set +a
fi

# Warn if UI hasn't been built yet.
if [[ ! -f "$ROOT_DIR/ui/out/index.html" ]]; then
  echo "[run_web_api] UI not built. Build it with: (cd ui/web && npm ci && npm run build)" >&2
fi

# If Supabase public config is available, run in Supabase auth mode.
if [[ -n "${PUBLIC_SUPABASE_URL:-}" && -n "${PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY:-}" ]]; then
  export ANKI_AUTH_MODE=${ANKI_AUTH_MODE:-supabase}
  export ANKI_JWT_ALG=${ANKI_JWT_ALG:-AUTO}
  export SUPABASE_PROJECT_URL=${SUPABASE_PROJECT_URL:-$PUBLIC_SUPABASE_URL}

  # Used by /ui-config.json for the static UI.
  export PUBLIC_SUPABASE_URL
  export PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY

  # Optional: service role key (server-side only) for admin ops like user counting/auto-confirm.
  # supabase.txt stores it as SECRET_KEYS.
  if [[ -n "${SUPABASE_SECRET_KEY:-}" ]]; then
    export SUPABASE_SECRET_KEY
  elif [[ -n "${SECRET_KEYS:-}" ]]; then
    export SUPABASE_SECRET_KEY="$SECRET_KEYS"
  fi
else
  echo "[run_web_api] Supabase env not found; starting in dev_header mode." >&2
  echo "[run_web_api] To enable Supabase auth, set PUBLIC_SUPABASE_URL and PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY" >&2
fi

exec uvicorn web_api:app --reload --host 0.0.0.0 --port 8000
