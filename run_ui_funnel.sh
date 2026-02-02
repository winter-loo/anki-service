#!/usr/bin/env bash
set -euo pipefail

# Run anki-service UI (port 8000) and expose it via Tailscale Funnel.
# Only one port is exposed.

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT_DIR"

# Load local supabase creds if present
if [[ -f "$HOME/credentials/supabase.txt" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/credentials/supabase.txt"
  set +a
fi

: "${PUBLIC_SUPABASE_URL:?missing PUBLIC_SUPABASE_URL}"
: "${PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY:?missing PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY}"

export ANKI_AUTH_MODE=${ANKI_AUTH_MODE:-supabase}
export ANKI_JWT_ALG=${ANKI_JWT_ALG:-AUTO}
export SUPABASE_PROJECT_URL=${SUPABASE_PROJECT_URL:-$PUBLIC_SUPABASE_URL}
export PUBLIC_SUPABASE_URL
export PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY

# Optional: service role key for server-side signup policy.
if [[ -n "${SUPABASE_SECRET_KEY:-}" ]]; then
  export SUPABASE_SECRET_KEY
elif [[ -n "${SECRET_KEYS:-}" ]]; then
  export SUPABASE_SECRET_KEY="$SECRET_KEYS"
fi

# Start backend
export PYTHONPATH="$ROOT_DIR/out/pylib:$ROOT_DIR/pylib"
source "$ROOT_DIR/out/pyenv/bin/activate"

# free port
if command -v lsof >/dev/null 2>&1; then
  lsof -ti tcp:8000 | xargs -r kill || true
fi

uvicorn web_api:app --host 0.0.0.0 --port 8000 >/tmp/anki-ui-funnel.log 2>&1 &
P1=$!

cleanup(){
  set +e
  if kill -0 "$P1" 2>/dev/null; then kill "$P1" 2>/dev/null; fi
}
trap cleanup EXIT

# Serve + Funnel 8000
if ! tailscale serve status >/dev/null 2>&1; then
  tailscale serve --bg 8000 >/dev/null
else
  tailscale serve --bg 8000 >/dev/null || true
fi
(tailscale funnel --bg 8000 >/dev/null) || true

TS_DNS=$(tailscale status --json | python3 -c 'import json,sys; s=json.loads(sys.stdin.read()); name=s.get("Self",{}).get("DNSName",""); print(name[:-1] if name.endswith(".") else name)')

echo "OK"
echo "UI (public via Funnel, if enabled): https://${TS_DNS}/"

wait
