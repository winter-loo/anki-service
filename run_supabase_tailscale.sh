#!/usr/bin/env bash
set -euo pipefail

# Run anki-service (Supabase auth mode) + demo frontend behind Tailscale Serve/Funnel.
#
# Goal:
# - Demo frontend (port 3000): reachable publicly via Tailscale Funnel for easy phone testing.
# - anki-service (port 8000): bound to tailscale IP; reachable from your phone if it's on the tailnet.
#
# Required env:
# - PUBLIC_SUPABASE_URL=https://<ref>.supabase.co
# - PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY=sb_publishable_...
#
# Optional env:
# - ANKI_DATA_DIR (default: tenants)
# - FUNNEL_URL (override public base URL)

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEMO_DIR="$ROOT_DIR/demo-better-auth-frontend"

TS_DNS=$(tailscale status --json | python3 -c 'import json,sys; s=json.loads(sys.stdin.read()); name=s.get("Self",{}).get("DNSName",""); print(name[:-1] if name.endswith(".") else name)')
TS_IP=$(tailscale ip -4)

FUNNEL_URL=${FUNNEL_URL:-"https://${TS_DNS}"}
ANKI_SERVICE_URL=${ANKI_SERVICE_URL:-"http://${TS_IP}:8000"}

if [[ -z "${PUBLIC_SUPABASE_URL:-}" || -z "${PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY:-}" ]]; then
  echo "Missing PUBLIC_SUPABASE_URL / PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY" >&2
  exit 1
fi

export ANKI_SERVICE_URL PUBLIC_SUPABASE_URL PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY

cleanup() {
  set +e
  if [[ -n "${P1:-}" ]] && kill -0 "$P1" 2>/dev/null; then kill "$P1" 2>/dev/null; fi
  if [[ -n "${P2:-}" ]] && kill -0 "$P2" 2>/dev/null; then kill "$P2" 2>/dev/null; fi
}
trap cleanup EXIT

free_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      kill $pids 2>/dev/null || true
      sleep 0.2
    fi
  fi
}
free_port 8000
free_port 3000

# Start anki-service bound to tailscale IP (tailnet-only access)
(
  cd "$ROOT_DIR"
  PYTHONPATH="$ROOT_DIR/out/pylib:$ROOT_DIR/pylib" \
    ANKI_AUTH_MODE=supabase \
    ANKI_JWT_ALG=AUTO \
    SUPABASE_PROJECT_URL="$PUBLIC_SUPABASE_URL" \
    ${ANKI_DATA_DIR:+ANKI_DATA_DIR="$ANKI_DATA_DIR"} \
    "$ROOT_DIR/out/pyenv/bin/uvicorn" web_api:app --host "$TS_IP" --port 8000
) >/tmp/anki-service-supabase-tail.log 2>&1 &
P1=$!

# Start demo frontend (local only, exposed via tailscale serve/funnel)
(
  cd "$DEMO_DIR"
  node server.mjs
) >/tmp/demo-frontend-tail.log 2>&1 &
P2=$!

# Ensure tailscale Serve/Funnel is configured for 3000.
if ! tailscale serve status >/dev/null 2>&1; then
  tailscale serve --bg 3000 >/dev/null
fi
(tailscale funnel --bg 3000 >/dev/null) || true

echo "OK"
echo "Frontend (public via Funnel, if enabled): ${FUNNEL_URL}/supabase.html"
echo "anki-service (tailnet only): ${ANKI_SERVICE_URL}/"
echo "Note: your phone must be on Tailscale to reach the anki-service URL."

wait
