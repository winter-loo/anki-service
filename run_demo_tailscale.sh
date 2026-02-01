#!/usr/bin/env bash
set -euo pipefail

# Run anki-service + Better Auth demo behind Tailscale Serve/Funnel.
#
# Assumptions:
# - tailscaled is running and logged in
# - (optional) tailscaled can reach controlplane (some networks require a proxy)
# - anki-service has been built (out/pylib exists)
#
# Required env:
# - GOOGLE_CLIENT_ID
# - GOOGLE_CLIENT_SECRET
# Optional env:
# - BETTER_AUTH_URL (defaults to this node's MagicDNS https URL)
# - ANKI_SERVICE_URL (defaults to http://<tailscale-ip>:8000)
# - BETTER_AUTH_SECRET (>=32 chars; demo only)

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEMO_DIR="$ROOT_DIR/demo-better-auth-frontend"

TS_DNS=$(tailscale status --json | python3 -c 'import json,sys; s=json.loads(sys.stdin.read()); name=s.get("Self",{}).get("DNSName",""); print(name[:-1] if name.endswith(".") else name)')
TS_IP=$(tailscale ip -4)

BETTER_AUTH_URL=${BETTER_AUTH_URL:-"https://${TS_DNS}"}
ANKI_SERVICE_URL=${ANKI_SERVICE_URL:-"http://${TS_IP}:8000"}
BETTER_AUTH_SECRET=${BETTER_AUTH_SECRET:-"local-demo-secret-that-is-long-enough-1234567890"}

if [[ -z "${GOOGLE_CLIENT_ID:-}" || -z "${GOOGLE_CLIENT_SECRET:-}" ]]; then
  echo "Missing GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET" >&2
  exit 1
fi

export BETTER_AUTH_URL ANKI_SERVICE_URL BETTER_AUTH_SECRET GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET

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
  # Tell anki-service how to validate demo RS256 tokens minted by the Better Auth demo.
  PYTHONPATH="$ROOT_DIR/out/pylib:$ROOT_DIR/pylib" \
    ANKI_AUTH_MODE=jwt \
    ANKI_JWT_ALG=RS256 \
    ANKI_JWT_ISSUER="$BETTER_AUTH_URL" \
    ANKI_JWKS_URL="$BETTER_AUTH_URL/.well-known/jwks.json" \
    BETTER_AUTH_URL="$BETTER_AUTH_URL" \
    "$ROOT_DIR/out/pyenv/bin/uvicorn" web_api:app --host "$TS_IP" --port 8000
) >/tmp/anki-service-tail.log 2>&1 &
P1=$!

# Start demo frontend (local only, exposed via tailscale serve/funnel)
(
  cd "$DEMO_DIR"
  node server.mjs
) >/tmp/better-auth-tail.log 2>&1 &
P2=$!

# Ensure tailscale Serve/Funnel is configured for 3000.
# Note: these commands require Serve enabled on the tailnet.
if ! tailscale serve status >/dev/null 2>&1; then
  tailscale serve --bg 3000 >/dev/null
fi
# Funnel is optional; keep it on if already allowed.
# If funnel isn't allowed, this will fail silently and Serve will still work within tailnet.
(tailscale funnel --bg 3000 >/dev/null) || true

echo "OK"
echo "Better Auth (public via Funnel, if enabled): ${BETTER_AUTH_URL}/"
echo "Anki-service (tailnet only): ${ANKI_SERVICE_URL}/"
echo "Docs (if available): ${ANKI_SERVICE_URL}/docs"

# Wait forever (until killed)
wait
