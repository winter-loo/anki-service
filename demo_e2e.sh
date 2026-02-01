#!/usr/bin/env bash
set -euo pipefail

# End-to-end demo verification for:
# - demo-better-auth-frontend (Better Auth issues RS256 JWT + JWKS)
# - anki-service (FastAPI validates via JWKS and serves APIs)
#
# This script:
# 1) starts both servers
# 2) signs up + signs in
# 3) fetches /api/anki-token
# 4) calls anki-service /api/auth/whoami and /api/note/list
#
# Requirements:
# - anki-service built (out/pylib exists)
# - Node deps installed in demo-better-auth-frontend

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEMO_DIR="$ROOT_DIR/demo-better-auth-frontend"

BETTER_AUTH_URL=${BETTER_AUTH_URL:-http://127.0.0.1:3000}
ANKI_SERVICE_URL=${ANKI_SERVICE_URL:-http://127.0.0.1:8000}
ORIGIN=${ORIGIN:-$BETTER_AUTH_URL}

# Better Auth requires a >=32 char secret
BETTER_AUTH_SECRET=${BETTER_AUTH_SECRET:-"$(python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(48))
PY
)"}

export BETTER_AUTH_URL ANKI_SERVICE_URL BETTER_AUTH_SECRET

cleanup() {
  set +e
  if [[ -n "${P1:-}" ]] && kill -0 "$P1" 2>/dev/null; then kill "$P1" 2>/dev/null; fi
  if [[ -n "${P2:-}" ]] && kill -0 "$P2" 2>/dev/null; then kill "$P2" 2>/dev/null; fi
}
trap cleanup EXIT

wait_http() {
  local url="$1"
  local tries=${2:-60}
  for _ in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.2
  done
  echo "Timed out waiting for $url" >&2
  return 1
}

echo "[demo_e2e] BETTER_AUTH_URL=$BETTER_AUTH_URL"
echo "[demo_e2e] ANKI_SERVICE_URL=$ANKI_SERVICE_URL"

# Ensure ports are free (helpful when rerunning the script).
# Best-effort: ignore errors if tools are missing.
free_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
    if [[ -n "$pids" ]]; then
      echo "[demo_e2e] freeing tcp:$port (pids: $pids)" >&2
      kill $pids 2>/dev/null || true
      sleep 0.2
    fi
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" 2>/dev/null || true
    sleep 0.2
  fi
}
free_port 8000
free_port 3000

# Start anki-service
(
  cd "$ROOT_DIR"
  # Tell anki-service how to validate demo RS256 tokens minted by the Better Auth demo.
  # The demo exposes JWKS at: ${BETTER_AUTH_URL}/.well-known/jwks.json
  PYTHONPATH="$ROOT_DIR/out/pylib:$ROOT_DIR/pylib" \
    ANKI_AUTH_MODE=jwt \
    ANKI_JWT_ALG=RS256 \
    ANKI_JWT_ISSUER="$BETTER_AUTH_URL" \
    ANKI_JWKS_URL="$BETTER_AUTH_URL/.well-known/jwks.json" \
    BETTER_AUTH_URL="$BETTER_AUTH_URL" \
    "$ROOT_DIR/out/pyenv/bin/uvicorn" web_api:app --host 127.0.0.1 --port 8000
) >/tmp/anki-service-demo.log 2>&1 &
P1=$!

# Start demo-better-auth-frontend
(
  cd "$DEMO_DIR"
  node server.mjs
) >/tmp/better-auth-demo.log 2>&1 &
P2=$!

wait_http "$BETTER_AUTH_URL/"

# In case native module is missing, attempt a rebuild once.
if ! (cd "$DEMO_DIR" && node -e "require('better-sqlite3');") >/dev/null 2>&1; then
  echo "[demo_e2e] better-sqlite3 missing native binding; attempting rebuild..." >&2
  (cd "$DEMO_DIR" && npm rebuild better-sqlite3) >/tmp/better-sqlite3-rebuild.log 2>&1 || {
    echo "[demo_e2e] rebuild failed; see /tmp/better-sqlite3-rebuild.log" >&2
    exit 1
  }
  # restart demo server after rebuild
  if kill -0 "$P2" 2>/dev/null; then kill "$P2"; fi
  (
    cd "$DEMO_DIR"
    node server.mjs
  ) >/tmp/better-auth-demo.log 2>&1 &
  P2=$!
  wait_http "$BETTER_AUTH_URL/"
fi

# anki-service requires auth on most /api routes; treat 401 as "ready".
wait_http_status() {
  local url="$1"
  local ok_codes_csv="$2"  # e.g. "200,401"
  local tries=${3:-60}
  for _ in $(seq 1 "$tries"); do
    code=$(curl -sS -o /dev/null -w "%{http_code}" "$url" || true)
    IFS=',' read -ra ok <<<"$ok_codes_csv"
    for c in "${ok[@]}"; do
      if [[ "$code" == "$c" ]]; then
        return 0
      fi
    done
    sleep 0.2
  done
  echo "Timed out waiting for $url (last http=$code)" >&2
  return 1
}

wait_http_status "$ANKI_SERVICE_URL/api/auth/whoami" "200,401" 80

EMAIL="test$(date +%s)@example.com"
PASS="test123456!"
JAR=$(mktemp)

# Sign up
curl -fsS -c "$JAR" -b "$JAR" \
  -H 'content-type: application/json' \
  -H "Origin: $ORIGIN" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\",\"name\":\"test\"}" \
  "$BETTER_AUTH_URL/api/auth/sign-up/email" >/dev/null

# Sign in
curl -fsS -c "$JAR" -b "$JAR" \
  -H 'content-type: application/json' \
  -H "Origin: $ORIGIN" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASS\"}" \
  "$BETTER_AUTH_URL/api/auth/sign-in/email" >/dev/null

# Get JWT for anki-service
TOKEN_JSON=$(curl -fsS -c "$JAR" -b "$JAR" -H "Origin: $ORIGIN" "$BETTER_AUTH_URL/api/anki-token")
TOKEN=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["token"])' <<<"$TOKEN_JSON")
SUB=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["sub"])' <<<"$TOKEN_JSON")

# Call whoami
WHOAMI=$(curl -fsS -H "Authorization: Bearer $TOKEN" "$ANKI_SERVICE_URL/api/auth/whoami")

# Call a real API route (requires RustBackend)
NOTES=$(curl -fsS -H "Authorization: Bearer $TOKEN" "$ANKI_SERVICE_URL/api/note/list")

rm -f "$JAR"

echo "[demo_e2e] ok"
echo "[demo_e2e] sub=$SUB"
echo "[demo_e2e] whoami=$WHOAMI"
echo "[demo_e2e] note_list=$NOTES"
