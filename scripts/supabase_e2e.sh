#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for Supabase Auth mode.
#
# Verifies:
# - Supabase issues access tokens (password grant)
# - anki-service validates Supabase RS256 JWT via Supabase JWKS
# - /api/auth/whoami works
# - a basic API route works (/api/note/list)
#
# Required env vars:
#   PUBLIC_SUPABASE_URL=https://<ref>.supabase.co
#   SUPABASE_PUBLISHABLE_KEY=<Supabase publishable key>
#     (Older name SUPABASE_ANON_KEY is also accepted for compatibility)
#   SUPABASE_TEST_EMAIL=<email>
#   SUPABASE_TEST_PASSWORD=<password>
# Optional:
#   ANKI_SERVICE_URL=http://127.0.0.1:8000
#   ANKI_DATA_DIR=users

ANKI_SERVICE_URL=${ANKI_SERVICE_URL:-http://127.0.0.1:8000}
PUBLIC_SUPABASE_URL=${PUBLIC_SUPABASE_URL:?missing PUBLIC_SUPABASE_URL}

# Supabase renamed keys: anon/public -> publishable.
# Accept both names to keep setup friction low.
SUPABASE_PUBLISHABLE_KEY=${SUPABASE_PUBLISHABLE_KEY:-${SUPABASE_ANON_KEY:-}}
if [[ -z "$SUPABASE_PUBLISHABLE_KEY" ]]; then
  echo "missing SUPABASE_PUBLISHABLE_KEY (or legacy SUPABASE_ANON_KEY)" >&2
  exit 1
fi

SUPABASE_TEST_EMAIL=${SUPABASE_TEST_EMAIL:?missing SUPABASE_TEST_EMAIL}
SUPABASE_TEST_PASSWORD=${SUPABASE_TEST_PASSWORD:?missing SUPABASE_TEST_PASSWORD}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cleanup() {
  set +e
  if [[ -n "${P1:-}" ]] && kill -0 "$P1" 2>/dev/null; then kill "$P1" 2>/dev/null; fi
}
trap cleanup EXIT

wait_http_status() {
  local url="$1"
  local ok_codes_csv="$2"  # e.g. "200,401"
  local tries=${3:-80}
  for _ in $(seq 1 "$tries"); do
    code=$(curl -sS -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || true)
    IFS=',' read -ra ok <<<"$ok_codes_csv"
    for c in "${ok[@]}"; do
      if [[ "$code" == "$c" ]]; then
        return 0
      fi
    done
    sleep 0.25
  done
  echo "Timed out waiting for $url (last http=$code)" >&2
  return 1
}

# Start anki-service in Supabase auth mode
(
  cd "$ROOT_DIR"
  PYTHONPATH="$ROOT_DIR/out/pylib:$ROOT_DIR/pylib" \
    ANKI_AUTH_MODE=supabase \
    # Supabase may use RS256 or ES256 depending on project; AUTO reads alg from token header.
    ANKI_JWT_ALG=RS256  \
    PUBLIC_SUPABASE_URL="$PUBLIC_SUPABASE_URL" \
    ${ANKI_DATA_DIR:+ANKI_DATA_DIR="$ANKI_DATA_DIR"} \
    "$ROOT_DIR/out/pyenv/bin/uvicorn" web_api:app --host 127.0.0.1 --port 8000
) >/tmp/anki-service-supabase-e2e.log 2>&1 &
P1=$!

# Wait until service is up (401 expected without token)
wait_http_status "$ANKI_SERVICE_URL/api/auth/whoami" "200,401" 120

# Get a Supabase access token via password grant
TOKEN_JSON=$(curl -fsS \
  -H "apikey: $SUPABASE_PUBLISHABLE_KEY" \
  -H "Authorization: Bearer $SUPABASE_PUBLISHABLE_KEY" \
  -H "content-type: application/json" \
  -d "{\"email\":\"$SUPABASE_TEST_EMAIL\",\"password\":\"$SUPABASE_TEST_PASSWORD\"}" \
  "$PUBLIC_SUPABASE_URL/auth/v1/token?grant_type=password")

ACCESS_TOKEN=$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["access_token"])' <<<"$TOKEN_JSON")

# Call whoami
WHOAMI=$(curl -fsS -H "Authorization: Bearer $ACCESS_TOKEN" "$ANKI_SERVICE_URL/api/auth/whoami")

# Call a real API route
NOTES=$(curl -fsS -H "Authorization: Bearer $ACCESS_TOKEN" "$ANKI_SERVICE_URL/api/note/list")

echo "[supabase_e2e] ok"
echo "[supabase_e2e] whoami=$WHOAMI"
echo "[supabase_e2e] note_list=$NOTES"
