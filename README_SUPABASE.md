# Supabase Auth mode (anki-service)

This repo supports verifying Supabase access tokens directly.

## Env vars

Set:

```bash
export ANKI_AUTH_MODE=supabase
export PUBLIC_SUPABASE_URL='https://<ref>.supabase.co'
# Optional: Supabase projects may use ES256; AUTO reads the token header.
# export ANKI_JWT_ALG='AUTO'  # or ES256
# ES256 requires cryptography (install PyJWT[crypto]).
# Optional overrides:
# export SUPABASE_JWT_ISSUER='https://<ref>.supabase.co/auth/v1'
# export SUPABASE_JWKS_URL='https://<ref>.supabase.co/auth/v1/.well-known/jwks.json'
# export SUPABASE_JWT_AUDIENCE='authenticated'

# Storage (attach a persistent volume here in production)
export ANKI_DATA_DIR='/mnt/anki-data'
# Optional admin setup:
export SUPABASE_SECRET_KEY='sb_secret_...'
```

## Request headers

All API requests:
- `Authorization: Bearer <supabase_access_token>`

Multi-collection selection:
- `X-Collection-Id: <collection_id>` (defaults to `default`)

## Useful endpoints

- `GET /api/auth/whoami` – returns `{ user_id, collection_id, auth_mode, jwt_alg }`
- `GET /api/collections` – list collections for the authenticated user
- `POST /api/collections` – create a collection directory (lazy init)
- `POST /api/collections/{collection_id}/init` – force-create/open the Anki DB (requires RustBackend)
- `DELETE /api/collections/{collection_id}` – delete a collection
