# Better Auth + anki-service demo frontend

This is a **minimal** Express + static HTML demo that:

1) Uses **Better Auth** for sign-up/sign-in (email + password)
2) Mints a short-lived **RS256 JWT** (server-side)
3) Exposes a **JWKS** endpoint and calls `anki-service` with `Authorization: Bearer <jwt>`

Why we mint our own JWT:
- Better Auth can represent sessions in different ways (cookies, compact cache, jwt cache, etc).
- `anki-service` needs a clean **Bearer token** containing at least `sub`.
- This demo validates the Better Auth session server-side and then issues a purpose-built JWT for `anki-service`.

## Run

In one terminal (anki-service):

```bash
cd /home/ldd/anki-service
export ANKI_AUTH_MODE=jwt
export ANKI_JWT_ALG=RS256
export BETTER_AUTH_URL='http://localhost:3000'   # used as issuer (iss)
# JWKS default is derived as ${BETTER_AUTH_URL}/.well-known/jwks.json
./run_web_api
```

In another terminal (frontend):

```bash
cd /home/ldd/anki-service/demo-better-auth-frontend
npm install
export BETTER_AUTH_URL='http://localhost:3000'
export ANKI_SERVICE_URL='http://localhost:8000'
node server.mjs
```

Then open:
- http://localhost:3000

## Google OAuth (optional)

This demo can also support **Sign in with Google** (OAuth redirect flow).

1) Create an OAuth Client ID in Google Cloud Console.
2) Add an Authorized redirect URI that matches your Better Auth base URL.

For the current Tailscale Funnel URL, use:
- `https://desktop-vl6jo22.taila6b1f7.ts.net/api/auth/callback/google`

3) Provide env vars when starting the frontend:

```bash
export GOOGLE_CLIENT_ID='...'
export GOOGLE_CLIENT_SECRET='...'
```

## Notes / Production hardening

- Replace SQLite with Postgres.
- Configure `trustedOrigins` correctly.
- Use HTTPS + secure cookies.
- Add real issuer/audience checks in both sides (`ANKI_JWT_ISSUER`, `ANKI_JWT_AUDIENCE`).
- Consider RS256/JWKS in production.
