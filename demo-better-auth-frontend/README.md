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

## Notes / Production hardening

- Replace SQLite with Postgres.
- Configure `trustedOrigins` correctly.
- Use HTTPS + secure cookies.
- Add real issuer/audience checks in both sides (`ANKI_JWT_ISSUER`, `ANKI_JWT_AUDIENCE`).
- Consider RS256/JWKS in production.
