# Anki Service

This project provides an Anki Web API, which can be run either locally or via a containerized environment (Podman/Docker).

## Multi-user + Auth

By default, the API is **multi-user**: each authenticated user maps to a separate Anki collection on disk.

- User storage root: `users/` (configurable via `ANKI_USER_BASE_DIR`)
- Per-user collection:
  - `users/<user_id>/collection.anki2`
  - `users/<user_id>/collection.media/`

### Auth modes

Set `ANKI_AUTH_MODE`:

- `dev_header` (default): send `X-User-Id: <user_id>` (useful for local dev)
- `jwt`: send `Authorization: Bearer <jwt>`; verifies JWTs.
  - Default: **RS256 + JWKS** (recommended for production; ES256 also supported)
  - Optional: HS256 (dev/legacy)

JWT-related env vars:
- `ANKI_JWT_ALG` (default: `RS256`; options: `RS256`, `ES256`, `HS256`, `AUTO`)
  - `AUTO` reads the token header to pick RS256/ES256 at runtime.
  - RS256/ES256 require `cryptography` (install `PyJWT[crypto]`).
- RS256/JWKS:
  - `ANKI_JWKS_URL` (optional) — if not set, derived as `<issuer>/.well-known/jwks.json`
  - `ANKI_JWT_ISSUER` (recommended) or `BETTER_AUTH_URL`
  - `ANKI_JWT_AUDIENCE` (optional)
- HS256:
  - `ANKI_JWT_HS256_SECRET` (recommended) or `BETTER_AUTH_SECRET`
  - `ANKI_JWT_ISSUER` (optional) or `BETTER_AUTH_URL`
  - `ANKI_JWT_AUDIENCE` (optional)

## Local Installation (No Docker)

If you prefer to run the service directly on your host machine, follow these steps.

### Prerequisites

- **Python**: Version 3.10 or higher.
- **Rust**: Install via [rustup](https://rustup.rs/).
- **System Dependencies**:
  - **Linux (Ubuntu/Debian)**:
    ```bash
    sudo apt-get update && sudo apt-get install -y \
        build-essential libssl-dev pkg-config clang protobuf-compiler
    ```
  - **macOS**:
    ```bash
    brew install protobuf
    ```

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/winter-loo/anki-service.git
   cd anki-service
   ```

2. **Run the initialization script**:
   This script will compile the Rust components, set up a local virtual environment in `out/pyenv`, and generate the necessary Python protocols.
   ```bash
   scripts/build_anki.sh
   ```

3. **Activate the local venv (optional)**:
   The build step creates `out/pyenv` and syncs `python/requirements.txt`.
   ```bash
   source out/pyenv/bin/activate
   ```

### Running the Service

Start the Web API using the provided helper script:

```bash
ANKI_AUTH_MODE=supabase \
PUBLIC_SUPABASE_URL=$(grep PUBLIC_SUPABASE_URL .env | cut -d= -f2) \
PUBLIC_SUPABASE_PUBLISHABLE_KEY=$(grep PUBLIC_SUPABASE_PUBLISHABLE_KEY .env | cut -d= -f2) \
SUPABASE_SECRET_KEY=$(grep SUPABASE_SECRET_KEY .env | cut -d= -f2) \
scripts/run_web_api.sh
```

The service will be available at `http://localhost:8000`. The helper script enables `--reload` mode, which is useful for development.

---

## Running with Podman/Docker

You can build the backend image using the following command from the repo root:

```bash
podman build -t anki-service .
```

## Running the Service

Run the container and map port `8000`:

```bash
podman run -d \
  --name anki-service-app \
  -p 8000:8000 \
  anki-service
```

## Verifying the Service

Once the service is up, the Anki API will be available at:
`http://localhost:8000`

## Stopping and Removing

```bash
podman stop anki-service-app
podman rm anki-service-app
```

## Deployment in Cloud

- [Northflank](https://northflank.com/)
- [Google Cloud Run](https://console.cloud.google.com/cloud-build)

Set environment variables listed below:

- PUBLIC_SUPABASE_URL
- PUBLIC_SUPABASE_PUBLISHABLE_KEY
- SUPABASE_SECRET_KEY
- ANKI_DATA_DIR
- ANKI_AUTH_MODE
