# Anki Service

This project provides an Anki Web API, which can be run either locally or via a containerized environment (Podman/Docker).

## Multi-tenant + Auth (recommended)

By default, the API is **multi-tenant**: each authenticated user maps to a separate Anki collection on disk.

- Tenant storage root: `tenants/` (configurable via `ANKI_TENANT_BASE_DIR`)
- Per-tenant collection:
  - `tenants/<tenant_id>/collection.anki2`
  - `tenants/<tenant_id>/collection.media/`

### Auth modes

Set `ANKI_AUTH_MODE`:

- `dev_header` (default): send `X-User-Id: <tenant_id>` (useful for local dev)
- `jwt`: send `Authorization: Bearer <jwt>`; verifies JWTs.
  - Default: **RS256 + JWKS** (recommended for production)
  - Optional: HS256 (dev/legacy)

JWT-related env vars:
- `ANKI_JWT_ALG` (default: `RS256`; options: `RS256`, `HS256`)
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
   ./build_anki
   ```

3. **Install additional dependencies**:
   ```bash
   source out/pyenv/bin/activate
   pip install --upgrade google-genai
   ```

### Running the Service

Start the Web API using the provided helper script:

```bash
./run_web_api
```

The service will be available at `http://localhost:8000`. The helper script enables `--reload` mode, which is useful for development.

### Web UI

- UI source lives in `ui/web/` (SvelteKit + adapter-static)
- Build output goes to `ui/out/` (not committed)
- The FastAPI app serves static files from `ui/out/`

Build the UI:

```bash
cd ui/web
npm ci
npm run build
```

---

## Running with Podman/Docker

You can build the image using the following command. By default, it clones the `main` branch of the repository.

```bash
podman build -t anki-service .
```

### Customizing the Build

You can specify a different repository or branch using build arguments:

```bash
podman build \
  --build-arg REPO_URL=https://github.com/winter-loo/anki-service.git \
  --build-arg BRANCH=main \
  -t anki-service .
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

## deploy in cloud

[northflank](https://northflank.com)
