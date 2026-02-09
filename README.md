# anki-service

Small FastAPI service that runs on top of an Anki Python layer baked into a base Docker image.

## Dependency Management (uv)

This repo uses `uv` to install Python dependencies for the web API layer.

There are two separate dependency sets:

1. **Anki Python layer (`import anki`)**
   - Built into the base image in `docker/Dockerfile.anki-pylib` by cloning Anki upstream and building a wheel.
   - Not managed by this repo's `uv` project files.

2. **Web API runtime dependencies**
   - Installed in `docker/Dockerfile.web-api` using `uv`'s pip-compatible interface:
     - `uv pip install --system fastapi python-dotenv "uvicorn[standard]" supabase google-genai`

### Recommended: Manage Web API Deps via `pyproject.toml` + `uv.lock`

If you want pinned, reproducible dependency resolution for local development and CI, initialize a `uv` project
in this repo and commit the resulting files:

```bash
uv init
uv add fastapi "uvicorn[standard]" python-dotenv supabase google-genai
uv lock
```

Then developers can install/update the environment with:

```bash
uv sync
```

Run the service with:

```bash
uv run uvicorn web_api:app --host 0.0.0.0 --port 8000
```

Notes:
- The code imports `from dotenv import load_dotenv`, so the PyPI dependency is `python-dotenv` (module name is `dotenv`).
- If you do not have a local Python installed, `uv` can manage one:
  - `uv python install 3.13`
  - `uv sync -p 3.13`

### Exporting `requirements.txt` (Optional)

If you need a `requirements.txt` for other tooling, you can export from `uv.lock`:

```bash
uv export --format requirements.txt --no-dev -o requirements.txt
```

## Docker

Build:

```bash
docker build -f docker/Dockerfile.anki-pylib -t anki-pylib:local .
docker build -f docker/Dockerfile.web-api --build-arg ANKI_PYLIB_IMAGE=anki-pylib:local -t anki-web-api:local .
```

Run:

```bash
docker run --rm -p 8000:8000 anki-web-api:local
```
