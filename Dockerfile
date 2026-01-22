FROM python:3.11-slim-bookworm AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    libssl-dev \
    pkg-config \
    clang \
    ninja-build \
    git \
    protobuf-compiler \
    libprotobuf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Configure git to allow running commands in the copied repository
# This prevents "fatal: detected dubious ownership in repository" errors
RUN git config --global --add safe.directory /app

# Ensure the 'run' script is executable
RUN chmod +x run ninja run_web_api

# Run the build
# This will build 'runner', set up pyenv, compile rust parts, and python protos.
RUN ./run

# Runtime stage
FROM python:3.11-slim-bookworm

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy build artifacts and source
COPY --from=builder /app /app

# Set environment variables for runtime
# The build created a venv at out/pyenv.
# We set VIRTUAL_ENV and update PATH so that 'uvicorn' (installed in venv) is found.
ENV VIRTUAL_ENV=/app/out/pyenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# 'run_web_api' sets PYTHONPATH, but we can set it here too to be safe/clear.
ENV PYTHONPATH="/app/out/pylib:/app/pylib"

# Expose port (default for uvicorn is 8000)
EXPOSE 8000

# Entrypoint
# We run uvicorn directly to bind to 0.0.0.0, as localhost binding in run_web_api won't be accessible.
CMD ["uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "8000"]
