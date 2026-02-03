FROM python:3.11-slim-bookworm AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
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
ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH="/usr/local/cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set working directory
WORKDIR /app

# Copy the repository from the build context
COPY . /app/anki-service

# Switch to the repository directory
WORKDIR /app/anki-service

# Ensure the 'build_anki' script is executable
RUN chmod +x build_anki run_build_system run_web_api

# Run the build
# This will build 'runner', set up pyenv, compile rust parts, and python protos.
RUN ./build_anki

# Runtime stage
FROM python:3.11-slim-bookworm

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy build artifacts and source from the repository directory
COPY --from=builder /app/anki-service /app/anki-service

# Set working directory to the repository
WORKDIR /app/anki-service

# Set environment variables for runtime
ENV VIRTUAL_ENV=/app/anki-service/out/pyenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="/app/anki-service/out/pylib:/app/anki-service/pylib"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

CMD ["uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "8000"]
