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

# Build arguments for repository and branch
ARG REPO_URL=https://github.com/winter-loo/anki-service.git
ARG BRANCH=main

# Clone the repository
RUN git clone --depth 1 --branch ${BRANCH} ${REPO_URL} anki-service

# Switch to the repository directory
WORKDIR /app/anki-service

# Configure git to allow running commands in the repository
# This prevents "fatal: detected dubious ownership in repository" errors
RUN git config --global --add safe.directory /app/anki-service

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
    gcc \
    g++ \
    pkg-config \
    libcairo2-dev \
    libicu-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy build artifacts and source from the repository directory
COPY --from=builder /app/anki-service /app/anki-service

# Set environment variables for runtime
ENV VIRTUAL_ENV=/app/anki-service/out/pyenv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONPATH="/app/anki-service/out/pylib:/app/anki-service/pylib"

RUN pip install --upgrade google-genai

# Expose ports
EXPOSE 8000

CMD ["uvicorn", "web_api:app", "--host", "0.0.0.0", "--port", "8000"]
