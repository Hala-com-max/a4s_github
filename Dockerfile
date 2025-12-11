# Production Dockerfile for A4S Evaluation Service
#
# This Dockerfile creates a production-ready image using a multi-stage build:
# 1. Base stage:
#    - Based on Python 3.12 slim image with uv package manager
#    - Installs essential system dependencies
#    - Creates a non-root application user
#
# 2. Builder stage:
#    - Installs production dependencies with caching
#    - Compiles Python bytecode
#    - Sets up the project environment
#
# 3. Final stage:
#    - Copies the built application from the builder stage
#    - Configures runtime environment variables
#    - Ensures proper directory permissions
#    - Runs the application as a non-root user
#    - Includes a health check for the evaluation service

# Base stage with Python 3.12 + uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Install system dependencies for production
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    procps \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Create application user for security
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# Builder stage for dependencies and compilation
FROM base AS builder

# Enable bytecode compilation and copy linking
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Install production dependencies with caching
RUN --mount=type=cache,target=/root/.cache/uv \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --frozen --no-install-project --no-dev

# Copy application code
COPY . /app

# Install project in production mode
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev

# Final stage for runtime
FROM base

# Copy built application from builder
COPY --from=builder --chown=appuser:appuser /app /app
WORKDIR /app

# Add virtual environment to PATH and set Python environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chmod 755 /app/data /app/logs

# Switch to non-root user
USER appuser

# Make startup scripts executable
RUN chmod +x /app/tasks/entrypoint.sh

# Use the new entrypoint script
ENTRYPOINT ["/app/tasks/entrypoint.sh"]

# Default to combined mode (eval module + celery - possible to override)
CMD ["combined"]

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1