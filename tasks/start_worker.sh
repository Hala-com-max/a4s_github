#!/bin/bash
# Script to start the development server for A4S Evaluation Service
# This script:
# 1. Validates and loads environment variables (priority: .env.dev-local > .env.dev)
# 2. Starts the uvicorn development server with hot-reloading

# Load and validate environment variables
# Priority: .env.dev-local > .env.dev
safe_source() {
  local file="$(pwd)/$1"
  if [ -f "$file" ] && [ -s "$file" ]; then
    set -o allexport
    . "$file"   # dot instead of source
    set +o allexport
    echo "Loaded environment from $file"
  fi
}

safe_source ".env.dev"
safe_source ".env.dev-local"

# uv run python -m a4s_eval.celery_worker

# Start Celery worker in background, redirect logs to /tmp to avoid permission issues
# Use --pool=solo to avoid multiprocessing issues that cause segmentation faults
uv run celery -A a4s_eval.celery_worker worker --loglevel=info --pool=solo --concurrency=1

# Start uvicorn development server:
# - UV package manager to run uvicorn
# - Host on all interfaces (0.0.0.0)
# - Port 8001
# - Enable auto-reload on code changes
# uv run uvicorn a4s_eval.main:app --host 0.0.0.0 --port 8001 --reload
