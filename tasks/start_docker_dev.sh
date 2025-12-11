#!/bin/bash
# Script to build and run the development Docker container for A4S Evaluation Service
# This script:
# 1. Builds the development Docker image using Dockerfile.dev
# 2. Runs the container with appropriate volume mounting and port forwarding

# Build development image
docker build -f Dockerfile.dev -t a4s-eval-dev .

# Run container with:
# --rm: Remove container when it stops
# -v .:/app: Mount current directory to /app in container
# -p 8001:8001: Forward port 8001 from container to host
# --name: Give container a recognizable name
# -it: Run interactively with a terminal
docker run --rm -v .:/app -p 8001:8001 --name a4s-eval-dev -it a4s-eval-dev

