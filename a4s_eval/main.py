"""Main FastAPI application module for the A4S Evaluation module.

This module initializes and configures the FastAPI application, sets up CORS middleware,
and includes all router modules. It provides the main entry point for the A4S Evaluation module.
"""

import os

from fastapi import APIRouter, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from a4s_eval.routers import datashape, evaluation
from a4s_eval.utils.logging import get_logger

# Initialize the FastAPI application
app = FastAPI(
    title="A4S Evaluation",
    description="AI Audit as a Service API",
    version="0.1.0",
)

# Configure CORS middleware to allow requests from the frontend
allowed_origins = os.getenv("CORS_ORIGINS", "localhost").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint for API health check.

    Returns:
        dict[str, str]: A simple hello world message.
    """
    get_logger().info("Hello world called.")
    return {"message": "Hello World"}


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint for AWS load balancer.

    Returns:
        dict[str, str]: Health status information.
    """
    return {"status": "healthy", "service": "a4s-eval", "version": "0.1.0"}


@app.get("/favicon.ico")
async def favicon() -> Response:
    """Handle favicon requests to prevent 404 errors.

    Returns:
        Response: Empty response with 204 status code.
    """
    return Response(status_code=204)


# Set up versioned routing
v1_router = APIRouter(prefix="/v1")

# Include all feature routers under v1
v1_router.include_router(evaluation.router)
v1_router.include_router(datashape.router)

# Group all versions under /api prefix
api_router = APIRouter(prefix="/api")
api_router.include_router(v1_router)

# Add versioned routes to main application
app.include_router(api_router)

get_logger().info("Application started.")
