"""
RTS Curator Recommendation System - FastAPI Application

Main entry point for the API server.
Environment-agnostic: configuration reads from settings (.env file).
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cts_recommender.settings import get_settings
from api.dependencies import lifespan_handler
from api.routers import recommendations, feedback, competition, catalog, health

# Get settings
cfg = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, cfg.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Application factory for creating FastAPI app.

    Configuration is loaded from settings (reads from .env file).

    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="RTS Curator Recommendation API",
        description="API for generating movie recommendations using Contextual Thompson Sampling",
        version="0.1.0",
        lifespan=lifespan_handler  # Handles startup/shutdown
    )

    # CORS configuration from settings
    logger.info(f"Configuring CORS with origins: {cfg.cors_origins}")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(recommendations.router, prefix="/api/v1", tags=["recommendations"])
    app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])
    app.include_router(competition.router, prefix="/api/v1/competition", tags=["competition"])
    app.include_router(catalog.router, prefix="/api/v1", tags=["catalog"])
    app.include_router(health.router, prefix="/api/v1/health", tags=["health"])

    logger.info(f"FastAPI application created (env={cfg.env})")

    return app


# Create app instance
app = create_app()


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "RTS Curator Recommendation API",
        "version": "0.1.0",
        "environment": cfg.env,
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health/ready"
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting API server on {cfg.api_host}:{cfg.api_port}")
    logger.info(f"Environment: {cfg.env}")
    logger.info(f"Reload: {cfg.api_reload}")
    logger.info(f"Workers: {cfg.api_workers}")

    uvicorn.run(
        "api.main:app",
        host=cfg.api_host,      # From settings!
        port=cfg.api_port,      # From settings!
        reload=cfg.api_reload,  # From settings!
        workers=cfg.api_workers if not cfg.api_reload else 1,  # Workers only work without reload
        log_level=cfg.log_level.lower()
    )
