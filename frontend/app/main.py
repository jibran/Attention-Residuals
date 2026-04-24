"""FastAPI application factory and Uvicorn entry point.

Run directly::

    python -m app.main

Or via the installed script::

    attnres-serve

Or with Uvicorn manually::

    uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import api, pages

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Run startup / shutdown logic around the application lifespan.

    Startup scans the checkpoints directory so the model list is ready
    before the first request arrives.

    Args:
        application: The FastAPI application instance.

    Yields:
        Control to the ASGI server while the application is running.
    """
    from app.model_registry import registry

    registry.list_models()  # eagerly scan checkpoints directory
    yield


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured :class:`~fastapi.FastAPI` instance.
    """
    from pathlib import Path

    application = FastAPI(
        title=settings.app_title,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    # CORS — restrict to same origin in production
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static files
    static_dir = Path(__file__).resolve().parent / "static"
    application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Routers
    application.include_router(pages.router)
    application.include_router(api.router)

    return application


app = create_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Start the Uvicorn development server.

    Reads host, port, workers, and log_level from :mod:`app.config`.
    """
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    run()
