from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
import json
from app.core.config import settings

import asyncio as _asyncio

from app.core.database import init_db, engine
from app.api.v1 import datasets, training, features
from app.api.v1 import api_router
from app.api.v1 import training_logs
from app.api.v1 import analytics

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup with retry — Docker DNS may not be ready immediately
    max_retries = 5
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Initializing database schema (attempt %d/%d)...", attempt, max_retries)
            await init_db()
            logger.info("Database schema initialized successfully")
            break
        except Exception as e:
            if attempt == max_retries:
                logger.error("Database initialization failed after %d attempts: %s", max_retries, e, exc_info=True)
                raise
            delay = 2 ** attempt
            logger.warning("Database init attempt %d failed (%s), retrying in %ds...", attempt, e, delay)
            await _asyncio.sleep(delay)
    
    yield
    
    # Shutdown
    try:
        await engine.dispose()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

app = FastAPI(
    title="Fraud Detection MLOps Platform",
    swagger_ui_parameters={"persistAuthorization": True},
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix="/api/v1")
app.include_router(training.router, prefix="/api/v1")
app.include_router(features.router, prefix="/api/v1")
app.include_router(training_logs.router, prefix="/api/v1")
app.include_router(api_router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Return 500 as JSON with error detail so the frontend can show the real message."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)