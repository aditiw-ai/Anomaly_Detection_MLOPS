from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import asyncio

import redis.asyncio as aioredis
from sqlalchemy import text

from app.core.config import settings
from app.core.database import init_db, engine
from app.api.v1 import api_router
from app.api.v1 import training_logs
from app.api.v1 import analytics
from app.core.storage import storage_service

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
            await asyncio.sleep(delay)
    
    yield
    
    # Shutdown
    try:
        await engine.dispose()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")

app = FastAPI(
    title=settings.APP_NAME,
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
    """Report ready only when the platform's required local services respond."""
    checks = {"database": False, "redis": False, "storage": False}
    errors = {}

    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as exc:
        errors["database"] = str(exc)

    redis_client = aioredis.from_url(settings.REDIS_URL, socket_timeout=2)
    try:
        checks["redis"] = bool(await redis_client.ping())
    except Exception as exc:
        errors["redis"] = str(exc)
    finally:
        await redis_client.aclose()

    try:
        await asyncio.to_thread(storage_service.client.get_account_information)
        checks["storage"] = True
    except Exception as exc:
        errors["storage"] = str(exc)

    ready = all(checks.values())
    payload = {"status": "healthy" if ready else "unhealthy", "checks": checks}
    if errors:
        payload["errors"] = errors

    return JSONResponse(status_code=200 if ready else 503, content=payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
