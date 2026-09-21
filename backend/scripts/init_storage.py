"""Create the Blob containers required by the local MLOps platform."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.core.storage import storage_service


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("storage-init")


CONTAINERS = (
    settings.AZURE_STORAGE_CONTAINER_DATASETS,
    settings.AZURE_STORAGE_CONTAINER_MODELS,
    settings.AZURE_STORAGE_CONTAINER_FEATURES,
    settings.AZURE_STORAGE_CONTAINER_MONITORING,
    settings.AZURE_STORAGE_CONTAINER_AUDIT_LOGS,
    settings.AZURE_STORAGE_CONTAINER_EXPERIMENTS,
    settings.AZURE_STORAGE_CONTAINER_BACKUPS,
    settings.AZURE_STORAGE_CONTAINER_TEMP_PROCESSING,
)


def initialize_storage(max_attempts: int = 30, retry_seconds: float = 2.0) -> None:
    """Wait for Blob storage and idempotently create all configured containers."""
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            for container_name in CONTAINERS:
                storage_service._get_container_client(container_name)
            logger.info("Blob storage is ready; %d containers verified", len(CONTAINERS))
            return
        except Exception as exc:  # bootstrap must tolerate storage startup delay
            last_error = exc
            logger.warning(
                "Blob storage is not ready (attempt %d/%d): %s",
                attempt,
                max_attempts,
                exc,
            )
            time.sleep(retry_seconds)

    raise RuntimeError("Blob storage initialization failed") from last_error


if __name__ == "__main__":
    initialize_storage()
