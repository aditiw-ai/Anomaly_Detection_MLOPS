"""
Per-job logging handler that streams log records to Redis.
Each training job gets its own Redis list: training:logs:{job_id}
"""
import logging
import redis
import os
from datetime import datetime, timezone


_LOG_TTL_SECONDS = 86400  # 24 hours


def _get_redis_client() -> redis.Redis:
    from app.core.config import settings
    return redis.from_url(settings.REDIS_URL, decode_responses=True, socket_timeout=2)


class JobLogHandler(logging.Handler):
    """
    Logging handler that pushes formatted records to a Redis list.
    Attach to the root logger at the start of a Celery training task.
    """

    def __init__(self, job_id: str, level: int = logging.DEBUG):
        super().__init__(level)
        self.job_id = job_id
        self.redis_key = f"training:logs:{job_id}"
        self._client: redis.Redis | None = None
        self._ttl_set = False
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        ))

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = _get_redis_client()
        return self._client

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            pipe = self.client.pipeline()
            pipe.rpush(self.redis_key, line)
            if not self._ttl_set:
                pipe.expire(self.redis_key, _LOG_TTL_SECONDS)
                self._ttl_set = True
            pipe.execute()
        except Exception:
            # Never let logging errors crash the training job
            self.handleError(record)

    def close(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        super().close()


def attach_job_logger(job_id: str) -> JobLogHandler:
    """
    Attach a JobLogHandler to the root logger.
    Returns the handler so the caller can detach it later.
    """
    handler = JobLogHandler(job_id)
    logging.getLogger().addHandler(handler)
    return handler


def detach_job_logger(handler: JobLogHandler) -> None:
    """Remove the handler from the root logger and close it."""
    logging.getLogger().removeHandler(handler)
    handler.close()
