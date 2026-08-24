"""
Training Log Streaming API
GET /api/v1/training/jobs/{job_id}/logs        — JSON snapshot
GET /api/v1/training/jobs/{job_id}/logs?stream=true — SSE live stream
"""
import asyncio
import json
import os
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["Training Logs"])

_LOG_POLL_INTERVAL = 1.0   # seconds between Redis polls during SSE stream
_TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


def _get_async_redis():
    from app.core.config import settings
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


@router.get("/jobs/{job_id}/logs")
async def get_training_logs(job_id: str, stream: bool = False):
    """
    Retrieve training logs for a job.

    - stream=false (default): Returns a JSON array of all log lines so far.
    - stream=true: Server-Sent Events stream — pushes new lines in real-time
      until the job reaches a terminal state.
    """
    redis_key = f"training:logs:{job_id}"

    if not stream:
        # --- Snapshot mode ---
        r = _get_async_redis()
        try:
            lines = await r.lrange(redis_key, 0, -1)
        finally:
            await r.aclose()

        return {
            "job_id": job_id,
            "log_lines": lines,
            "total": len(lines),
        }

    # --- SSE streaming mode ---
    async def event_generator():
        r = _get_async_redis()
        try:
            cursor = 0  # next unread index in the Redis list
            yield "retry: 2000\n\n"  # tell client to reconnect after 2s on drop

            while True:
                # Fetch any new lines since last read
                lines = await r.lrange(redis_key, cursor, -1)
                for line in lines:
                    payload = json.dumps({"line": line})
                    yield f"data: {payload}\n\n"
                cursor += len(lines)

                # Check job terminal state via a lightweight job-status key
                # (set by training_worker on completion/failure)
                status = await r.get(f"training:status:{job_id}")
                if status and status in _TERMINAL_STATUSES:
                    yield f"data: {json.dumps({'event': 'done', 'status': status})}\n\n"
                    break

                await asyncio.sleep(_LOG_POLL_INTERVAL)
        except asyncio.CancelledError:
            pass
        finally:
            await r.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
