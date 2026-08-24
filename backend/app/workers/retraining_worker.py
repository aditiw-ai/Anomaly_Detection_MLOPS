"""
Retraining Worker
Async orchestration entrypoint for retraining runs.

This worker task is intentionally short-lived: it spawns a background thread
that runs the async pipeline so the solo Celery worker can continue processing
training tasks in parallel.
"""
from celery import shared_task
import asyncio
import logging
import threading

logger = logging.getLogger(__name__)

_RUNNING: dict[str, threading.Thread] = {}
_LOCK = threading.Lock()


def _run_pipeline_thread(job_id: str) -> None:
    try:
        from app.services.retraining_service import get_retraining_pipeline

        async def _run():
            pipeline = get_retraining_pipeline()
            await pipeline.run_pipeline(job_id)

        asyncio.run(_run())
    except Exception as exc:
        logger.error(f"Retraining background thread failed for {job_id}: {exc}")
    finally:
        with _LOCK:
            _RUNNING.pop(job_id, None)


@shared_task(name="app.workers.retraining_worker.run_retraining_job")
def run_retraining_job(job_id: str):
    """Launch retraining pipeline asynchronously and return immediately."""
    with _LOCK:
        t = _RUNNING.get(job_id)
        if t and t.is_alive():
            return {"status": "already_running", "job_id": job_id}

        t = threading.Thread(
            target=_run_pipeline_thread,
            args=(job_id,),
            daemon=True,
            name=f"retrain-{job_id[:8]}",
        )
        _RUNNING[job_id] = t
        t.start()

    logger.info(f"Retraining background thread started for {job_id}")
    return {"status": "started", "job_id": job_id}

