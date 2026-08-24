"""
Monitoring Worker
Background tasks for drift, bias, and performance monitoring.
"""
from app.core.time import IST, now_ist
from celery import shared_task
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@shared_task(name="app.workers.monitoring_worker.compute_drift_metrics")
def compute_drift_metrics(model_id: str = None):
    """
    Compute real drift metrics for production models.
    If model_id is provided, compute for that model only.
    Otherwise, compute for all STAGING/PRODUCTION models.
    """
    import asyncio
    import os
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select

    DATABASE_URL = os.environ.get("DATABASE_URL", "")

    async def _compute():
        engine = create_async_engine(DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        from app.models.ml_model import MLModel
        from app.services.drift_service import DriftMonitoringService

        async with async_session() as db:
            if model_id:
                model_ids = [model_id]
            else:
                result = await db.execute(
                    select(MLModel.id).where(MLModel.status.in_(["STAGING", "PRODUCTION"]))
                )
                model_ids = [str(r) for r in result.scalars().all()]

            results = []
            for mid in model_ids:
                try:
                    service = DriftMonitoringService(db)
                    r = await service.run_drift_check(mid)
                    logger.info(
                        f"Drift check done for {mid}: {r.overall_status}, "
                        f"{r.drifted_features} drifted features"
                    )
                    results.append({
                        "model_id": mid,
                        "status": r.overall_status,
                        "drifted_features": r.drifted_features,
                    })
                except Exception as e:
                    logger.error(f"Drift check failed for model {mid}: {e}")
                    results.append({"model_id": mid, "status": "ERROR", "error": str(e)})

        await engine.dispose()
        return {"computed": len(results), "results": results}

    return asyncio.run(_compute())


@shared_task(name="app.workers.monitoring_worker.compute_bias_metrics")
def compute_bias_metrics(model_id: str = None):
    """
    Compute real bias metrics for production models.
    Evaluates fairness across detected protected attributes.
    """
    import asyncio
    import os
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select

    DATABASE_URL = os.environ.get("DATABASE_URL", "")

    async def _compute():
        engine = create_async_engine(DATABASE_URL)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        from app.models.ml_model import MLModel
        from app.services.bias_service import BiasMonitoringService

        async with async_session() as db:
            if model_id:
                model_ids = [model_id]
            else:
                result = await db.execute(
                    select(MLModel.id).where(MLModel.status.in_(["STAGING", "PRODUCTION"]))
                )
                model_ids = [str(r) for r in result.scalars().all()]

            results = []
            for mid in model_ids:
                try:
                    service = BiasMonitoringService(db)
                    r = await service.run_bias_check(mid)
                    logger.info(
                        f"Bias check done for {mid}: {r.overall_status}, "
                        f"{r.attributes_checked} attributes checked"
                    )
                    results.append({
                        "model_id": mid,
                        "status": r.overall_status,
                        "attributes_checked": r.attributes_checked,
                    })
                except Exception as e:
                    logger.error(f"Bias check failed for model {mid}: {e}")
                    results.append({"model_id": mid, "status": "ERROR", "error": str(e)})

        await engine.dispose()
        return {"computed": len(results), "results": results}

    return asyncio.run(_compute())


@shared_task(name="app.workers.monitoring_worker.check_performance_baselines")
def check_performance_baselines(model_id: str):
    """
    Check if model performance is meeting baseline thresholds.
    """
    import asyncio
    
    async def _check():
        try:
            logger.info(f"Checking baselines for model {model_id}")

            import os
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker
            from sqlalchemy import select
            from app.models.ml_model import MLModel
            from app.models.inference_log import InferenceLog
            from app.services.baseline_service import BaselineService
            from uuid import UUID
            from datetime import timedelta

            DATABASE_URL = os.environ.get("DATABASE_URL")
            engine = create_async_engine(DATABASE_URL, echo=False)
            async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

            async with async_session() as db:
                # Fetch real model metrics from DB
                result = await db.execute(
                    select(MLModel).where(MLModel.id == UUID(model_id))
                )
                model = result.scalar_one_or_none()

                if not model:
                    logger.error(f"Model {model_id} not found in DB")
                    return {"status": "error", "model_id": model_id, "reason": "Model not found"}

                # Prefer real, feedback-backed inference metrics from recent production traffic.
                cutoff = now_ist() - timedelta(days=7)
                logs_result = await db.execute(
                    select(InferenceLog.prediction, InferenceLog.actual_label).where(
                        InferenceLog.model_id == UUID(model_id),
                        InferenceLog.created_at >= cutoff,
                        InferenceLog.actual_label.isnot(None),
                    )
                )
                logs = logs_result.all()

                if logs:
                    tp = fp = fn = tn = 0
                    for row in logs:
                        pred = int(row.prediction)
                        actual = int(row.actual_label)
                        if pred == 1 and actual == 1:
                            tp += 1
                        elif pred == 1 and actual == 0:
                            fp += 1
                        elif pred == 0 and actual == 1:
                            fn += 1
                        elif pred == 0 and actual == 0:
                            tn += 1

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                    # AUC requires full score distributions; keep stored value if present.
                    current_metrics = {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "fpr": fpr,
                        "auc": (model.metrics or {}).get("auc", 0.0),
                    }
                else:
                    # Fallback to stored training metrics only when no feedback labels exist.
                    current_metrics = model.metrics or {}

                if not current_metrics:
                    logger.warning(f"No metrics stored for model {model_id}, skipping check")
                    return {"status": "skipped", "model_id": model_id, "reason": "No metrics"}

                # Run real baseline check
                service = BaselineService(db)
                results = await service.check_baselines(model_id, current_metrics)

                if not results:
                    logger.warning(f"No baselines configured for model {model_id}")
                    return {"status": "skipped", "model_id": model_id, "reason": "No baselines set"}

                violations = [r for r in results if not r.passed]
                critical = [r for r in violations if r.severity == "CRITICAL"]

                for v in violations:
                    log_fn = logger.error if v.severity == "CRITICAL" else logger.warning
                    log_fn(f"BASELINE VIOLATION [{v.severity}]: {v.message}")

                # Persist violations as performance alerts
                from app.services.alert_service import AlertService
                alert_svc = AlertService(db)
                for v in violations:
                    try:
                        await alert_svc.create_performance_alert(
                            model_id=model_id,
                            metric=v.metric,
                            current_value=v.current_value,
                            baseline_value=v.threshold,
                            threshold=abs(v.current_value - v.threshold) / max(v.threshold, 1e-9),
                        )
                    except Exception as e:
                        logger.error(f"Failed to persist performance alert for {v.metric}: {e}")

                logger.info(
                    f"Baseline check completed for model {model_id}: "
                    f"{len(results) - len(violations)} passed, {len(violations)} violated "
                    f"({len(critical)} critical)"
                )

                return {
                    "status": "completed",
                    "model_id": model_id,
                    "total_checks": len(results),
                    "violations": len(violations),
                    "critical": len(critical),
                    "overall_status": "CRITICAL" if critical else ("WARNING" if violations else "OK"),
                    "details": [
                        {
                            "metric": r.metric,
                            "current_value": r.current_value,
                            "threshold": r.threshold,
                            "passed": r.passed,
                            "severity": r.severity,
                        }
                        for r in results
                    ],
                }

        except Exception as e:
            logger.error(f"Baseline check failed: {e}")
            raise
    
    return asyncio.run(_check())


@shared_task(name="app.workers.monitoring_worker.scheduled_drift_check")
def scheduled_drift_check():
    """
    Scheduled task to check drift for all production models.
    Runs hourly.
    """
    logger.info("Starting scheduled drift check for all production models")
    
    # Would fetch all production models and run drift check
    # For now, just log
    compute_drift_metrics.delay(model_id=None)
    
    return {"status": "triggered", "timestamp": datetime.now(IST).isoformat()}


@shared_task(name="app.workers.monitoring_worker.scheduled_bias_check")
def scheduled_bias_check():
    """
    Scheduled task to check bias for all production models.
    Runs every 6 hours.
    """
    logger.info("Starting scheduled bias check for all production models")
    
    compute_bias_metrics.delay(model_id=None)
    
    return {"status": "triggered", "timestamp": datetime.now(IST).isoformat()}


@shared_task(name="app.workers.monitoring_worker.scheduled_performance_check")
def scheduled_performance_check():
    """
    Scheduled task to check performance baselines.
    Runs every 30 minutes.
    """
    logger.info("Starting scheduled performance check for deployed models")
    
    import asyncio
    import os
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import select
    from app.models.ml_model import MLModel

    DATABASE_URL = os.environ.get("DATABASE_URL", "")

    async def _fetch_and_dispatch():
        engine = create_async_engine(DATABASE_URL, echo=False)
        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        count = 0
        async with async_session() as db:
            result = await db.execute(
                select(MLModel.id).where(MLModel.status.in_(["STAGING", "PRODUCTION"]))
            )
            model_ids = [str(r) for r in result.scalars().all()]
            
            for mid in model_ids:
                check_performance_baselines.delay(model_id=mid)
                count += 1
                
        await engine.dispose()
        return count

    dispatched = asyncio.run(_fetch_and_dispatch())
    logger.info(f"Dispatched performance checks for {dispatched} model(s)")
    
    return {"status": "triggered", "models_checked": dispatched, "timestamp": datetime.now(IST).isoformat()}



@shared_task(name="app.workers.monitoring_worker.trigger_automated_retraining")
def trigger_automated_retraining():
    # Intentionally disabled: retraining is manual-only by product policy.
    logger.info("Automated retraining check skipped (manual-only mode enabled).")
    return {
        "status": "disabled",
        "models_queued": 0,
        "model_ids": [],
        "timestamp": datetime.now(IST).isoformat(),
    }
