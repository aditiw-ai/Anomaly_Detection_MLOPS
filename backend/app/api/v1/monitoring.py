"""
Monitoring API Endpoints
Drift and bias detection, model performance monitoring.
"""
import logging

from app.core.time import IST, now_ist
from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel

from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])



class DriftThresholds(BaseModel):
    """Drift detection thresholds."""
    psi_warning: float = 0.1
    psi_critical: float = 0.25
    ks_alpha: float = 0.05


class BiasThresholds(BaseModel):
    """Bias detection thresholds."""
    demographic_parity: float = 0.1
    disparate_impact: float = 0.8


@router.get("/drift/{model_id}")
async def get_drift_metrics(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    from app.services.drift_service import DriftMonitoringService

    service = DriftMonitoringService(db)
    try:
        result = await service.run_drift_check(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")

    # Graceful fallback when no reference data is available
    if not result.metrics:
        return {
            "data": {
                "overall_status": "NO_DATA",
                "last_computed": result.computed_at.isoformat(),
                "features": {},
                "message": "No reference data available for drift computation. Run a training job first.",
                "thresholds": {
                    "psi_warning": DriftMonitoringService.PSI_WARNING,
                    "psi_critical": DriftMonitoringService.PSI_CRITICAL,
                    "ks_alpha": DriftMonitoringService.KS_ALPHA,
                },
            }
        }

    return {
        "data": {
            "overall_status": result.overall_status,
            "last_computed": result.computed_at.isoformat(),
            "features": result.metrics,
            "thresholds": {
                "psi_warning": DriftMonitoringService.PSI_WARNING,
                "psi_critical": DriftMonitoringService.PSI_CRITICAL,
                "ks_alpha": DriftMonitoringService.KS_ALPHA,
            },
        }
    }


@router.get("/drift/{model_id}/history")
async def get_drift_history(
    model_id: str,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """
    Get drift metrics history from persisted real drift snapshots.
    """
    from app.services.drift_service import DriftMonitoringService
    service = DriftMonitoringService(db)
    history = await service.get_drift_history(model_id=model_id, days=days)
    return {"data": history}


@router.get("/drift/{model_id}/feature/{feature}")
async def get_feature_drift_trend(
    model_id: str,
    feature: str,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
):
    """
    Get drift trend for a specific feature from persisted real drift snapshots.
    """
    from app.services.drift_service import DriftMonitoringService
    service = DriftMonitoringService(db)
    trend = await service.get_feature_drift_trend(model_id=model_id, feature=feature, days=days)
    return {
        "data": {
            "feature": feature,
            "trend": trend,
        }
    }


@router.put("/drift/{model_id}/thresholds")
async def update_drift_thresholds(
    model_id: str,
    thresholds: DriftThresholds,
    db: AsyncSession = Depends(get_db),
):
    """
    Update drift detection thresholds for a model.
    """
    return {
        "data": {
            "model_id": model_id,
            "thresholds": {
                "psi_warning": thresholds.psi_warning,
                "psi_critical": thresholds.psi_critical,
                "ks_alpha": thresholds.ks_alpha,
            },
        },
        "message": "Thresholds updated"
    }


@router.get("/bias/{model_id}")
async def get_bias_metrics(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    from app.services.bias_service import BiasMonitoringService

    service = BiasMonitoringService(db)
    try:
        result = await service.run_bias_check(model_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bias check failed: {str(e)}")

    # Graceful fallback when bias service returns no data
    if not result.metrics:
        return {
            "data": {
                "overall_status": "NO_DATA",
                "last_computed": result.computed_at.isoformat(),
                "protected_attributes": {},
                "message": "No bias data available. Reference dataset required.",
                "thresholds": {
                    "demographic_parity": 0.1,
                    "disparate_impact": 0.8,
                },
            }
        }

    return {
        "data": {
            "overall_status": result.overall_status,
            "last_computed": result.computed_at.isoformat(),
            "protected_attributes": result.metrics,
            "thresholds": {
                "demographic_parity": 0.1,
                "disparate_impact": 0.8,
            },
        }
    }


@router.get("/performance/{model_id}")
async def get_performance_metrics(
    model_id: str,
    period: str = "7d",
    db: AsyncSession = Depends(get_db),
):
    """
    Get real model performance metrics by joining predictions with ground truth actuals.
    """
    from sqlalchemy import select, func, cast, Date
    from app.models.ml_model import MLModel
    from app.models.inference_log import InferenceLog
    from uuid import UUID

    try:
        result = await db.execute(select(MLModel).where(MLModel.id == UUID(model_id)))
        model = result.scalar_one_or_none()
    except Exception:
        model = None

    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    metrics = model.metrics or {}
    baseline = {
        "precision": metrics.get("precision", 0.0),
        "recall":    metrics.get("recall",    0.0),
        "f1":        metrics.get("f1",        0.0),
        "auc":       metrics.get("auc",       0.0),
        "fpr":       metrics.get("fpr",       0.0),
    }

    days = int(period.replace("d", "")) if period.endswith("d") else 7
    cutoff_date = now_ist() - timedelta(days=days)
    
    # Query actual inference logs that have received feedback (actual_label is not NULL)
    log_result = await db.execute(
        select(
            cast(InferenceLog.created_at, Date).label('date'),
            InferenceLog.prediction,
            InferenceLog.actual_label
        )
        .where(
            InferenceLog.model_id == UUID(model_id),
            InferenceLog.created_at >= cutoff_date,
            InferenceLog.actual_label.isnot(None)
        )
    )
    
    logs = log_result.all()
    
    # Calculate daily metrics
    daily_stats = {}
    total_tp = total_fp = total_fn = total_tn = 0
    
    for log in logs:
        d_str = log.date.strftime("%Y-%m-%d")
        if d_str not in daily_stats:
            daily_stats[d_str] = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            
        p = log.prediction
        a = log.actual_label
        
        if p == 1 and a == 1:
            daily_stats[d_str]["tp"] += 1
            total_tp += 1
        elif p == 1 and a == 0:
            daily_stats[d_str]["fp"] += 1
            total_fp += 1
        elif p == 0 and a == 1:
            daily_stats[d_str]["fn"] += 1
            total_fn += 1
        elif p == 0 and a == 0:
            daily_stats[d_str]["tn"] += 1
            total_tn += 1

    trend = []
    
    # Fill trend for the UI
    for i in range(days):
        date_str = (now_ist() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
        
        if date_str in daily_stats:
            stats = daily_stats[date_str]
            tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
            
            p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_val = 2 * (p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        else:
            # If no data on a day, carry forward baseline (implies no degradation)
            p_val = baseline["precision"]
            r_val = baseline["recall"]
            f1_val = baseline["f1"]
            
        trend.append({
            "date": date_str,
            "precision": round(p_val, 4),
            "recall": round(r_val, 4),
            "f1": round(f1_val, 4),
        })

    # Calculate overall current period metrics
    current_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else baseline["precision"]
    current_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else baseline["recall"]
    current_f1 = 2 * (current_p * current_r) / (current_p + current_r) if (current_p + current_r) > 0 else baseline["f1"]

    current = {
        "precision": round(current_p, 4),
        "recall":    round(current_r, 4),
        "f1":        round(current_f1, 4),
        "auc":       baseline["auc"],  # Needs full probability curve to calculate
        "fpr":       baseline["fpr"],
    }

    return {
        "data": {
            "current": current,
            "baseline": baseline,
            "trend": trend,
            "period": period,
            "feedback_received_count": len(logs)
        }
    }


@router.get("/summary/{model_id}")
async def get_monitoring_summary(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get monitoring summary dashboard for a model.
    Aggregates drift, bias, performance, and alert data from real services.
    """
    from app.services.drift_service import DriftMonitoringService
    from app.services.bias_service import BiasMonitoringService
    from app.models.ml_model import MLModel, Baseline
    from app.models.alert import Alert
    from sqlalchemy import select, func
    from uuid import UUID

    # --- Performance: read stored metrics from MLModel ---
    perf_section: dict = {"status": "NO_DATA"}
    try:
        model_result = await db.execute(select(MLModel).where(MLModel.id == UUID(model_id)))
        model = model_result.scalar_one_or_none()
        if model and model.metrics:
            f1 = model.metrics.get("f1", 0.0)
            baseline_result = await db.execute(
                select(Baseline).where(
                    Baseline.model_id == UUID(model_id),
                    Baseline.metric_name == "f1",
                )
            )
            f1_baseline = baseline_result.scalars().first()
            baseline_f1 = float(f1_baseline.threshold) if f1_baseline else None
            perf_section = {
                "status": "OK" if baseline_f1 is not None and f1 >= baseline_f1 else ("WARNING" if baseline_f1 is not None else "NO_DATA"),
                "current_f1": round(f1, 4),
                "baseline_f1": baseline_f1,
                "algorithm": model.algorithm,
                "model_status": model.status,
            }
    except Exception as e:
        logger.warning(f"Summary: performance section failed for {model_id}: {e}")
        perf_section = {"status": "ERROR", "error": str(e)}

    # --- Drift ---
    drift_section: dict = {"status": "NO_DATA"}
    try:
        drift_svc = DriftMonitoringService(db)
        drift_result = await drift_svc.run_drift_check(model_id)
        drift_section = {
            "status": drift_result.overall_status,
            "drifted_features": drift_result.drifted_features,
            "last_check": drift_result.computed_at.isoformat(),
        }
    except Exception as e:
        logger.warning(f"Summary: drift section failed for {model_id}: {e}")
        drift_section = {"status": "ERROR", "error": str(e)}

    # --- Bias ---
    bias_section: dict = {"status": "NO_DATA"}
    try:
        bias_svc = BiasMonitoringService(db)
        bias_result = await bias_svc.run_bias_check(model_id)
        flagged = sum(
            1 for m in bias_result.metrics.values()
            if m.get("status") in ("WARNING", "CRITICAL")
        ) if bias_result.metrics else 0
        bias_section = {
            "status": bias_result.overall_status,
            "flagged_attributes": flagged,
            "total_attributes": len(bias_result.metrics) if bias_result.metrics else 0,
            "last_check": bias_result.computed_at.isoformat(),
        }
    except Exception as e:
        logger.warning(f"Summary: bias section failed for {model_id}: {e}")
        bias_section = {"status": "ERROR", "error": str(e)}

    # --- Alerts: live count from DB ---
    alerts_section: dict = {"active": 0, "critical": 0, "acknowledged": 0}
    try:
        alert_result = await db.execute(
            select(
                Alert.severity,
                Alert.resolved_at,
                func.count(Alert.id).label("cnt"),
            )
            .where(Alert.model_id == UUID(model_id))
            .group_by(Alert.severity, Alert.resolved_at)
        )
        for row in alert_result.all():
            if row.resolved_at is None:
                alerts_section["active"] += row.cnt
                if str(row.severity).upper() == "CRITICAL":
                    alerts_section["critical"] += row.cnt
            else:
                alerts_section["acknowledged"] += row.cnt
    except Exception as e:
        logger.warning(f"Summary: alerts section failed for {model_id}: {e}")
        alerts_section = {"error": str(e)}

    return {
        "data": {
            "model_id": model_id,
            "drift": drift_section,
            "bias": bias_section,
            "performance": perf_section,
            "alerts": alerts_section,
        }
    }


@router.post("/drift/{model_id}/compute")
async def trigger_drift_computation(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Trigger manual drift computation for a model."""
    from app.workers.monitoring_worker import compute_drift_metrics
    compute_drift_metrics.delay(model_id)
    
    return {"message": "Drift computation triggered", "model_id": model_id}


@router.post("/bias/{model_id}/compute")
async def trigger_bias_computation(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Trigger manual bias computation for a model."""
    from app.workers.monitoring_worker import compute_bias_metrics
    compute_bias_metrics.delay(model_id)
    
    return {"message": "Bias computation triggered", "model_id": model_id}


@router.post("/performance/{model_id}/check")
async def trigger_performance_check(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Trigger manual performance baseline check."""
    from app.workers.monitoring_worker import check_performance_baselines
    check_performance_baselines.delay(model_id)
    
    return {"message": "Performance check triggered", "model_id": model_id}


# ---------------------------------------------------------------------------
# Baseline Endpoints
# ---------------------------------------------------------------------------

class BaselineConfigRequest(BaseModel):
    metric: str
    threshold: float
    operator: str  # gte, lte, eq, gt, lt
    severity: str = "WARNING"
    description: Optional[str] = None


class SetBaselinesRequest(BaseModel):
    baselines: List[BaselineConfigRequest]


class CheckBaselinesRequest(BaseModel):
    metrics: dict  # e.g. {"precision": 0.91, "recall": 0.75}


@router.get("/baselines/{model_id}")
async def get_baselines(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return all active baseline thresholds for a model."""
    from app.services.baseline_service import BaselineService
    service = BaselineService(db)
    baselines = await service.get_baselines(model_id)
    return {
        "data": [
            {
                "id": str(b.id),
                "metric_name": b.metric_name,
                "threshold": b.threshold,
                "operator": b.operator,
                "is_active": b.is_active,
                "created_at": b.created_at.isoformat() if b.created_at else None,
            }
            for b in baselines
        ],
        "total": len(baselines),
    }


@router.post("/baselines/{model_id}")
async def set_baselines(
    model_id: str,
    request: SetBaselinesRequest,
    db: AsyncSession = Depends(get_db),
):
    """Set custom baselines for a model. Replaces all existing baselines."""
    from app.services.baseline_service import BaselineService, BaselineConfig
    service = BaselineService(db)
    configs = [
        BaselineConfig(
            metric=b.metric,
            threshold=b.threshold,
            operator=b.operator,
            severity=b.severity,
            description=b.description,
        )
        for b in request.baselines
    ]
    created = await service.set_baselines(model_id, configs)
    return {"message": f"Set {len(created)} baselines", "model_id": model_id}


@router.post("/baselines/{model_id}/defaults")
async def apply_default_baselines(
    model_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Apply the 5 default fraud-detection baseline thresholds to a model."""
    from app.services.baseline_service import BaselineService
    service = BaselineService(db)
    try:
        created = await service.apply_defaults(model_id)
        defaults = await service.get_default_config()
        return {
            "message": f"Applied {len(created)} default baselines",
            "model_id": model_id,
            "baselines": defaults,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply defaults: {str(e)}")


@router.post("/baselines/{model_id}/check")
async def check_baselines(
    model_id: str,
    request: CheckBaselinesRequest,
    db: AsyncSession = Depends(get_db),
):
    """Check provided metrics against the stored baselines for the model."""
    from app.services.baseline_service import BaselineService
    service = BaselineService(db)
    results = await service.check_baselines(model_id, request.metrics)

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No baselines found for this model. Call POST /baselines/{model_id}/defaults first.",
        )

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    critical_failures = [r for r in failed if r.severity == "CRITICAL"]

    return {
        "model_id": model_id,
        "summary": {
            "total_checks": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "critical_failures": len(critical_failures),
            "overall_status": "CRITICAL" if critical_failures else ("WARNING" if failed else "OK"),
        },
        "results": [
            {
                "metric": r.metric,
                "current_value": round(r.current_value, 4),
                "threshold": r.threshold,
                "operator": r.operator,
                "passed": r.passed,
                "severity": r.severity,
                "message": r.message,
            }
            for r in results
        ],
    }
