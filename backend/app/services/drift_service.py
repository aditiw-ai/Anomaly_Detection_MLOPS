"""
Drift Monitoring Service
Production drift detection and alerting.
"""
from app.core.time import IST, now_ist
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from uuid import UUID
from datetime import datetime, timedelta
import logging
import json
import os
import pandas as pd
from io import BytesIO

from sqlalchemy import select, insert, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ml_model import MLModel

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert for drift detection."""
    feature: str
    metric: str  # psi, ks_statistic
    current_value: float
    threshold: float
    severity: str  # WARNING, CRITICAL
    message: str


@dataclass
class DriftMonitoringResult:
    """Result of drift monitoring run."""
    model_id: str
    computed_at: datetime
    overall_status: str
    feature_count: int
    drifted_features: int
    alerts: List[DriftAlert]
    metrics: Dict[str, Any]


class DriftMonitoringService:
    """
    Service for production drift monitoring.
    
    Monitors:
    - Data drift (PSI, KS-test)
    - Concept drift (performance degradation)
    - Feature distribution changes
    """
    
    # Default thresholds
    PSI_WARNING = 0.1
    PSI_CRITICAL = 0.25
    KS_ALPHA = 0.05
    _HISTORY_TTL_SECONDS = 60 * 60 * 24 * 90  # 90 days
    _MAX_HISTORY_POINTS = 500
    
    def __init__(self, db: AsyncSession):
        self.db = db

    def _get_redis_client(self):
        """Build a short-timeout Redis client for drift history snapshots."""
        try:
            import redis
            url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            return redis.from_url(url, decode_responses=True, socket_timeout=2)
        except Exception as exc:
            logger.warning(f"Drift history Redis init failed: {exc}")
            return None

    def _history_key(self, model_id: str) -> str:
        return f"monitoring:drift:history:{model_id}"
    
    async def run_drift_check(
        self,
        model_id: str,
        reference_data: Optional[Any] = None,
        current_data: Optional[Any] = None,
    ) -> DriftMonitoringResult:
        """
        Run drift detection for a model.
        
        Args:
            model_id: Model to check
            reference_data: Training/reference data
            current_data: Production/current data
        
        Returns:
            DriftMonitoringResult with metrics and alerts
        """
        logger.info(f"Running drift check for model {model_id}")
        
        # Get model info
        model = await self._get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Compute real drift metrics from reference/test splits
        drift_metrics = await self._compute_drift_metrics(model)
        
        # Generate alerts based on thresholds
        alerts = self._generate_alerts(drift_metrics)
        
        # Determine overall status
        if any(a.severity == "CRITICAL" for a in alerts):
            overall_status = "CRITICAL"
        elif any(a.severity == "WARNING" for a in alerts):
            overall_status = "WARNING"
        else:
            overall_status = "OK"
        
        # Store metrics in database
        await self._store_drift_metrics(model_id, drift_metrics, overall_status)
        
        # Create alerts in database
        if alerts:
            await self._create_alerts(model_id, alerts)
        
        result = DriftMonitoringResult(
            model_id=model_id,
            computed_at=now_ist(),
            overall_status=overall_status,
            feature_count=len(drift_metrics),
            drifted_features=len([m for m in drift_metrics.values() if m.get("status") != "OK"]),
            alerts=alerts,
            metrics=drift_metrics,
        )
        
        logger.info(f"Drift check complete: {overall_status}, {len(alerts)} alerts")
        return result
    
    async def _get_model(self, model_id: str) -> Optional[MLModel]:
        """Get model by ID."""
        try:
            uuid_id = UUID(model_id)
        except ValueError:
            return None
        
        result = await self.db.execute(
            select(MLModel).where(MLModel.id == uuid_id)
        )
        return result.scalar_one_or_none()
    
    async def _compute_drift_metrics(self, model: MLModel) -> Dict[str, Any]:
        """
        Compute real drift metrics using DataDriftDetector on train/test splits.
        Falls back to empty dict if data is unavailable.
        """
        from app.models.training_job import TrainingJob
        from app.models.dataset import Dataset
        from app.core.storage import storage_service
        from ml.drift.data_drift import DataDriftDetector, DriftConfig

        try:
            # 1. Find the TrainingJob linked to this model
            job_result = await self.db.execute(
                select(TrainingJob).where(TrainingJob.model_id == model.id)
            )
            job = job_result.scalar_one_or_none()
            if not job:
                logger.warning(f"No TrainingJob found for model {model.id}, cannot compute drift.")
                return {}

            train_dataset_id = job.metrics.get("train_dataset_id")
            test_dataset_id = job.metrics.get("test_dataset_id")
            if not train_dataset_id or not test_dataset_id:
                logger.warning(f"Missing dataset IDs in job metrics for model {model.id}.")
                return {}

            # 2. Resolve storage paths
            from uuid import UUID as _UUID
            ds_result = await self.db.execute(
                select(Dataset).where(Dataset.id.in_([
                    _UUID(train_dataset_id), _UUID(test_dataset_id)
                ]))
            )
            datasets = {str(d.id): d for d in ds_result.scalars().all()}

            train_ds = datasets.get(train_dataset_id)
            test_ds = datasets.get(test_dataset_id)
            if not train_ds or not test_ds:
                logger.warning(f"Dataset records not found for model {model.id}.")
                return {}

            # 3. Download parquet files from Azure Blob (sync client)
            def _download(storage_path: str) -> pd.DataFrame:
                parts = storage_path.split("/", 1)
                container, blob_name = parts[0], parts[1]
                client = storage_service.client.get_blob_client(container=container, blob=blob_name)
                data = client.download_blob().readall()
                return pd.read_parquet(BytesIO(data))

            reference_df = _download(train_ds.storage_path)
            current_df   = _download(test_ds.storage_path)

            # 4. Use model's feature names; fallback to shared numeric columns
            features = model.feature_names or list(
                set(reference_df.columns) & set(current_df.columns)
            )
            # Only numeric features that exist in both
            numeric_features = [
                f for f in features
                if f in reference_df.columns
                and f in current_df.columns
                and pd.api.types.is_numeric_dtype(reference_df[f])
            ]

            if not numeric_features:
                logger.warning(f"No overlapping numeric features for drift check on model {model.id}.")
                return {}

            # 5. Run real drift detection
            detector = DataDriftDetector(DriftConfig())
            results = detector.compute_drift(reference_df, current_df, numeric_features)

            # 6. Convert DriftResult dataclasses → dict (same shape as before)
            metrics = {}
            for feature, r in results.items():
                if r.psi > self.PSI_CRITICAL:
                    trend = "increasing"
                elif r.psi > self.PSI_WARNING:
                    trend = "stable"
                else:
                    trend = "stable"

                metrics[feature] = {
                    "psi": r.psi,
                    "ks_statistic": r.ks_statistic,
                    "ks_p_value": r.ks_p_value,
                    "status": r.status,
                    "trend": trend,
                }

            logger.info(f"Drift computed for {len(metrics)} features on model {model.id}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to compute real drift metrics for model {model.id}: {e}", exc_info=True)
            return {}
    
    def _generate_alerts(self, metrics: Dict[str, Any]) -> List[DriftAlert]:
        """Generate alerts from drift metrics."""
        alerts = []
        
        for feature, m in metrics.items():
            psi = m.get("psi", 0)
            ks_stat = m.get("ks_statistic", 0)
            ks_p = m.get("ks_p_value", 1.0)
            
            # PSI-based alerts
            if psi > self.PSI_CRITICAL:
                alerts.append(DriftAlert(
                    feature=feature,
                    metric="psi",
                    current_value=psi,
                    threshold=self.PSI_CRITICAL,
                    severity="CRITICAL",
                    message=f"Critical drift in '{feature}': PSI={psi:.3f} exceeds {self.PSI_CRITICAL}",
                ))
            elif psi > self.PSI_WARNING:
                alerts.append(DriftAlert(
                    feature=feature,
                    metric="psi",
                    current_value=psi,
                    threshold=self.PSI_WARNING,
                    severity="WARNING",
                    message=f"Drift detected in '{feature}': PSI={psi:.3f} exceeds {self.PSI_WARNING}",
                ))
            
            # KS-test alerts
            if ks_p < self.KS_ALPHA and ks_stat > 0.1:
                alerts.append(DriftAlert(
                    feature=feature,
                    metric="ks_statistic",
                    current_value=ks_stat,
                    threshold=self.KS_ALPHA,
                    severity="WARNING",
                    message=f"Distribution shift in '{feature}': KS={ks_stat:.3f}, p={ks_p:.4f}",
                ))
        
        return alerts
    
    async def _store_drift_metrics(
        self,
        model_id: str,
        metrics: Dict[str, Any],
        status: str,
    ):
        """Persist drift snapshots in Redis for historical trend endpoints."""
        payload = {
            "computed_at": now_ist().isoformat(),
            "overall_status": status,
            "metrics": metrics or {},
        }
        client = self._get_redis_client()
        if not client:
            logger.info(f"Stored drift metrics in-memory only for model {model_id}: {status}")
            return
        try:
            key = self._history_key(model_id)
            client.lpush(key, json.dumps(payload, default=str))
            client.ltrim(key, 0, self._MAX_HISTORY_POINTS - 1)
            client.expire(key, self._HISTORY_TTL_SECONDS)
            logger.info(f"Persisted drift snapshot for model {model_id}: {status}")
        except Exception as exc:
            logger.warning(f"Failed to persist drift snapshot for model {model_id}: {exc}")
    
    async def _create_alerts(
        self,
        model_id: str,
        alerts: List[DriftAlert],
    ):
        """Persist drift alerts to the database via AlertService."""
        from app.services.alert_service import AlertService
        alert_svc = AlertService(self.db)
        for alert in alerts:
            try:
                await alert_svc.create_drift_alert(
                    model_id=model_id,
                    feature=alert.feature,
                    psi=alert.current_value,
                    threshold=alert.threshold,
                )
                logger.info(f"Alert persisted: {alert.message}")
            except Exception as e:
                logger.error(f"Failed to persist drift alert for {alert.feature}: {e}")
    
    async def get_drift_history(
        self,
        model_id: str,
        days: int = 7,
    ) -> List[Dict]:
        """Get real drift history from persisted snapshots."""
        client = self._get_redis_client()
        day_keys = [
            (now_ist() - timedelta(days=(days - 1 - i))).strftime("%Y-%m-%d")
            for i in range(days)
        ]
        if not client:
            return [
                {
                    "date": d,
                    "overall_status": "NO_DATA",
                    "drifted_features": 0,
                    "avg_psi": 0.0,
                    "volume_processed": 0,
                }
                for d in day_keys
            ]

        try:
            key = self._history_key(model_id)
            raw_items = client.lrange(key, 0, self._MAX_HISTORY_POINTS - 1)
            per_day: Dict[str, Dict[str, Any]] = {}
            cutoff = now_ist() - timedelta(days=days)

            for raw in raw_items:
                try:
                    snap = json.loads(raw)
                    ts_raw = snap.get("computed_at")
                    ts = datetime.fromisoformat(ts_raw) if ts_raw else None
                    if not ts:
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=IST)
                    if ts < cutoff:
                        continue

                    day = ts.strftime("%Y-%m-%d")
                    metrics = snap.get("metrics") or {}
                    psi_values = [
                        float(m.get("psi", 0.0))
                        for m in metrics.values()
                        if isinstance(m, dict)
                    ]
                    drifted = [
                        m for m in metrics.values()
                        if isinstance(m, dict) and m.get("status") in ("WARNING", "CRITICAL")
                    ]
                    candidate = {
                        "ts": ts,
                        "date": day,
                        "overall_status": snap.get("overall_status", "NO_DATA"),
                        "drifted_features": len(drifted),
                        "avg_psi": round(sum(psi_values) / len(psi_values), 4) if psi_values else 0.0,
                        "volume_processed": len(metrics),
                    }
                    prev = per_day.get(day)
                    if not prev or candidate["ts"] > prev["ts"]:
                        per_day[day] = candidate
                except Exception:
                    continue

            out = []
            for day in day_keys:
                row = per_day.get(day)
                if row:
                    out.append({
                        "date": day,
                        "overall_status": row["overall_status"],
                        "drifted_features": row["drifted_features"],
                        "avg_psi": row["avg_psi"],
                        "volume_processed": row["volume_processed"],
                    })
                else:
                    out.append({
                        "date": day,
                        "overall_status": "NO_DATA",
                        "drifted_features": 0,
                        "avg_psi": 0.0,
                        "volume_processed": 0,
                    })
            return out
        except Exception as exc:
            logger.warning(f"Failed to read drift history for {model_id}: {exc}")
            return [
                {
                    "date": d,
                    "overall_status": "NO_DATA",
                    "drifted_features": 0,
                    "avg_psi": 0.0,
                    "volume_processed": 0,
                }
                for d in day_keys
            ]
    
    async def get_feature_drift_trend(
        self,
        model_id: str,
        feature: str,
        days: int = 7,
    ) -> List[Dict]:
        """Get real feature drift trend from persisted snapshots."""
        client = self._get_redis_client()
        day_keys = [
            (now_ist() - timedelta(days=(days - 1 - i))).strftime("%Y-%m-%d")
            for i in range(days)
        ]
        if not client:
            return [{"date": d, "psi": 0.0, "ks_statistic": 0.0} for d in day_keys]

        try:
            key = self._history_key(model_id)
            raw_items = client.lrange(key, 0, self._MAX_HISTORY_POINTS - 1)
            per_day: Dict[str, Dict[str, Any]] = {}
            cutoff = now_ist() - timedelta(days=days)

            for raw in raw_items:
                try:
                    snap = json.loads(raw)
                    ts_raw = snap.get("computed_at")
                    ts = datetime.fromisoformat(ts_raw) if ts_raw else None
                    if not ts:
                        continue
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=IST)
                    if ts < cutoff:
                        continue

                    feature_metric = (snap.get("metrics") or {}).get(feature)
                    if not isinstance(feature_metric, dict):
                        continue

                    day = ts.strftime("%Y-%m-%d")
                    candidate = {
                        "ts": ts,
                        "psi": round(float(feature_metric.get("psi", 0.0)), 4),
                        "ks_statistic": round(float(feature_metric.get("ks_statistic", 0.0)), 4),
                    }
                    prev = per_day.get(day)
                    if not prev or candidate["ts"] > prev["ts"]:
                        per_day[day] = candidate
                except Exception:
                    continue

            out = []
            for day in day_keys:
                row = per_day.get(day)
                if row:
                    out.append({
                        "date": day,
                        "psi": row["psi"],
                        "ks_statistic": row["ks_statistic"],
                    })
                else:
                    out.append({
                        "date": day,
                        "psi": 0.0,
                        "ks_statistic": 0.0,
                    })
            return out
        except Exception as exc:
            logger.warning(f"Failed to read feature drift trend for {model_id}/{feature}: {exc}")
            return [{"date": d, "psi": 0.0, "ks_statistic": 0.0} for d in day_keys]
