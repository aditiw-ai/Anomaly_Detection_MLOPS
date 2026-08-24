"""
Bias Monitoring Service
Production bias detection using real model predictions on the test split.
"""
from app.core.time import IST, now_ist
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from uuid import UUID
from datetime import datetime
import logging
import pandas as pd
from io import BytesIO

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ml_model import MLModel

logger = logging.getLogger(__name__)

# Common protected attribute column name patterns to look for in data
PROTECTED_ATTR_PATTERNS = [
    "gender", "sex", "age", "age_group", "age_band",
    "region", "nationality", "ethnicity", "race",
    "marital_status", "education",
]


@dataclass
class BiasMonitoringResult:
    """Result of a bias monitoring run."""
    model_id: str
    computed_at: datetime
    overall_status: str
    attributes_checked: int
    metrics: Dict[str, Any]


@dataclass
class BiasAlert:
    """Alert for bias detection."""
    attribute: str
    severity: str  # WARNING, CRITICAL
    message: str
    details: Dict[str, Any]


class BiasMonitoringService:
    """
    Service for production bias monitoring.

    Uses the stored test split (real data) and the trained pickle model
    to generate predictions, then computes fairness metrics across
    any detectable protected attributes.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def run_bias_check(self, model_id: str) -> BiasMonitoringResult:
        """
        Run bias detection for a model.

        Steps:
        1. Load MLModel + linked TrainingJob from DB.
        2. Download test split parquet from Azure Blob.
        3. Load pickle model from Azure Blob.
        4. Generate predictions on test features.
        5. Run BiasDetector on detected protected attributes.
        6. Return real BiasMonitoringResult.
        """
        logger.info(f"Running bias check for model {model_id}")

        model = await self._get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        metrics = await self._compute_bias_metrics(model)
        alerts = self._generate_alerts(metrics)

        # Overall status
        statuses = [v.get("status", "OK") for v in metrics.values()]
        if "CRITICAL" in statuses:
            overall_status = "CRITICAL"
        elif "WARNING" in statuses:
            overall_status = "WARNING"
        else:
            overall_status = "OK"

        # Persist alerts (and trigger Slack webhook notifications) when bias is detected.
        if alerts:
            await self._create_alerts(model_id=model_id, alerts=alerts)

        return BiasMonitoringResult(
            model_id=model_id,
            computed_at=now_ist(),
            overall_status=overall_status,
            attributes_checked=len(metrics),
            metrics=metrics,
        )

    async def _get_model(self, model_id: str) -> Optional[MLModel]:
        try:
            uuid_id = UUID(model_id)
        except ValueError:
            return None
        result = await self.db.execute(
            select(MLModel).where(MLModel.id == uuid_id)
        )
        return result.scalar_one_or_none()

    async def _compute_bias_metrics(self, model: MLModel) -> Dict[str, Any]:
        """
        Load test split + pickle model → generate predictions → compute bias.
        Returns empty dict with a warning if data or protected attributes
        are unavailable.
        """
        from app.models.training_job import TrainingJob
        from app.models.dataset import Dataset
        from app.core.storage import storage_service
        from ml.bias.bias_detector import BiasDetector, BiasConfig
        from ml.transformers.column_role_detector import ColumnRoleDetector
        import pickle
        import numpy as np

        try:
            # 1. Get linked TrainingJob
            job_result = await self.db.execute(
                select(TrainingJob).where(TrainingJob.model_id == model.id)
            )
            job = job_result.scalar_one_or_none()
            if not job:
                logger.warning(f"No TrainingJob linked to model {model.id}.")
                return {}

            test_dataset_id = job.metrics.get("test_dataset_id")
            if not test_dataset_id:
                logger.warning(f"No test_dataset_id in job metrics for model {model.id}.")
                return {}

            # 2. Resolve Dataset record
            ds_result = await self.db.execute(
                select(Dataset).where(Dataset.id == UUID(test_dataset_id))
            )
            test_ds = ds_result.scalar_one_or_none()
            if not test_ds:
                logger.warning(f"Test dataset {test_dataset_id} not found in DB.")
                return {}

            # 3. Download helpers (sync blob client)
            def _download_parquet(storage_path: str) -> pd.DataFrame:
                parts = storage_path.split("/", 1)
                container, blob_name = parts[0], parts[1]
                client = storage_service.client.get_blob_client(
                    container=container, blob=blob_name
                )
                return pd.read_parquet(BytesIO(client.download_blob().readall()))

            def _download_bytes(storage_path: str) -> bytes:
                parts = storage_path.split("/", 1)
                container, blob_name = parts[0], parts[1]
                client = storage_service.client.get_blob_client(
                    container=container, blob=blob_name
                )
                return client.download_blob().readall()

            # 4. Load test split
            test_df = _download_parquet(test_ds.storage_path)

            # 5. Detect target column
            detector = ColumnRoleDetector()
            roles = detector.detect(test_df)
            target_col = roles.target_col
            if not target_col:
                common_targets = ["is_fraud", "fraud_label", "target", "label", "is_fraudulent"]
                target_col = next((c for c in common_targets if c in test_df.columns), None)
            if not target_col:
                logger.warning(f"Cannot detect target column in test data for model {model.id}.")
                return {}

            test_df = test_df.dropna(subset=[target_col])
            y_true = test_df[target_col].values.astype(int)

            # 6. Load pickle model and generate predictions
            model_bytes = _download_bytes(model.storage_path)
            pipeline = pickle.loads(model_bytes)

            feature_cols = [
                c for c in (model.feature_names or [])
                if c in test_df.columns
            ] or [
                c for c in test_df.columns
                if c != target_col
                and pd.api.types.is_numeric_dtype(test_df[c])
            ]

            X_test = test_df[feature_cols].fillna(0)
            y_pred = pipeline.predict(X_test).astype(int)

            # 7. Detect protected attributes present in the test data
            all_cols = set(test_df.columns)
            detected_attrs = [
                col for pattern in PROTECTED_ATTR_PATTERNS
                for col in all_cols
                if pattern in col.lower()
            ]

            if not detected_attrs:
                logger.info(
                    f"No protected attributes found in test data for model {model.id}. "
                    f"Columns checked: {sorted(all_cols)}"
                )
                return {}

            # 8. Run BiasDetector
            config = BiasConfig(protected_attributes=detected_attrs)
            bias_detector = BiasDetector(config)
            results = bias_detector.compute_bias(
                y_true=y_true,
                y_pred=y_pred,
                protected_features=test_df[detected_attrs].reset_index(drop=True),
            )

            # 9. Convert BiasResult dataclasses → dict (matches existing API response shape)
            metrics: Dict[str, Any] = {}
            for attr, r in results.items():
                metrics[attr] = {
                    "demographic_parity_diff": r.demographic_parity_diff,
                    "equalized_odds_diff": r.equalized_odds_diff,
                    "disparate_impact": r.disparate_impact,
                    "status": r.status,
                    "group_rates": r.group_rates,
                }

            logger.info(
                f"Bias computed for {len(metrics)} protected attributes on model {model.id}"
            )
            return metrics

        except Exception as e:
            logger.error(
                f"Failed to compute bias metrics for model {model.id}: {e}",
                exc_info=True,
            )
            return {}

    def _generate_alerts(self, metrics: Dict[str, Any]) -> List[BiasAlert]:
        """Generate bias alerts from computed protected-attribute metrics."""
        alerts: List[BiasAlert] = []
        for attr, m in metrics.items():
            status = str(m.get("status", "OK")).upper()
            if status not in ("WARNING", "CRITICAL"):
                continue

            di = float(m.get("disparate_impact", 0.0) or 0.0)
            dp = float(m.get("demographic_parity_diff", 0.0) or 0.0)

            alerts.append(
                BiasAlert(
                    attribute=attr,
                    severity=status,
                    message=(
                        f"Protected attribute '{attr}' shows {status} bias — "
                        f"Disparate Impact={di:.3f}, DP Diff={dp:.3f}"
                    ),
                    details={
                        "attribute": attr,
                        "disparate_impact": di,
                        "demographic_parity_diff": dp,
                    },
                )
            )
        return alerts

    async def _create_alerts(self, model_id: str, alerts: List[BiasAlert]) -> None:
        """Persist bias alerts using AlertService (includes Slack notifications)."""
        from app.services.alert_service import AlertService, AlertCreate
        from app.models.alert import AlertType, AlertSeverity

        alert_svc = AlertService(self.db)
        for alert in alerts:
            severity = AlertSeverity.CRITICAL if alert.severity == "CRITICAL" else AlertSeverity.WARNING
            try:
                await alert_svc.create_alert(
                    AlertCreate(
                        model_id=model_id,
                        alert_type=AlertType.BIAS,
                        severity=severity,
                        title=f"Bias Detected: {alert.attribute}",
                        message=alert.message,
                        details=alert.details,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to persist bias alert for {alert.attribute}: {e}")
