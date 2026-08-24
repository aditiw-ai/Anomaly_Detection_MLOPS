"""
Baseline Configuration Service
Set and validate performance baselines for model monitoring.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from uuid import UUID
from datetime import datetime
import logging

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ml_model import MLModel, Baseline

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Baseline configuration for a metric."""
    metric: str
    threshold: float
    operator: str  # gte, lte, eq
    severity: str = "WARNING"  # WARNING, CRITICAL
    description: Optional[str] = None


@dataclass
class BaselineCheckResult:
    """Result of checking current metrics against baselines."""
    metric: str
    current_value: float
    threshold: float
    operator: str
    passed: bool
    severity: str
    message: str


class BaselineService:
    """
    Service for managing performance baselines.
    
    Baselines are thresholds that trigger alerts when breached.
    """
    
    # Default baselines for fraud detection
    DEFAULT_BASELINES = [
        BaselineConfig(
            metric="precision",
            threshold=0.85,
            operator="gte",
            severity="WARNING",
            description="Precision should stay above 85%",
        ),
        BaselineConfig(
            metric="recall",
            threshold=0.80,
            operator="gte",
            severity="CRITICAL",
            description="Recall is critical for fraud detection",
        ),
        BaselineConfig(
            metric="f1",
            threshold=0.82,
            operator="gte",
            severity="WARNING",
            description="F1 score should remain balanced",
        ),
        BaselineConfig(
            metric="auc",
            threshold=0.90,
            operator="gte",
            severity="WARNING",
            description="AUC should stay above 90%",
        ),
        BaselineConfig(
            metric="fpr",
            threshold=0.10,
            operator="lte",
            severity="WARNING",
            description="False positive rate should stay below 10%",
        ),
    ]
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_baselines(self, model_id: str) -> List[Baseline]:
        """Get all baselines for a model."""
        try:
            uuid_id = UUID(model_id)
        except ValueError:
            return []
        
        result = await self.db.execute(
            select(Baseline).where(Baseline.model_id == uuid_id)
        )
        return result.scalars().all()
    
    async def set_baselines(
        self,
        model_id: str,
        baselines: List[BaselineConfig],
    ) -> List[Baseline]:
        """
        Set baselines for a model.
        
        Replaces existing baselines with new configuration.
        """
        uuid_id = UUID(model_id)
        
        # Delete existing baselines
        await self.db.execute(
            delete(Baseline).where(Baseline.model_id == uuid_id)
        )
        
        # Create new baselines
        created = []
        for config in baselines:
            baseline = Baseline(
                model_id=uuid_id,
                metric_name=config.metric,
                threshold=config.threshold,
                operator=config.operator,
            )
            self.db.add(baseline)
            created.append(baseline)
        
        await self.db.commit()
        
        logger.info(f"Set {len(created)} baselines for model {model_id}")
        return created
    
    async def apply_defaults(self, model_id: str) -> List[Baseline]:
        """Apply default baselines to a model."""
        return await self.set_baselines(model_id, self.DEFAULT_BASELINES)
    
    async def check_baselines(
        self,
        model_id: str,
        current_metrics: Dict[str, float],
    ) -> List[BaselineCheckResult]:
        """
        Check current metrics against baselines.
        
        Returns list of check results with pass/fail status.
        """
        baselines = await self.get_baselines(model_id)
        
        results = []
        for baseline in baselines:
            metric_name = baseline.metric_name
            
            if metric_name not in current_metrics:
                results.append(BaselineCheckResult(
                    metric=metric_name,
                    current_value=0.0,
                    threshold=baseline.threshold,
                    operator=baseline.operator,
                    passed=False,
                    severity="WARNING",
                    message=f"Metric '{metric_name}' not found in current metrics",
                ))
                continue
            
            current_value = current_metrics[metric_name]
            passed = self._evaluate_baseline(
                current_value,
                baseline.threshold,
                baseline.operator,
            )
            
            # Get severity from config match
            severity = "WARNING"
            for cfg in self.DEFAULT_BASELINES:
                if cfg.metric == metric_name:
                    severity = cfg.severity
                    break
            
            if passed:
                message = f"{metric_name}: {current_value:.4f} meets threshold {baseline.operator} {baseline.threshold}"
            else:
                message = f"{metric_name}: {current_value:.4f} violates threshold {baseline.operator} {baseline.threshold}"
            
            results.append(BaselineCheckResult(
                metric=metric_name,
                current_value=current_value,
                threshold=baseline.threshold,
                operator=baseline.operator,
                passed=passed,
                severity=severity,
                message=message,
            ))
        
        return results
    
    def _evaluate_baseline(
        self,
        current: float,
        threshold: float,
        operator: str,
    ) -> bool:
        """Evaluate if current value passes the baseline."""
        if operator == "gte":
            return current >= threshold
        elif operator == "lte":
            return current <= threshold
        elif operator == "eq":
            return abs(current - threshold) < 0.0001
        elif operator == "gt":
            return current > threshold
        elif operator == "lt":
            return current < threshold
        else:
            logger.warning(f"Unknown operator: {operator}, defaulting to True")
            return True
    
    async def get_default_config(self) -> List[Dict]:
        """Get default baseline configuration."""
        return [
            {
                "metric": cfg.metric,
                "threshold": cfg.threshold,
                "operator": cfg.operator,
                "severity": cfg.severity,
                "description": cfg.description,
            }
            for cfg in self.DEFAULT_BASELINES
        ]
