"""
Model Comparison Service
Compare performance metrics between models.
"""
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from uuid import UUID
import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ml_model import MLModel

logger = logging.getLogger(__name__)


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""
    metric: str
    model_a_value: Optional[float]
    model_b_value: Optional[float]
    difference: Optional[float]
    percent_change: Optional[float]
    winner: Optional[str]  # model_a, model_b, or tie
    significant: bool  # Is the difference significant?


@dataclass
class ComparisonResult:
    """Full comparison result between two models."""
    model_a_id: str
    model_a_name: str
    model_b_id: str
    model_b_name: str
    metrics: List[MetricComparison]
    overall_winner: Optional[str]
    recommendation: str


class ModelComparisonService:
    """
    Service for comparing ML models.
    
    Compares:
    - Performance metrics (precision, recall, F1, AUC)
    - Feature importance rankings
    - Inference latency
    - Model size
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.significance_threshold = 0.01  # 1% difference threshold
    
    async def compare_models(
        self,
        model_a_id: str,
        model_b_id: str,
    ) -> ComparisonResult:
        """
        Compare two models by their metrics.
        
        Args:
            model_a_id: First model ID
            model_b_id: Second model ID
        
        Returns:
            ComparisonResult with detailed metric comparisons
        """
        # Fetch models
        model_a = await self._get_model(model_a_id)
        model_b = await self._get_model(model_b_id)
        
        if not model_a or not model_b:
            raise ValueError("One or both models not found")
        
        # Compare each metric
        metrics = []
        all_metric_names = set(model_a.metrics.keys()) | set(model_b.metrics.keys())
        
        for metric_name in all_metric_names:
            comparison = self._compare_metric(
                metric_name,
                model_a.metrics.get(metric_name),
                model_b.metrics.get(metric_name),
                model_a_id,
                model_b_id,
            )
            metrics.append(comparison)
        
        # Determine overall winner
        overall_winner, recommendation = self._determine_winner(metrics, model_a, model_b)
        
        return ComparisonResult(
            model_a_id=model_a_id,
            model_a_name=model_a.name,
            model_b_id=model_b_id,
            model_b_name=model_b.name,
            metrics=metrics,
            overall_winner=overall_winner,
            recommendation=recommendation,
        )
    
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
    
    def _compare_metric(
        self,
        metric_name: str,
        value_a: Optional[float],
        value_b: Optional[float],
        model_a_id: str,
        model_b_id: str,
    ) -> MetricComparison:
        """Compare a single metric between two models."""
        if value_a is None or value_b is None:
            return MetricComparison(
                metric=metric_name,
                model_a_value=value_a,
                model_b_value=value_b,
                difference=None,
                percent_change=None,
                winner=None,
                significant=False,
            )
        
        difference = value_a - value_b
        
        # Calculate percent change (relative to model_b as baseline)
        if value_b != 0:
            percent_change = (difference / value_b) * 100
        else:
            percent_change = 100 if value_a > 0 else 0
        
        # Determine winner (higher is better for most metrics)
        higher_is_better = metric_name.lower() not in ["fpr", "false_positive_rate", "loss"]
        
        if abs(percent_change) < self.significance_threshold * 100:
            winner = None  # Tie
            significant = False
        elif higher_is_better:
            winner = model_a_id if difference > 0 else model_b_id
            significant = abs(percent_change) > self.significance_threshold * 100
        else:
            winner = model_a_id if difference < 0 else model_b_id
            significant = abs(percent_change) > self.significance_threshold * 100
        
        return MetricComparison(
            metric=metric_name,
            model_a_value=value_a,
            model_b_value=value_b,
            difference=difference,
            percent_change=percent_change,
            winner=winner,
            significant=significant,
        )
    
    def _determine_winner(
        self,
        metrics: List[MetricComparison],
        model_a: MLModel,
        model_b: MLModel,
    ) -> Tuple[Optional[str], str]:
        """Determine overall winner based on weighted metrics."""
        # Weight important metrics higher
        weights = {
            "f1": 3.0,
            "auc": 2.5,
            "precision": 2.0,
            "recall": 2.0,
            "accuracy": 1.0,
        }
        
        model_a_score = 0.0
        model_b_score = 0.0
        
        for m in metrics:
            if m.winner is None:
                continue
            
            weight = weights.get(m.metric.lower(), 1.0)
            
            if m.winner == str(model_a.id):
                model_a_score += weight * (1 if m.significant else 0.5)
            else:
                model_b_score += weight * (1 if m.significant else 0.5)
        
        # Determine winner
        if abs(model_a_score - model_b_score) < 0.5:
            overall_winner = None
            recommendation = (
                f"Models are comparable in performance. "
                f"Consider other factors like inference speed and complexity."
            )
        elif model_a_score > model_b_score:
            overall_winner = str(model_a.id)
            recommendation = (
                f"{model_a.name} shows better overall performance. "
                f"Recommend promoting to production."
            )
        else:
            overall_winner = str(model_b.id)
            recommendation = (
                f"{model_b.name} shows better overall performance. "
                f"Recommend promoting to production."
            )
        
        return overall_winner, recommendation
    
    async def compare_feature_importance(
        self,
        model_a_id: str,
        model_b_id: str,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Compare feature importance between models."""
        model_a = await self._get_model(model_a_id)
        model_b = await self._get_model(model_b_id)
        
        if not model_a or not model_b:
            raise ValueError("One or both models not found")
        
        importance_a = model_a.feature_importance or {}
        importance_b = model_b.feature_importance or {}
        
        all_features = set(importance_a.keys()) | set(importance_b.keys())
        
        comparison = []
        for feature in all_features:
            val_a = importance_a.get(feature, 0)
            val_b = importance_b.get(feature, 0)
            
            comparison.append({
                "feature": feature,
                "model_a": val_a,
                "model_b": val_b,
                "difference": val_a - val_b,
            })
        
        # Sort by max importance
        comparison.sort(key=lambda x: max(x["model_a"], x["model_b"]), reverse=True)
        
        return {
            "model_a": {"id": model_a_id, "name": model_a.name},
            "model_b": {"id": model_b_id, "name": model_b.name},
            "features": comparison[:top_n],
        }
