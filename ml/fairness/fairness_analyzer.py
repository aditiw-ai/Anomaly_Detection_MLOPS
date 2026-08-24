"""
Fairlearn Bias Remediation
Advanced fairness metrics and mitigation strategies.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MitigationStrategy(str, Enum):
    """Bias mitigation strategies."""
    REWEIGH = "REWEIGH"           # Pre-processing: adjust sample weights
    THRESHOLD = "THRESHOLD"       # Post-processing: adjust thresholds per group
    EXPONENTIATED = "EXPONENTIATED"  # In-processing: exponentiated gradient
    GRID_SEARCH = "GRID_SEARCH"   # In-processing: grid search reduction


@dataclass
class FairnessMetrics:
    """Comprehensive fairness metrics."""
    demographic_parity_difference: float
    demographic_parity_ratio: float
    equalized_odds_difference: float
    equalized_odds_ratio: float
    accuracy_difference: float
    selection_rate: Dict[str, float]
    false_positive_rate: Dict[str, float]
    false_negative_rate: Dict[str, float]
    true_positive_rate: Dict[str, float]


@dataclass
class MitigationResult:
    """Result of bias mitigation."""
    strategy: MitigationStrategy
    original_metrics: FairnessMetrics
    mitigated_metrics: FairnessMetrics
    improvement: Dict[str, float]
    model_weights: Optional[np.ndarray] = None
    thresholds: Optional[Dict[str, float]] = None


class FairnessAnalyzer:
    """
    Fairness analysis and mitigation using Fairlearn.
    
    Supports multiple fairness constraints:
    - Demographic Parity (equal selection rates)
    - Equalized Odds (equal TPR and FPR)
    - True Positive Rate Parity
    """
    
    def __init__(self, protected_features: List[str] = None):
        """
        Initialize analyzer.
        
        Args:
            protected_features: Names of protected features
        """
        self.protected_features = protected_features or ["gender", "age_group"]
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> FairnessMetrics:
        """
        Compute comprehensive fairness metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            sensitive_features: Protected attribute values
        
        Returns:
            FairnessMetrics with all computed metrics
        """
        try:
            from fairlearn.metrics import (
                demographic_parity_difference,
                demographic_parity_ratio,
                equalized_odds_difference,
                equalized_odds_ratio,
                MetricFrame,
            )
            from sklearn.metrics import accuracy_score
            
            # Compute group-level metrics
            metric_frame = MetricFrame(
                metrics={
                    "selection_rate": lambda y_true, y_pred: np.mean(y_pred),
                    "accuracy": accuracy_score,
                    "fpr": lambda y_true, y_pred: np.sum((y_pred == 1) & (y_true == 0)) / max(np.sum(y_true == 0), 1),
                    "fnr": lambda y_true, y_pred: np.sum((y_pred == 0) & (y_true == 1)) / max(np.sum(y_true == 1), 1),
                    "tpr": lambda y_true, y_pred: np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_true == 1), 1),
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
            )
            
            # Extract per-group metrics
            by_group = metric_frame.by_group
            
            selection_rate = by_group["selection_rate"].to_dict()
            fpr = by_group["fpr"].to_dict()
            fnr = by_group["fnr"].to_dict()
            tpr = by_group["tpr"].to_dict()
            
            # Compute disparity metrics
            dp_diff = demographic_parity_difference(
                y_true, y_pred, sensitive_features=sensitive_features
            )
            dp_ratio = demographic_parity_ratio(
                y_true, y_pred, sensitive_features=sensitive_features
            )
            eo_diff = equalized_odds_difference(
                y_true, y_pred, sensitive_features=sensitive_features
            )
            eo_ratio = equalized_odds_ratio(
                y_true, y_pred, sensitive_features=sensitive_features
            )
            
            # Compute accuracy difference
            accuracies = by_group["accuracy"].values
            acc_diff = max(accuracies) - min(accuracies) if len(accuracies) > 0 else 0
            
            return FairnessMetrics(
                demographic_parity_difference=float(dp_diff),
                demographic_parity_ratio=float(dp_ratio),
                equalized_odds_difference=float(eo_diff),
                equalized_odds_ratio=float(eo_ratio),
                accuracy_difference=float(acc_diff),
                selection_rate=selection_rate,
                false_positive_rate=fpr,
                false_negative_rate=fnr,
                true_positive_rate=tpr,
            )
            
        except ImportError:
            logger.warning("Fairlearn not installed, using fallback metrics")
            return self._compute_fallback_metrics(y_true, y_pred, sensitive_features)
    
    def _compute_fallback_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> FairnessMetrics:
        """Compute basic fairness metrics without Fairlearn."""
        groups = np.unique(sensitive_features)
        
        selection_rate = {}
        fpr = {}
        fnr = {}
        tpr = {}
        
        for group in groups:
            mask = sensitive_features == group
            group_pred = y_pred[mask]
            group_true = y_true[mask]
            
            selection_rate[str(group)] = float(np.mean(group_pred))
            
            # FPR, FNR, TPR
            pos = group_true == 1
            neg = group_true == 0
            
            if np.sum(neg) > 0:
                fpr[str(group)] = float(np.sum((group_pred == 1) & neg) / np.sum(neg))
            else:
                fpr[str(group)] = 0.0
            
            if np.sum(pos) > 0:
                fnr[str(group)] = float(np.sum((group_pred == 0) & pos) / np.sum(pos))
                tpr[str(group)] = float(np.sum((group_pred == 1) & pos) / np.sum(pos))
            else:
                fnr[str(group)] = 0.0
                tpr[str(group)] = 0.0
        
        # Compute differences
        rates = list(selection_rate.values())
        dp_diff = max(rates) - min(rates) if rates else 0
        dp_ratio = min(rates) / max(rates) if rates and max(rates) > 0 else 1
        
        return FairnessMetrics(
            demographic_parity_difference=dp_diff,
            demographic_parity_ratio=dp_ratio,
            equalized_odds_difference=dp_diff,  # Simplified
            equalized_odds_ratio=dp_ratio,
            accuracy_difference=0.0,
            selection_rate=selection_rate,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            true_positive_rate=tpr,
        )
    
    def mitigate_bias(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        strategy: MitigationStrategy = MitigationStrategy.THRESHOLD,
    ) -> MitigationResult:
        """
        Apply bias mitigation strategy.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Labels
            sensitive_features: Protected attribute values
            strategy: Mitigation strategy to apply
        
        Returns:
            MitigationResult with before/after metrics
        """
        # Get original predictions and metrics
        y_pred_original = model.predict(X)
        original_metrics = self.compute_metrics(y, y_pred_original, sensitive_features)
        
        if strategy == MitigationStrategy.THRESHOLD:
            result = self._apply_threshold_optimizer(
                model, X, y, sensitive_features, original_metrics
            )
        elif strategy == MitigationStrategy.EXPONENTIATED:
            result = self._apply_exponentiated_gradient(
                model, X, y, sensitive_features, original_metrics
            )
        elif strategy == MitigationStrategy.GRID_SEARCH:
            result = self._apply_grid_search(
                model, X, y, sensitive_features, original_metrics
            )
        else:
            # Default to threshold optimizer
            result = self._apply_threshold_optimizer(
                model, X, y, sensitive_features, original_metrics
            )
        
        return result
    
    def _apply_threshold_optimizer(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        original_metrics: FairnessMetrics,
    ) -> MitigationResult:
        """Apply threshold optimization for equalized odds."""
        try:
            from fairlearn.postprocessing import ThresholdOptimizer
            
            # Get probability predictions
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)[:, 1]
            else:
                y_prob = model.predict(X)
            
            # Create and fit threshold optimizer
            postprocessor = ThresholdOptimizer(
                estimator=model,
                constraints="equalized_odds",
                prefit=True,
            )
            postprocessor.fit(X, y, sensitive_features=sensitive_features)
            
            # Get mitigated predictions
            y_pred_mitigated = postprocessor.predict(X, sensitive_features=sensitive_features)
            
            # Compute mitigated metrics
            mitigated_metrics = self.compute_metrics(y, y_pred_mitigated, sensitive_features)
            
            # Compute improvement
            improvement = {
                "dp_diff": original_metrics.demographic_parity_difference - mitigated_metrics.demographic_parity_difference,
                "eo_diff": original_metrics.equalized_odds_difference - mitigated_metrics.equalized_odds_difference,
            }
            
            return MitigationResult(
                strategy=MitigationStrategy.THRESHOLD,
                original_metrics=original_metrics,
                mitigated_metrics=mitigated_metrics,
                improvement=improvement,
            )
            
        except ImportError:
            logger.warning("Fairlearn postprocessing not available")
            return MitigationResult(
                strategy=MitigationStrategy.THRESHOLD,
                original_metrics=original_metrics,
                mitigated_metrics=original_metrics,
                improvement={},
            )
    
    def _apply_exponentiated_gradient(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        original_metrics: FairnessMetrics,
    ) -> MitigationResult:
        """Apply exponentiated gradient for in-processing mitigation."""
        try:
            from fairlearn.reductions import ExponentiatedGradient, DemographicParity
            from sklearn.base import clone
            
            # Create mitigated estimator
            mitigator = ExponentiatedGradient(
                estimator=clone(model),
                constraints=DemographicParity(),
            )
            mitigator.fit(X, y, sensitive_features=sensitive_features)
            
            # Get mitigated predictions
            y_pred_mitigated = mitigator.predict(X)
            
            # Compute mitigated metrics
            mitigated_metrics = self.compute_metrics(y, y_pred_mitigated, sensitive_features)
            
            improvement = {
                "dp_diff": original_metrics.demographic_parity_difference - mitigated_metrics.demographic_parity_difference,
            }
            
            return MitigationResult(
                strategy=MitigationStrategy.EXPONENTIATED,
                original_metrics=original_metrics,
                mitigated_metrics=mitigated_metrics,
                improvement=improvement,
            )
            
        except ImportError:
            logger.warning("Fairlearn reductions not available")
            return MitigationResult(
                strategy=MitigationStrategy.EXPONENTIATED,
                original_metrics=original_metrics,
                mitigated_metrics=original_metrics,
                improvement={},
            )
    
    def _apply_grid_search(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        original_metrics: FairnessMetrics,
    ) -> MitigationResult:
        """Apply grid search reduction for bias mitigation."""
        try:
            from fairlearn.reductions import GridSearch, EqualizedOdds
            from sklearn.base import clone
            
            # Create mitigated estimator
            mitigator = GridSearch(
                estimator=clone(model),
                constraints=EqualizedOdds(),
                grid_size=10,
            )
            mitigator.fit(X, y, sensitive_features=sensitive_features)
            
            # Get mitigated predictions
            y_pred_mitigated = mitigator.predict(X)
            
            # Compute mitigated metrics
            mitigated_metrics = self.compute_metrics(y, y_pred_mitigated, sensitive_features)
            
            improvement = {
                "eo_diff": original_metrics.equalized_odds_difference - mitigated_metrics.equalized_odds_difference,
            }
            
            return MitigationResult(
                strategy=MitigationStrategy.GRID_SEARCH,
                original_metrics=original_metrics,
                mitigated_metrics=mitigated_metrics,
                improvement=improvement,
            )
            
        except ImportError:
            logger.warning("Fairlearn grid search not available")
            return MitigationResult(
                strategy=MitigationStrategy.GRID_SEARCH,
                original_metrics=original_metrics,
                mitigated_metrics=original_metrics,
                improvement={},
            )
