"""
Bias Detection
Fairness metrics for model predictions.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BiasConfig:
    """Configuration for bias detection."""
    protected_attributes: List[str] = None
    demographic_parity_threshold: float = 0.1
    equalized_odds_threshold: float = 0.1
    disparate_impact_threshold: float = 0.8  # 80% rule
    
    def __post_init__(self):
        if self.protected_attributes is None:
            self.protected_attributes = ["gender", "age_group"]


@dataclass
class BiasResult:
    """Result of bias detection for a protected attribute."""
    attribute: str
    demographic_parity_diff: float
    equalized_odds_diff: float
    disparate_impact: float
    group_rates: Dict[str, float]
    status: str  # OK, WARNING, CRITICAL


class BiasDetector:
    """
    Detects bias in model predictions across protected attributes.
    
    Metrics:
    - Demographic Parity: Equal positive prediction rates across groups
    - Equalized Odds: Equal TPR and FPR across groups
    - Disparate Impact: Ratio of positive rates (80% rule)
    """
    
    def __init__(self, config: BiasConfig = None):
        self.config = config or BiasConfig()
    
    def compute_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_features: pd.DataFrame
    ) -> Dict[str, BiasResult]:
        """
        Compute bias metrics for all protected attributes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_features: DataFrame with protected attribute columns
        
        Returns:
            Dict mapping attribute name to BiasResult
        """
        results = {}
        
        for attr in self.config.protected_attributes:
            if attr not in protected_features.columns:
                logger.warning(f"Protected attribute '{attr}' not found in data")
                continue
            
            sensitive = protected_features[attr]
            
            result = self._compute_attribute_bias(y_true, y_pred, sensitive, attr)
            results[attr] = result
        
        return results
    
    def _compute_attribute_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: pd.Series,
        attr_name: str
    ) -> BiasResult:
        """Compute bias metrics for a single protected attribute."""
        groups = sensitive.unique()
        
        # Compute positive rate for each group
        group_rates = {}
        group_tpr = {}
        group_fpr = {}
        
        for group in groups:
            mask = sensitive == group
            group_pred = y_pred[mask]
            group_true = y_true[mask]
            
            # Positive rate (demographic parity)
            positive_rate = group_pred.mean() if len(group_pred) > 0 else 0
            group_rates[str(group)] = float(positive_rate)
            
            # TPR and FPR (for equalized odds)
            if (group_true == 1).sum() > 0:
                tpr = ((group_pred == 1) & (group_true == 1)).sum() / (group_true == 1).sum()
            else:
                tpr = 0
            
            if (group_true == 0).sum() > 0:
                fpr = ((group_pred == 1) & (group_true == 0)).sum() / (group_true == 0).sum()
            else:
                fpr = 0
            
            group_tpr[str(group)] = float(tpr)
            group_fpr[str(group)] = float(fpr)
        
        # Demographic parity difference
        rates = list(group_rates.values())
        dp_diff = max(rates) - min(rates) if rates else 0
        
        # Equalized odds difference (max of TPR and FPR differences)
        tprs = list(group_tpr.values())
        fprs = list(group_fpr.values())
        eo_diff = max(
            max(tprs) - min(tprs) if tprs else 0,
            max(fprs) - min(fprs) if fprs else 0
        )
        
        # Disparate impact (min rate / max rate)
        if max(rates) > 0:
            disparate_impact = min(rates) / max(rates)
        else:
            disparate_impact = 1.0
        
        # Determine status
        if (dp_diff > self.config.demographic_parity_threshold * 2 or 
            disparate_impact < self.config.disparate_impact_threshold - 0.1):
            status = "CRITICAL"
        elif (dp_diff > self.config.demographic_parity_threshold or
              disparate_impact < self.config.disparate_impact_threshold):
            status = "WARNING"
        else:
            status = "OK"
        
        return BiasResult(
            attribute=attr_name,
            demographic_parity_diff=float(dp_diff),
            equalized_odds_diff=float(eo_diff),
            disparate_impact=float(disparate_impact),
            group_rates=group_rates,
            status=status
        )
    
    def get_summary(self, results: Dict[str, BiasResult]) -> Dict:
        """Get summary of bias detection results."""
        total = len(results)
        ok = sum(1 for r in results.values() if r.status == "OK")
        warning = sum(1 for r in results.values() if r.status == "WARNING")
        critical = sum(1 for r in results.values() if r.status == "CRITICAL")
        
        # Overall status
        if critical > 0:
            overall_status = "CRITICAL"
        elif warning > 0:
            overall_status = "WARNING"
        else:
            overall_status = "OK"
        
        return {
            "overall_status": overall_status,
            "total_attributes": total,
            "ok": ok,
            "warning": warning,
            "critical": critical,
            "details": {
                attr: {
                    "status": r.status,
                    "demographic_parity_diff": r.demographic_parity_diff,
                    "disparate_impact": r.disparate_impact,
                    "group_rates": r.group_rates,
                }
                for attr, r in results.items()
            }
        }
