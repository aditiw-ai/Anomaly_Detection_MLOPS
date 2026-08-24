"""
Data Drift Detection
Statistical methods for detecting distribution changes.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.stats import ks_2samp, chi2_contingency
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    psi_bins: int = 10
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.25
    ks_alpha: float = 0.05


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    status: str  # OK, WARNING, CRITICAL
    details: Dict


class DataDriftDetector:
    """
    Detects data drift between reference and current distributions.
    
    Methods:
    - PSI (Population Stability Index)
    - KS Test (Kolmogorov-Smirnov)
    - Chi-Square (for categorical features)
    """
    
    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
    
    def compute_drift(
        self, 
        reference: pd.DataFrame, 
        current: pd.DataFrame,
        features: List[str] = None
    ) -> Dict[str, DriftResult]:
        """
        Compute drift metrics for all features.
        
        Args:
            reference: Reference (training) data
            current: Current (production) data
            features: Features to check (default: all numeric)
        
        Returns:
            Dict mapping feature name to DriftResult
        """
        if features is None:
            features = [
                c for c in reference.columns 
                if c in current.columns and reference[c].dtype in ['int64', 'float64']
            ]
        
        results = {}
        for feature in features:
            if feature not in current.columns:
                continue
            
            ref_data = reference[feature].dropna()
            curr_data = current[feature].dropna()
            
            if len(ref_data) == 0 or len(curr_data) == 0:
                continue
            
            # Compute PSI
            psi = self._compute_psi(ref_data.values, curr_data.values)
            
            # Compute KS test
            ks_stat, ks_p = ks_2samp(ref_data, curr_data)
            
            # Determine status
            if psi > self.config.psi_critical_threshold:
                status = "CRITICAL"
            elif psi > self.config.psi_warning_threshold:
                status = "WARNING"
            else:
                status = "OK"
            
            results[feature] = DriftResult(
                feature=feature,
                psi=float(psi),
                ks_statistic=float(ks_stat),
                ks_p_value=float(ks_p),
                status=status,
                details={
                    "ref_mean": float(ref_data.mean()),
                    "ref_std": float(ref_data.std()),
                    "curr_mean": float(curr_data.mean()),
                    "curr_std": float(curr_data.std()),
                    "ref_count": len(ref_data),
                    "curr_count": len(curr_data),
                }
            )
        
        return results
    
    def _compute_psi(
        self, 
        reference: np.ndarray, 
        current: np.ndarray
    ) -> float:
        """
        Compute Population Stability Index.
        
        PSI < 0.1: No significant change
        PSI 0.1-0.25: Moderate change, monitoring needed
        PSI > 0.25: Significant change, action required
        """
        # Create bins from reference
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        bins = np.linspace(min_val, max_val, self.config.psi_bins + 1)
        
        # Compute histograms
        ref_hist, _ = np.histogram(reference, bins=bins)
        curr_hist, _ = np.histogram(current, bins=bins)
        
        # Convert to percentages
        ref_pct = ref_hist / len(reference) + 1e-6
        curr_pct = curr_hist / len(current) + 1e-6
        
        # Compute PSI
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        
        return float(psi)
    
    def compute_categorical_drift(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> Tuple[float, float, str]:
        """
        Compute drift for categorical features using Chi-Square test.
        
        Returns:
            Tuple of (chi2_statistic, p_value, status)
        """
        # Get all categories
        all_categories = list(set(reference.unique()) | set(current.unique()))
        
        # Build contingency table
        ref_counts = reference.value_counts()
        curr_counts = current.value_counts()
        
        observed = np.array([
            [ref_counts.get(c, 0) for c in all_categories],
            [curr_counts.get(c, 0) for c in all_categories]
        ])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(observed)
        
        # Determine status
        if p_value < 0.001:
            status = "CRITICAL"
        elif p_value < 0.05:
            status = "WARNING"
        else:
            status = "OK"
        
        return float(chi2), float(p_value), status
    
    def get_summary(self, results: Dict[str, DriftResult]) -> Dict:
        """Get summary of drift detection results."""
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
            "total_features": total,
            "ok": ok,
            "warning": warning,
            "critical": critical,
            "critical_features": [f for f, r in results.items() if r.status == "CRITICAL"],
            "warning_features": [f for f, r in results.items() if r.status == "WARNING"],
        }
