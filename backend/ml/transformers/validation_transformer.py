"""
Validation Transformer
Data quality validation before inference.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ValidationTransformer(BaseEstimator, TransformerMixin):
    """
    Validate data quality before model inference.
    
    Checks for:
    - Missing values
    - Outliers (based on training distribution)
    - Data drift (distribution changes)
    - Schema mismatches
    
    Parameters
    ----------
    strict_mode : bool, default=False
        If True, raise errors on validation failures.
        If False, log warnings and continue.
    outlier_threshold : float, default=5.0
        Number of standard deviations for outlier detection.
    """
    
    def __init__(self, strict_mode: bool = False, outlier_threshold: float = 5.0):
        self.strict_mode = strict_mode
        self.outlier_threshold = outlier_threshold
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ValidationTransformer':
        """
        Learn training data statistics for validation.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training data
        y : pd.Series, optional
            Target variable (not used)
        
        Returns
        -------
        self : ValidationTransformer
            Fitted transformer
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        
        # Store schema
        self.feature_names_in_ = list(X.columns)
        self.dtypes_in_ = X.dtypes.to_dict()
        
        # Store statistics for numerical columns
        self.training_stats_ = {}
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            self.training_stats_[col] = {
                'mean': float(X[col].mean()),
                'std': float(X[col].std()),
                'min': float(X[col].min()),
                'max': float(X[col].max()),
                'null_rate': float(X[col].isnull().mean())
            }
        
        logger.info(f"ValidationTransformer fitted on {len(X)} samples, {len(numerical_cols)} numerical features")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data and return unchanged (or raise error if strict).
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to validate
        
        Returns
        -------
        X : pd.DataFrame
            Unchanged data (validation is side-effect only)
        """
        issues = []
        
        # Check 1: Missing columns
        missing_cols = set(self.feature_names_in_) - set(X.columns)
        if missing_cols:
            msg = f"Missing columns: {missing_cols}"
            if self.strict_mode:
                raise ValueError(msg)
            issues.append(msg)
        
        # Check 2: Extra columns (warning only)
        extra_cols = set(X.columns) - set(self.feature_names_in_)
        if extra_cols:
            logger.warning(f"Extra columns (will be ignored): {extra_cols}")
        
        # Check 3: Missing values
        null_cols = X.columns[X.isnull().any()].tolist()
        if null_cols:
            null_rates = {col: X[col].isnull().mean() for col in null_cols}
            msg = f"Missing values detected: {null_rates}"
            
            # Check if null rate increased significantly
            for col in null_cols:
                if col in self.training_stats_:
                    training_null_rate = self.training_stats_[col]['null_rate']
                    current_null_rate = null_rates[col]
                    if current_null_rate > training_null_rate + 0.1:  # 10% increase
                        msg += f"\n  {col}: null rate increased from {training_null_rate:.2%} to {current_null_rate:.2%}"
            
            if self.strict_mode and null_cols:
                raise ValueError(msg)
            if null_cols:
                logger.warning(msg)
                issues.append(msg)
        
        # Check 4: Outliers
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in self.training_stats_:
                stats = self.training_stats_[col]
                mean, std = stats['mean'], stats['std']
                
                if std > 0:  # Avoid division by zero
                    outliers = (np.abs(X[col] - mean) > self.outlier_threshold * std).sum()
                    outlier_rate = outliers / len(X)
                    
                    if outlier_rate > 0.05:  # More than 5% outliers
                        msg = f"High outlier rate in {col}: {outlier_rate:.2%} ({outliers}/{len(X)} samples)"
                        logger.warning(msg)
                        issues.append(msg)
        
        # Check 5: Distribution drift (simple check)
        for col in numerical_cols:
            if col in self.training_stats_:
                stats = self.training_stats_[col]
                current_mean = X[col].mean()
                training_mean = stats['mean']
                
                if training_mean != 0:
                    drift = abs(current_mean - training_mean) / abs(training_mean)
                    if drift > 0.5:  # 50% drift
                        msg = f"Significant drift in {col}: mean changed from {training_mean:.2f} to {current_mean:.2f} ({drift:.1%})"
                        logger.warning(msg)
                        issues.append(msg)
        
        # Log summary
        if issues:
            logger.warning(f"Validation found {len(issues)} issues")
        else:
            logger.info("Validation passed: no issues detected")
        
        return X
    
    def get_validation_report(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed validation report without raising errors.
        
        Parameters
        ----------
        X : pd.DataFrame
            Data to validate
        
        Returns
        -------
        report : dict
            Validation report with issues and statistics
        """
        report = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Run validation (non-strict)
        original_strict = self.strict_mode
        self.strict_mode = False
        
        try:
            self.transform(X)
        except Exception as e:
            report['is_valid'] = False
            report['issues'].append(str(e))
        finally:
            self.strict_mode = original_strict
        
        # Add statistics
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in self.training_stats_:
                report['statistics'][col] = {
                    'current_mean': float(X[col].mean()),
                    'training_mean': self.training_stats_[col]['mean'],
                    'current_std': float(X[col].std()),
                    'training_std': self.training_stats_[col]['std']
                }
        
        return report
