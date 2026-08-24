"""
Feature Selection Pipeline
Selects the most informative features using multiple methods.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection."""
    max_features: int = 30
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    mi_weight: float = 0.5
    importance_weight: float = 0.5


class FeatureSelector:
    """
    Multi-stage feature selection pipeline.
    
    Stages:
    1. Variance Threshold - Remove near-constant features
    2. Correlation Filter - Remove highly correlated features
    3. Mutual Information - Rank by information gain
    4. Model Importance - Validate with XGBoost
    5. Combined Ranking - Select top N features
    """
    
    def __init__(self, config: FeatureSelectionConfig = None):
        self.config = config or FeatureSelectionConfig()
        self.selected_features: List[str] = []
        self.selection_report: Dict = {}
        self.scaler = StandardScaler()
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        exclude_columns: List[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply feature selection pipeline.
        
        Args:
            df: DataFrame with features
            target_column: Name of target column
            exclude_columns: Columns to exclude from selection
        
        Returns:
            Tuple of (selected features DataFrame, selection report)
        """
        exclude = set(exclude_columns or [])
        exclude.add(target_column)
        
        # Get feature columns (only numeric — supports int/float of any width)
        feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        # Handle empty features case immediately
        if not feature_cols:
            logger.warning("No numeric features found in dataset.")
            self.selected_features = []
            self.selection_report = {
                "stages": {"original": 0, "final_selected": 0},
                "removed": {},
                "scores": {},
                "config_used": {},
                "error": "No numeric features found for selection"
            }
            return df[[target_column]].copy(), self.selection_report

        # Handle missing values
        if not X.empty:
            X = X.fillna(X.median())
        
        logger.info(f"Starting feature selection: {len(feature_cols)} features")
        
        # CRITICAL CHANGE: Calculate importance scores for ALL features FIRST
        # This ensures every feature gets evaluated, even if it has low variance
        logger.info("Computing importance scores for all features...")
        mi_scores = self._compute_mutual_information(X, y)
        importance_scores = self._compute_model_importance(X, y)
        logger.info(f"Computed scores for {len(mi_scores)} features")
        
        # Now apply filters for the SELECTION process (not for scoring)
        # Stage 1: Variance filter
        X_var, var_removed = self._apply_variance_filter(X)
        logger.info(f"After variance filter: {X_var.shape[1]} features (removed {len(var_removed)})")
        
        # Stage 2: Correlation filter
        X_uncorr, corr_removed = self._apply_correlation_filter(X_var)
        logger.info(f"After correlation filter: {X_uncorr.shape[1]} features (removed {len(corr_removed)})")
        
        # Validation: Check if any features remain
        if len(X_uncorr.columns) == 0:
            logger.warning("All features removed by filters. Returning empty selection.")
            self.selected_features = []
            
            # Create a report explaining why features were dropped
            scores_dict = {}
            for f in feature_cols:
                reason = "Dropped: filtered out."
                if f in var_removed:
                    reason = "Dropped: near-constant (failed variance filter)."
                elif f in corr_removed:
                    reason = "Dropped: highly correlated with another feature (redundant)."
                
                scores_dict[f] = {
                    "mutual_information": float(mi_scores.get(f, 0)),
                    "importance": float(importance_scores.get(f, 0)),
                    "rank": 999,
                    "recommendation": "dropped",
                    "reason": reason,
                }

            self.selection_report = {
                "stages": {
                    "original": len(feature_cols),
                    "after_variance": X_var.shape[1],
                    "after_correlation": X_uncorr.shape[1],
                    "final_selected": 0,
                },
                "removed": {
                    "variance_filter": var_removed,
                    "correlation_filter": corr_removed,
                },
                "scores": scores_dict,
                "config_used": {
                    "variance_threshold": self.config.variance_threshold,
                    "correlation_threshold": self.config.correlation_threshold,
                    "max_features": self.config.max_features,
                    "mi_weight": self.config.mi_weight,
                    "importance_weight": self.config.importance_weight,
                },
                "error": "No features passed variance/correlation filters"
            }
            return df[[target_column]].copy(), self.selection_report
        
        # Stage 3: Combined ranking (only on filtered features for selection)
        # But we keep scores for ALL features in the report
        filtered_mi = {k: v for k, v in mi_scores.items() if k in X_uncorr.columns}
        filtered_importance = {k: v for k, v in importance_scores.items() if k in X_uncorr.columns}
        
        self.selected_features = self._combine_rankings(
            filtered_mi, 
            filtered_importance, 
            max_features=self.config.max_features
        )
        logger.info(f"Selected {len(self.selected_features)} features")
        
        # Full combined score and rank for all filtered features (for report transparency)
        filtered_features = list(filtered_mi.keys())
        
        if len(filtered_features) == 0:
            # Handle empty case
            combined_rank_map = {}
            combined_scores_arr = np.array([])
        else:
            mi_vals = np.array([filtered_mi[f] for f in filtered_features])
            imp_vals = np.array([filtered_importance[f] for f in filtered_features])
            
            # Normalization logic moved inside else block to ensure arrays are valid
            if mi_vals.max() > 0:
                mi_vals = mi_vals / mi_vals.max()
            if imp_vals.max() > 0:
                imp_vals = imp_vals / imp_vals.max()
            combined_scores_arr = (
                self.config.mi_weight * mi_vals + self.config.importance_weight * imp_vals
            )
        
        if len(combined_scores_arr) > 0:
            sorted_idx = np.argsort(combined_scores_arr)[::-1]
            # feature -> (combined_score, rank 1..N)
            combined_rank_map = {}
            for rank_one_based, idx in enumerate(sorted_idx, start=1):
                f = filtered_features[idx]
                combined_rank_map[f] = (float(combined_scores_arr[idx]), rank_one_based)
        else:
            combined_rank_map = {}
        
        max_features = self.config.max_features
        
        def _recommendation_and_reason(feat: str) -> Tuple[str, str]:
            if feat in var_removed:
                return "dropped", "Dropped: near-constant (failed variance filter)."
            if feat in corr_removed:
                return "dropped", "Dropped: highly correlated with another feature (redundant)."
            if feat in combined_rank_map:
                comb_score, rank = combined_rank_map[feat]
                if feat in self.selected_features:
                    return "selected", f"Selected: in top {max_features} by combined score (rank {rank})."
                return "dropped", f"Dropped: below top {max_features} by combined score (rank {rank})."
            return "dropped", "Dropped: not ranked (filtered out earlier)."
        
        # Build report with scores for ALL features, including recommendation and reason
        scores_dict = {}
        for f in feature_cols:
            rec, reason = _recommendation_and_reason(f)
            comb_score, rank = combined_rank_map.get(f, (None, 999))
            entry = {
                "mutual_information": float(mi_scores.get(f, 0)),
                "importance": float(importance_scores.get(f, 0)),
                "rank": rank if f in combined_rank_map else 999,
                "recommendation": rec,
                "reason": reason,
            }
            if comb_score is not None:
                entry["combined_score"] = comb_score
            scores_dict[f] = entry
        
        self.selection_report = {
            "stages": {
                "original": len(feature_cols),
                "after_variance": X_var.shape[1],
                "after_correlation": X_uncorr.shape[1],
                "final_selected": len(self.selected_features),
            },
            "removed": {
                "variance_filter": var_removed,
                "correlation_filter": corr_removed,
            },
            "scores": scores_dict,
            "config_used": {
                "variance_threshold": self.config.variance_threshold,
                "correlation_threshold": self.config.correlation_threshold,
                "max_features": self.config.max_features,
                "mi_weight": self.config.mi_weight,
                "importance_weight": self.config.importance_weight,
            },
        }
        
        # Return selected features
        result = df[self.selected_features + [target_column]].copy()
        return result, self.selection_report
    
    def _apply_variance_filter(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove features with low variance."""
        selector = VarianceThreshold(threshold=self.config.variance_threshold)
        
        try:
            # Fit and transform
            X_scaled = self.scaler.fit_transform(X)
            selector.fit(X_scaled)
            
            mask = selector.get_support()
            selected_cols = X.columns[mask].tolist()
            removed_cols = X.columns[~mask].tolist()
            
            return X[selected_cols], removed_cols
        except ValueError as e:
            if "No feature in X meets the variance threshold" in str(e):
                logger.warning(f"Variance filter removed ALL features: {e}")
                return X[[]], X.columns.tolist()
            raise e
        except Exception as e:
            logger.error(f"Variance filter failed: {e}")
            # If fail, keep all? Or drop all? Safe to drop if failed?
            # Better to return empty to be safe
            return X[[]], X.columns.tolist()
    
    def _apply_correlation_filter(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        
        # Get upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find columns with high correlation
        to_drop = [
            col for col in upper.columns 
            if any(upper[col] > self.config.correlation_threshold)
        ]
        
        return X.drop(columns=to_drop), to_drop
    
    def _compute_mutual_information(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """Compute mutual information scores."""
        scores = mutual_info_classif(X, y, random_state=42)
        return dict(zip(X.columns, scores))
    
    def _compute_model_importance(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Dict[str, float]:
        """Compute feature importance using XGBoost."""
        try:
            from xgboost import XGBClassifier
            from sklearn.preprocessing import LabelEncoder
            
            # Encode string targets for XGBoost
            y_encoded = y
            if y.dtype == 'object' or pd.api.types.is_string_dtype(y):
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                
            model = XGBClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            model.fit(X, y_encoded)
            
            return dict(zip(X.columns, model.feature_importances_))
        except Exception as e:
            logger.warning(f"XGBoost importance failed: {e}, using zeros")
            return {col: 0.0 for col in X.columns}
    
    def _combine_rankings(
        self, 
        mi_scores: Dict[str, float],
        importance_scores: Dict[str, float],
        max_features: int
    ) -> List[str]:
        """Combine rankings from multiple methods."""
        features = list(mi_scores.keys())
        
        # Normalize scores
        mi_values = np.array([mi_scores[f] for f in features])
        imp_values = np.array([importance_scores[f] for f in features])
        
        # Handle edge cases
        if mi_values.max() > 0:
            mi_values = mi_values / mi_values.max()
        if imp_values.max() > 0:
            imp_values = imp_values / imp_values.max()
        
        # Weighted combination
        combined = (
            self.config.mi_weight * mi_values + 
            self.config.importance_weight * imp_values
        )
        
        # Sort and select top N
        sorted_indices = np.argsort(combined)[::-1]
        selected = [features[i] for i in sorted_indices[:max_features]]
        
        return selected
    
    def get_report(self) -> Dict:
        """Get the selection report."""
        return self.selection_report
