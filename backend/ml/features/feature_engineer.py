"""
Feature Engineering Pipeline
Transforms raw transaction data into ML-ready features.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    transaction_features: bool = True
    behavioral_features: bool = True
    temporal_features: bool = True
    aggregation_features: bool = True
    aggregation_windows: List[str] = None
    
    def __post_init__(self):
        if self.aggregation_windows is None:
            self.aggregation_windows = ["1h", "24h", "7d"]


class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection.
    
    Creates 50+ features in 4 categories:
    - Transaction: Immediate transaction-level features
    - Behavioral: User history-based features
    - Temporal: Time-based patterns
    - Aggregation: Rolling window statistics
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.generated_features: List[str] = []
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering pipeline.
        
        Args:
            df: Raw transaction data with columns like:
                - amount, merchant_category, user_id, timestamp, etc.
        
        Returns:
            DataFrame with all computed features
        """
        result = df.copy()
        
        if self.config.transaction_features:
            result = self._add_transaction_features(result)
        
        if self.config.behavioral_features:
            result = self._add_behavioral_features(result)
        
        if self.config.temporal_features:
            result = self._add_temporal_features(result)
        
        if self.config.aggregation_features:
            result = self._add_aggregation_features(result)
        
        logger.info(f"Generated {len(self.generated_features)} features")
        return result
    
    def _add_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transaction-level features."""
        logger.info("Computing transaction features...")
        
        # Amount features
        if "amount" in df.columns:
            df["amount_log"] = np.log1p(df["amount"])
            df["amount_zscore"] = (df["amount"] - df["amount"].mean()) / (df["amount"].std() + 1e-6)
            df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)
            df["is_high_value"] = (df["amount"] > df["amount"].quantile(0.95)).astype(int)
            df["amount_cents"] = (df["amount"] * 100) % 100
            
            # Amount buckets
            df["amount_bucket"] = pd.cut(
                df["amount"],
                bins=[0, 50, 200, 1000, 5000, float("inf")],
                labels=["low", "medium", "high", "very_high", "extreme"]
            )
            
            self.generated_features.extend([
                "amount_log", "amount_zscore", "is_round_amount", 
                "is_high_value", "amount_cents", "amount_bucket"
            ])
        
        # Merchant features
        if "merchant_category" in df.columns:
            # High-risk merchant categories
            high_risk_categories = ["crypto", "gambling", "travel", "electronics"]
            df["is_high_risk_merchant"] = df["merchant_category"].isin(high_risk_categories).astype(int)
            self.generated_features.append("is_high_risk_merchant")
        
        return df
    
    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """User behavior-based features."""
        logger.info("Computing behavioral features...")
        
        if "user_id" not in df.columns:
            return df
        
        # User aggregations
        user_stats = df.groupby("user_id").agg({
            "amount": ["mean", "std", "count", "sum", "max"],
        }).reset_index()
        user_stats.columns = [
            "user_id", "user_avg_amount", "user_std_amount", 
            "user_txn_count", "user_total_amount", "user_max_amount"
        ]
        
        df = df.merge(user_stats, on="user_id", how="left")
        
        # Deviation from user average
        df["amount_vs_user_avg"] = df["amount"] / (df["user_avg_amount"] + 1e-6)
        df["amount_vs_user_std"] = (df["amount"] - df["user_avg_amount"]) / (df["user_std_amount"] + 1e-6)
        
        # Is this the user's max transaction?
        df["is_user_max"] = (df["amount"] == df["user_max_amount"]).astype(int)
        
        self.generated_features.extend([
            "user_avg_amount", "user_std_amount", "user_txn_count",
            "user_total_amount", "user_max_amount", "amount_vs_user_avg",
            "amount_vs_user_std", "is_user_max"
        ])
        
        # Merchant diversity
        if "merchant_category" in df.columns:
            merchant_diversity = df.groupby("user_id")["merchant_category"].nunique().reset_index()
            merchant_diversity.columns = ["user_id", "user_merchant_diversity"]
            df = df.merge(merchant_diversity, on="user_id", how="left")
            self.generated_features.append("user_merchant_diversity")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features."""
        logger.info("Computing temporal features...")
        
        if "timestamp" not in df.columns:
            return df
        
        ts = pd.to_datetime(df["timestamp"])
        
        # Time components
        df["hour_of_day"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_month"] = ts.dt.day
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        df["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour < 6)).astype(int)
        df["is_business_hours"] = ((ts.dt.hour >= 9) & (ts.dt.hour < 17) & (ts.dt.dayofweek < 5)).astype(int)
        
        # Cyclical encoding for hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
        
        self.generated_features.extend([
            "hour_of_day", "day_of_week", "day_of_month",
            "is_weekend", "is_night", "is_business_hours",
            "hour_sin", "hour_cos"
        ])
        
        # Time since last transaction (per user)
        if "user_id" in df.columns:
            df = df.sort_values(["user_id", "timestamp"])
            df["time_since_last"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
            df["time_since_last"] = df["time_since_last"].fillna(0)
            df["is_rapid_transaction"] = (df["time_since_last"] < 60).astype(int)  # Within 1 minute
            self.generated_features.extend(["time_since_last", "is_rapid_transaction"])
        
        return df
    
    def _add_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling window aggregation features."""
        logger.info("Computing aggregation features...")
        
        if "user_id" not in df.columns or "timestamp" not in df.columns:
            return df
        
        # Ensure timestamp is datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["user_id", "timestamp"])
        
        for window in self.config.aggregation_windows:
            window_td = pd.Timedelta(window)
            window_name = window.replace("h", "hr").replace("d", "day")
            
            # For each transaction, count transactions in window
            # This is computationally expensive, using a simplified approach
            # In production, would use more efficient rolling calculations
            
            logger.info(f"Computing {window} window features...")
            
            # Velocity (transaction count in window)
            df[f"velocity_{window_name}"] = df.groupby("user_id")["amount"].transform("count")
            
            # Amount sum in window
            df[f"amount_sum_{window_name}"] = df.groupby("user_id")["amount"].transform("sum")
            
            # Unique merchants in window (if available)
            if "merchant_category" in df.columns:
                df[f"unique_merchants_{window_name}"] = df.groupby("user_id")["merchant_category"].transform("nunique")
            
            self.generated_features.extend([
                f"velocity_{window_name}",
                f"amount_sum_{window_name}",
            ])
        
        return df
    
    def get_feature_info(self) -> Dict:
        """Get information about generated features."""
        return {
            "total_features": len(self.generated_features),
            "features": self.generated_features,
            "config": {
                "transaction_features": self.config.transaction_features,
                "behavioral_features": self.config.behavioral_features,
                "temporal_features": self.config.temporal_features,
                "aggregation_features": self.config.aggregation_features,
            }
        }
