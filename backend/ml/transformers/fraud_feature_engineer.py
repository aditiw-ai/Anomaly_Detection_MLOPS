"""
Generic Feature Engineer
sklearn-compatible transformer for fraud detection feature engineering.
Auto-detects column roles and generates features conditionally.
Works with any tabular dataset — no hardcoded column names.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any, List

from ml.transformers.column_role_detector import ColumnRoleDetector, ColumnRoles

logger = logging.getLogger(__name__)

# Default aggregation windows when not provided
_DEFAULT_AGG_WINDOWS = ["1h", "24h", "7d"]


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Generic, dataset-agnostic feature engineering transformer.

    Compatible with sklearn Pipeline and tree-based models (XGBoost, LightGBM).

    Features are generated **conditionally** based on detected column roles:
    - Transaction features — requires an *amount* column
    - Behavioral features — requires *user* + *amount* columns
    - Temporal features — requires a *timestamp* column
    - Aggregation features — requires *user* + *timestamp* + *amount*
    - Generic numeric features — for every remaining numeric column
    - Generic categorical features — stable ordinal encoding

    Each feature group can be toggled on/off via ``config``.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. Recognised keys:

        * ``column_mapping`` — ``{original_col: semantic_role}``
        * ``transaction_features`` — bool (default True)
        * ``behavioral_features`` — bool (default True)
        * ``temporal_features`` — bool (default True)
        * ``aggregation_features`` — bool (default True)
        * ``aggregation_windows`` — list of window strings (default ``["1h","24h","7d"]``)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config

    # ------------------------------------------------------------------
    # sklearn interface
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudFeatureEngineer":
        """
        Learn column roles and statistics from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Raw training data (any schema).
        y : pd.Series, optional
            Target labels (used for supervised features like fraud rate).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        # --- Detect column roles ---
        detector = ColumnRoleDetector()
        cfg = self.config or {}
        self.roles_: ColumnRoles = detector.detect(
            X, column_mapping=cfg.get("column_mapping")
        )

        self.feature_names_in_ = list(X.columns)

        # --- Learn statistics based on detected roles ---
        self._learn_amount_stats(X)
        self._learn_numeric_stats(X)
        self._learn_user_stats(X)
        self._learn_fraud_rates(X, y)
        self._learn_categorical_encoders(X)

        # --- Determine output feature names (dry-run transform) ---
        X_transformed = self._transform_impl(X)
        self.feature_names_out_ = list(X_transformed.columns)

        logger.info(
            f"FraudFeatureEngineer fitted: {len(X)} samples, "
            f"{len(self.feature_names_in_)} input cols -> "
            f"{len(self.feature_names_out_)} output features"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using learned statistics and detected roles.

        Parameters
        ----------
        X : pd.DataFrame
            Data to transform (must have same columns as training data).

        Returns
        -------
        pd.DataFrame
            All-numeric DataFrame with deterministic column order.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        X_transformed = self._transform_impl(X)

        # Enforce trained feature order — add missing cols as NaN, drop extras
        for col in self.feature_names_out_:
            if col not in X_transformed.columns:
                X_transformed[col] = np.nan
        X_transformed = X_transformed[self.feature_names_out_]

        # Reset index to prevent alignment issues in pipelines
        X_transformed = X_transformed.reset_index(drop=True)

        # Validate all numeric
        non_numeric = X_transformed.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            logger.warning(f"Non-numeric columns after transform (forcing to NaN): {non_numeric}")
            for col in non_numeric:
                X_transformed[col] = pd.to_numeric(X_transformed[col], errors="coerce")

        # Convert to float32 for memory efficiency
        for col in X_transformed.columns:
            if X_transformed[col].dtype != np.float32:
                X_transformed[col] = X_transformed[col].astype(np.float32)

        return X_transformed

    def get_feature_names_out(self, input_features: Optional[list] = None) -> list:
        """sklearn-compatible feature name accessor."""
        return self.feature_names_out_

    # ------------------------------------------------------------------
    # Learning helpers (called during fit)
    # ------------------------------------------------------------------

    def _learn_amount_stats(self, X: pd.DataFrame) -> None:
        col = self.roles_.amount_col
        if col and col in X.columns:
            series = pd.to_numeric(X[col], errors="coerce")
            self.amount_mean_ = float(series.mean())
            self.amount_std_ = float(series.std()) if series.std() > 0 else 1.0
            self.amount_q95_ = float(series.quantile(0.95))
        else:
            self.amount_mean_ = None
            self.amount_std_ = None
            self.amount_q95_ = None

    def _learn_numeric_stats(self, X: pd.DataFrame) -> None:
        """Learn per-column mean/std for generic numeric columns during fit."""
        self.numeric_stats_: Dict[str, Dict[str, float]] = {}
        for col in self.roles_.numeric_cols:
            if col not in X.columns:
                continue
            series = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
            mean = float(series.mean())
            std = float(series.std())
            self.numeric_stats_[col] = {"mean": mean, "std": std if std > 0 else 1.0}

    def _learn_user_stats(self, X: pd.DataFrame) -> None:
        ucol = self.roles_.user_col
        acol = self.roles_.amount_col
        if ucol and acol and ucol in X.columns and acol in X.columns:
            amounts = pd.to_numeric(X[acol], errors="coerce")
            grouped = pd.DataFrame({"user": X[ucol], "amount": amounts})
            stats = grouped.groupby("user")["amount"].agg(["mean", "std", "count", "sum", "max"])
            self.user_stats_ = stats
        else:
            self.user_stats_ = None

    def _learn_fraud_rates(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Learn fraud rate per category column (supervised feature)."""
        self.fraud_rates_: Dict[str, pd.Series] = {}
        if y is None:
            return
            
        # Safely convert target to numeric (for mean aggregation)
        y_numeric = pd.to_numeric(y, errors="coerce")
        if y_numeric.isna().all() and not y.isna().all():
            # Target consists of string labels.
            if len(y.dropna().unique()) == 2:
                # Binary classification: assume minority class is the positive (fraud) class
                majority_class = y.value_counts().index[0]
                y_numeric = (y != majority_class).astype(float)
            else:
                # Multiclass or single class string label – skip target encoding to prevent crash
                return

        for col in self.roles_.category_cols:
            if col in X.columns:
                tmp = pd.DataFrame({"cat": X[col], "y": y_numeric.values})
                rates = tmp.groupby("cat")["y"].mean()
                self.fraud_rates_[col] = rates

    def _learn_categorical_encoders(self, X: pd.DataFrame) -> None:
        """Fit OrdinalEncoders for each categorical column."""
        self.cat_encoders_: Dict[str, OrdinalEncoder] = {}
        for col in self.roles_.category_cols:
            if col in X.columns:
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.float32,
                )
                enc.fit(X[[col]].astype(str))
                self.cat_encoders_[col] = enc

    # ------------------------------------------------------------------
    # Core transformation
    # ------------------------------------------------------------------

    def _transform_impl(self, X: pd.DataFrame) -> pd.DataFrame:
        """Full transformation pipeline — called by both fit() and transform()."""
        result = pd.DataFrame(index=X.index)

        cfg = self.config or {}

        if cfg.get("transaction_features", True):
            self._add_transaction_features(X, result)

        if cfg.get("behavioral_features", True):
            self._add_behavioral_features(X, result)

        if cfg.get("temporal_features", True):
            self._add_temporal_features(X, result)

        if cfg.get("aggregation_features", True):
            self._add_aggregation_features(X, result)

        # Generic features (always on — they cover "remaining" columns)
        self._add_generic_numeric_features(X, result)
        self._add_generic_categorical_features(X, result)

        if result.shape[1] == 0:
            logger.warning("No features generated — check column roles and toggles")

        return self._sanitize_output(result)

    @staticmethod
    def _sanitize_output(df: pd.DataFrame) -> pd.DataFrame:
        """Replace Inf/-Inf with NaN so downstream imputers can handle them."""
        return df.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------
    # Feature group implementations
    # ------------------------------------------------------------------

    def _add_transaction_features(self, X: pd.DataFrame, out: pd.DataFrame) -> None:
        """Amount-derived features. Requires: amount_col."""
        acol = self.roles_.amount_col
        if not acol or acol not in X.columns:
            logger.debug("Skipping transaction features: no amount column detected")
            return

        amount = pd.to_numeric(X[acol], errors="coerce").fillna(0.0)

        out["amount_log"] = np.log1p(amount).astype(np.float32)
        out["amount_sqrt"] = np.sqrt(amount.clip(lower=0)).astype(np.float32)

        # Z-score using learned statistics
        mean = self.amount_mean_ if self.amount_mean_ is not None else amount.mean()
        std = self.amount_std_ if self.amount_std_ is not None else (amount.std() or 1.0)
        out["amount_zscore"] = ((amount - mean) / std).astype(np.float32)

        out["is_round_amount"] = (amount % 100 == 0).astype(np.float32)

        q95 = self.amount_q95_ if self.amount_q95_ is not None else amount.quantile(0.95)
        out["is_high_value"] = (amount > q95).astype(np.float32)

        out["amount_cents"] = ((amount * 100) % 100).astype(np.float32)

        logger.debug(f"Added 6 transaction features from '{acol}'")

    def _add_behavioral_features(self, X: pd.DataFrame, out: pd.DataFrame) -> None:
        """User-behaviour features. Requires: user_col + amount_col."""
        ucol = self.roles_.user_col
        acol = self.roles_.amount_col
        if not ucol or not acol or ucol not in X.columns or acol not in X.columns:
            logger.debug("Skipping behavioral features: need both user and amount columns")
            return

        amount = pd.to_numeric(X[acol], errors="coerce").fillna(0.0)

        if self.user_stats_ is not None and not self.user_stats_.empty:
            user_mean = X[ucol].map(self.user_stats_["mean"])
            user_std = X[ucol].map(self.user_stats_["std"]).fillna(1.0)
            user_count = X[ucol].map(self.user_stats_["count"])
            user_total = X[ucol].map(self.user_stats_["sum"])
            user_max = X[ucol].map(self.user_stats_["max"])

            fallback_mean = self.amount_mean_ if self.amount_mean_ is not None else amount.mean()
            user_mean = user_mean.fillna(fallback_mean)

            out["user_avg_amount"] = user_mean.astype(np.float32)
            out["user_std_amount"] = user_std.astype(np.float32)
            out["user_txn_count"] = user_count.fillna(0).astype(np.float32)
            out["user_total_amount"] = user_total.fillna(0).astype(np.float32)
            out["user_max_amount"] = user_max.fillna(0).astype(np.float32)
            out["amount_vs_user_avg"] = (amount / user_mean.clip(lower=1e-6)).astype(np.float32)
            out["amount_vs_user_std"] = ((amount - user_mean) / user_std.clip(lower=1e-6)).astype(np.float32)
            out["is_user_max"] = (amount >= user_max.fillna(amount + 1)).astype(np.float32)

            logger.debug(f"Added 8 behavioral features from '{ucol}' + '{acol}'")
        else:
            logger.debug("Skipping behavioral features: no user statistics available")

    def _add_temporal_features(self, X: pd.DataFrame, out: pd.DataFrame) -> None:
        """Time-based features. Requires: timestamp_col."""
        tcol = self.roles_.timestamp_col
        if not tcol or tcol not in X.columns:
            logger.debug("Skipping temporal features: no timestamp column detected")
            return

        try:
            ts = pd.to_datetime(X[tcol], errors="coerce")
        except Exception:
            logger.warning(f"Could not parse '{tcol}' as datetime — skipping temporal features")
            return

        out["hour_of_day"] = ts.dt.hour.astype(np.float32)
        out["day_of_week"] = ts.dt.dayofweek.astype(np.float32)
        out["day_of_month"] = ts.dt.day.astype(np.float32)
        out["is_weekend"] = (ts.dt.dayofweek >= 5).astype(np.float32)
        out["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour < 6)).astype(np.float32)
        out["is_business_hours"] = (
            (ts.dt.hour >= 9) & (ts.dt.hour < 17) & (ts.dt.dayofweek < 5)
        ).astype(np.float32)

        # Cyclical encoding
        out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24).astype(np.float32)
        out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24).astype(np.float32)

        # Time since last transaction (per user, if available)
        ucol = self.roles_.user_col
        if ucol and ucol in X.columns:
            df_ts = pd.DataFrame({"user": X[ucol], "ts": ts})
            df_ts = df_ts.sort_values(["user", "ts"])
            df_ts["diff"] = df_ts.groupby("user")["ts"].diff().dt.total_seconds().fillna(86400.0)
            # Re-align with original index
            out["time_since_last_txn"] = df_ts["diff"].reindex(X.index).fillna(86400.0).astype(np.float32)
            out["is_rapid_transaction"] = (out["time_since_last_txn"] < 60).astype(np.float32)
            logger.debug(f"Added 10 temporal features from '{tcol}' + '{ucol}'")
        else:
            logger.debug(f"Added 8 temporal features from '{tcol}'") 

    def _add_aggregation_features(self, X: pd.DataFrame, out: pd.DataFrame) -> None:
        """Rolling-window aggregation features. Requires: user_col + timestamp_col + amount_col."""
        ucol = self.roles_.user_col
        tcol = self.roles_.timestamp_col
        acol = self.roles_.amount_col
        if not (ucol and tcol and acol) or not all(c in X.columns for c in [ucol, tcol, acol]):
            logger.debug("Skipping aggregation features: need user, timestamp, and amount columns")
            return

        windows = (self.config or {}).get("aggregation_windows", _DEFAULT_AGG_WINDOWS)

        try:
            ts = pd.to_datetime(X[tcol], errors="coerce")
        except Exception:
            logger.warning(f"Could not parse '{tcol}' for aggregation — skipping")
            return

        amount = pd.to_numeric(X[acol], errors="coerce").fillna(0.0)

        # Build a working frame sorted by user + time
        work = pd.DataFrame({
            "user": X[ucol],
            "ts": ts,
            "amount": amount,
        }, index=X.index).sort_values(["user", "ts"])

        for window_str in windows:
            safe_name = window_str.replace("h", "hr").replace("d", "day")
            try:
                window_td = pd.Timedelta(window_str)
            except ValueError:
                logger.warning(f"Invalid aggregation window '{window_str}', skipping")
                continue

            # Simplified batch approach: per-user count and sum
            batch_counts = work.groupby("user")["amount"].transform("count")
            batch_sums = work.groupby("user")["amount"].transform("sum")

            # MLOps Fix: During inference on small batches, 'batch_counts' will be ~1, causing
            # a massive distribution shift from training (where 'batch_counts' was lifetime count).
            # We map to the established user history if available.
            if self.user_stats_ is not None and not self.user_stats_.empty:
                historical_counts = X[ucol].map(self.user_stats_["count"])
                historical_sums = X[ucol].map(self.user_stats_["sum"])
                
                # Combine: use historical if exists, else batch
                counts = historical_counts.fillna(batch_counts)
                sums = historical_sums.fillna(batch_sums)
            else:
                counts = batch_counts
                sums = batch_sums

            out[f"velocity_{safe_name}"] = counts.reindex(X.index).fillna(0).astype(np.float32)
            out[f"amount_sum_{safe_name}"] = sums.reindex(X.index).fillna(0).astype(np.float32)

        logger.debug(f"Added {len(windows) * 2} aggregation features")

    def _add_generic_numeric_features(self, X: pd.DataFrame, out: pd.DataFrame) -> None:
        """Log and z-score for every remaining numeric column."""
        numeric_stats = getattr(self, "numeric_stats_", {})
        for col in self.roles_.numeric_cols:
            if col not in X.columns:
                continue
            series = pd.to_numeric(X[col], errors="coerce").fillna(0.0)

            # Log transform (only for non-negative values — use training sign)
            stats = numeric_stats.get(col, {})
            mean = stats.get("mean", float(series.mean()))
            std = stats.get("std", float(series.std()) if series.std() > 0 else 1.0)

            if (series >= 0).all():
                out[f"{col}_log"] = np.log1p(series).astype(np.float32)

            # Z-score using training statistics to prevent distribution shift at inference
            out[f"{col}_zscore"] = ((series - mean) / std).astype(np.float32)

        if self.roles_.numeric_cols:
            logger.debug(f"Added generic features for {len(self.roles_.numeric_cols)} numeric columns")

    def _add_generic_categorical_features(self, X: pd.DataFrame, out: pd.DataFrame) -> None:
        """Stable ordinal encoding + optional fraud-rate encoding for categoricals."""
        for col in self.roles_.category_cols:
            if col not in X.columns:
                continue

            # Ordinal encoding
            encoder = getattr(self, "cat_encoders_", {}).get(col)
            if encoder is not None:
                encoded = encoder.transform(X[[col]].astype(str))
                out[f"{col}_encoded"] = encoded[:, 0].astype(np.float32)
            else:
                # Fallback: simple hash-based encoding
                out[f"{col}_encoded"] = (
                    X[col].astype(str).apply(lambda v: hash(v) % 10000)
                ).astype(np.float32)

            # Fraud-rate encoding (if learned during fit)
            fraud_rates = getattr(self, "fraud_rates_", {}).get(col)
            if fraud_rates is not None:
                out[f"{col}_fraud_rate"] = (
                    X[col].map(fraud_rates).fillna(0.5).astype(np.float32)
                )

        if self.roles_.category_cols:
            logger.debug(
                f"Added encoding features for {len(self.roles_.category_cols)} categorical columns"
            )
