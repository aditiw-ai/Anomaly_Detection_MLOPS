"""
Unit tests for the generic FraudFeatureEngineer transformer.
Tests cover: auto-detection, conditional feature groups, toggles,
different dataset schemas, sklearn compatibility, and edge cases.
"""
import pytest
import pandas as pd
import numpy as np

from ml.transformers.fraud_feature_engineer import FraudFeatureEngineer


# ---------------------------------------------------------------------------
# Fixtures: different dataset schemas
# ---------------------------------------------------------------------------

@pytest.fixture
def full_fraud_df():
    """Full dataset with amount, timestamp, user, categories, and target."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "amount": np.random.uniform(10, 5000, n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "user_id": [f"user_{i % 10}" for i in range(n)],
        "merchant_category": np.random.choice(["food", "tech", "travel", "health"], n),
        "device_type": np.random.choice(["mobile", "web", "app"], n),
        "is_fraud": np.random.choice([0, 1], n, p=[0.9, 0.1]),
    })


@pytest.fixture
def amount_only_df():
    """Dataset with only an amount column — no timestamp, no user."""
    return pd.DataFrame({
        "amount": [100.0, 250.0, 50.0, 1000.0, 75.0],
    })


@pytest.fixture
def nonstandard_names_df():
    """Dataset with non-standard column names."""
    return pd.DataFrame({
        "Record ID": range(50),
        "Transaction Amount": np.random.uniform(10, 500, 50),
        "Transaction Date": pd.date_range("2024-01-01", periods=50, freq="h"),
        "Customer ID": [f"cust_{i % 8}" for i in range(50)],
        "Product Category": np.random.choice(["electronics", "grocery"], 50),
        "Is Fraudulent": np.random.choice([0, 1], 50),
    })


@pytest.fixture
def numeric_only_df():
    """Dataset with only generic numeric columns."""
    np.random.seed(0)
    return pd.DataFrame({
        "feature_a": np.random.randn(30),
        "feature_b": np.abs(np.random.randn(30)),
        "feature_c": np.random.uniform(0, 100, 30),
    })


# ---------------------------------------------------------------------------
# Tests: Core sklearn interface
# ---------------------------------------------------------------------------

class TestSklearnInterface:

    def test_fit_returns_self(self, full_fraud_df):
        t = FraudFeatureEngineer()
        result = t.fit(full_fraud_df)
        assert result is t

    def test_fit_sets_attributes(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        assert hasattr(t, "feature_names_in_")
        assert hasattr(t, "feature_names_out_")
        assert hasattr(t, "roles_")
        assert len(t.feature_names_out_) > 0

    def test_transform_returns_dataframe(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(full_fraud_df)

    def test_output_all_numeric(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        non_numeric = result.select_dtypes(exclude=[np.number]).columns.tolist()
        assert non_numeric == [], f"Non-numeric columns found: {non_numeric}"

    def test_output_all_float32(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        for col in result.columns:
            assert result[col].dtype == np.float32, f"{col} has dtype {result[col].dtype}"

    def test_get_feature_names_out(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        names = t.get_feature_names_out()
        assert names == t.feature_names_out_

    def test_feature_order_consistency(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        r1 = t.transform(full_fraud_df)
        r2 = t.transform(full_fraud_df)
        assert list(r1.columns) == list(r2.columns)

    def test_rejects_non_dataframe(self):
        t = FraudFeatureEngineer()
        with pytest.raises(TypeError, match="must be a pandas DataFrame"):
            t.fit(np.array([[1, 2], [3, 4]]))


# ---------------------------------------------------------------------------
# Tests: Transaction features
# ---------------------------------------------------------------------------

class TestTransactionFeatures:

    def test_amount_features_present(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        expected = {"amount_log", "amount_sqrt", "amount_zscore", "is_round_amount", "is_high_value", "amount_cents"}
        assert expected.issubset(set(result.columns))

    def test_amount_log_values(self, amount_only_df):
        t = FraudFeatureEngineer()
        t.fit(amount_only_df)
        result = t.transform(amount_only_df)
        expected_log = np.log1p(amount_only_df["amount"]).astype(np.float32)
        np.testing.assert_array_almost_equal(result["amount_log"].values, expected_log.values, decimal=4)

    def test_skipped_when_no_amount(self, numeric_only_df):
        """No amount column detected -> transaction features skipped, no crash."""
        t = FraudFeatureEngineer()
        t.fit(numeric_only_df)
        result = t.transform(numeric_only_df)
        assert "amount_log" not in result.columns

    def test_toggle_off(self, full_fraud_df):
        t = FraudFeatureEngineer(config={"transaction_features": False})
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert "amount_log" not in result.columns


# ---------------------------------------------------------------------------
# Tests: Behavioral features
# ---------------------------------------------------------------------------

class TestBehavioralFeatures:

    def test_behavioral_features_present(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        expected = {"user_avg_amount", "user_std_amount", "user_txn_count", "amount_vs_user_avg"}
        assert expected.issubset(set(result.columns))

    def test_skipped_without_user(self, amount_only_df):
        t = FraudFeatureEngineer()
        t.fit(amount_only_df)
        result = t.transform(amount_only_df)
        assert "user_avg_amount" not in result.columns

    def test_toggle_off(self, full_fraud_df):
        t = FraudFeatureEngineer(config={"behavioral_features": False})
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert "user_avg_amount" not in result.columns


# ---------------------------------------------------------------------------
# Tests: Temporal features
# ---------------------------------------------------------------------------

class TestTemporalFeatures:

    def test_temporal_features_present(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        expected = {"hour_of_day", "day_of_week", "is_weekend", "is_night", "hour_sin", "hour_cos"}
        assert expected.issubset(set(result.columns))

    def test_time_since_last_with_user(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert "time_since_last_txn" in result.columns

    def test_skipped_without_timestamp(self, amount_only_df):
        t = FraudFeatureEngineer()
        t.fit(amount_only_df)
        result = t.transform(amount_only_df)
        assert "hour_of_day" not in result.columns

    def test_toggle_off(self, full_fraud_df):
        t = FraudFeatureEngineer(config={"temporal_features": False})
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert "hour_of_day" not in result.columns


# ---------------------------------------------------------------------------
# Tests: Aggregation features
# ---------------------------------------------------------------------------

class TestAggregationFeatures:

    def test_aggregation_features_present(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert "velocity_1hr" in result.columns
        assert "amount_sum_24hr" in result.columns

    def test_custom_windows(self, full_fraud_df):
        t = FraudFeatureEngineer(config={"aggregation_windows": ["2h", "48h"]})
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert "velocity_2hr" in result.columns
        assert "amount_sum_48hr" in result.columns

    def test_toggle_off(self, full_fraud_df):
        t = FraudFeatureEngineer(config={"aggregation_features": False})
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert "velocity_1hr" not in result.columns


# ---------------------------------------------------------------------------
# Tests: Generic numeric & categorical features
# ---------------------------------------------------------------------------

class TestGenericFeatures:

    def test_numeric_log_and_zscore(self, numeric_only_df):
        t = FraudFeatureEngineer()
        t.fit(numeric_only_df)
        result = t.transform(numeric_only_df)
        # feature_b is non-negative -> should have log
        assert "feature_b_log" in result.columns
        assert "feature_b_zscore" in result.columns

    def test_categorical_encoding(self, full_fraud_df):
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert "merchant_category_encoded" in result.columns
        assert "device_type_encoded" in result.columns

    def test_categorical_fraud_rate(self, full_fraud_df):
        """When target is provided, fraud-rate encoding should be created."""
        y = full_fraud_df["is_fraud"]
        t = FraudFeatureEngineer()
        t.fit(full_fraud_df, y=y)
        result = t.transform(full_fraud_df)
        assert "merchant_category_fraud_rate" in result.columns


# ---------------------------------------------------------------------------
# Tests: Non-standard column names (auto-detection)
# ---------------------------------------------------------------------------

class TestNonStandardNames:

    def test_detects_and_engineers_features(self, nonstandard_names_df):
        t = FraudFeatureEngineer()
        t.fit(nonstandard_names_df)
        result = t.transform(nonstandard_names_df)
        # Should still generate amount features
        assert "amount_log" in result.columns
        # Should generate temporal features
        assert "hour_of_day" in result.columns
        # Should generate behavioral features (Customer ID detected)
        assert "user_avg_amount" in result.columns

    def test_feature_count_reasonable(self, nonstandard_names_df):
        t = FraudFeatureEngineer()
        t.fit(nonstandard_names_df)
        result = t.transform(nonstandard_names_df)
        assert result.shape[1] >= 10, f"Expected at least 10 features, got {result.shape[1]}"


# ---------------------------------------------------------------------------
# Tests: User-provided column_mapping via config
# ---------------------------------------------------------------------------

class TestColumnMappingConfig:

    def test_explicit_mapping(self):
        df = pd.DataFrame({
            "money": [100, 200, 300, 400, 500],
            "when": pd.date_range("2024-01-01", periods=5, freq="h"),
            "who": ["a", "b", "a", "c", "b"],
        })
        config = {
            "column_mapping": {
                "money": "amount",
                "when": "timestamp",
                "who": "user_id",
            }
        }
        t = FraudFeatureEngineer(config=config)
        t.fit(df)
        result = t.transform(df)
        assert "amount_log" in result.columns
        assert "hour_of_day" in result.columns
        assert "user_avg_amount" in result.columns


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_row(self, full_fraud_df):
        single = full_fraud_df.head(1)
        t = FraudFeatureEngineer()
        t.fit(single)
        result = t.transform(single)
        assert len(result) == 1
        assert result.shape[1] > 0

    def test_all_toggles_off(self, full_fraud_df):
        config = {
            "transaction_features": False,
            "behavioral_features": False,
            "temporal_features": False,
            "aggregation_features": False,
        }
        t = FraudFeatureEngineer(config=config)
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        # Should still have generic numeric/categorical features
        assert result.shape[1] > 0

    def test_empty_config(self, full_fraud_df):
        t = FraudFeatureEngineer(config={})
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert result.shape[1] > 0

    def test_none_config(self, full_fraud_df):
        t = FraudFeatureEngineer(config=None)
        t.fit(full_fraud_df)
        result = t.transform(full_fraud_df)
        assert result.shape[1] > 0

    def test_nan_in_amount(self):
        df = pd.DataFrame({
            "amount": [100.0, np.nan, 50.0, np.nan, 75.0],
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="h"),
        })
        t = FraudFeatureEngineer()
        t.fit(df)
        result = t.transform(df)
        # Should not crash; NaN handled gracefully
        assert len(result) == 5
