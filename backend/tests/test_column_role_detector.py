"""
Unit tests for ColumnRoleDetector.
Verifies auto-detection heuristics and user-override behaviour.
"""
import pytest
import pandas as pd
import numpy as np

from ml.transformers.column_role_detector import ColumnRoleDetector, ColumnRoles


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def detector():
    return ColumnRoleDetector()


@pytest.fixture
def standard_fraud_df():
    """Standard fraud-detection dataset with obvious column names."""
    return pd.DataFrame({
        "transaction_id": [1, 2, 3, 4, 5],
        "amount": [100.0, 250.0, 50.0, 1000.0, 75.0],
        "timestamp": pd.to_datetime([
            "2024-01-15 14:30:00",
            "2024-01-15 23:45:00",
            "2024-01-16 09:15:00",
            "2024-01-16 18:00:00",
            "2024-01-17 02:00:00",
        ]),
        "user_id": ["u1", "u2", "u1", "u3", "u2"],
        "merchant_category": ["electronics", "grocery", "electronics", "travel", "grocery"],
        "device_type": ["mobile", "web", "mobile", "app", "web"],
        "is_fraud": [0, 0, 1, 0, 1],
    })


@pytest.fixture
def nonstandard_df():
    """Dataset with unusual column names — tests heuristic flexibility."""
    return pd.DataFrame({
        "Record ID": range(100),
        "Transaction Amount": np.random.uniform(10, 500, 100),
        "Transaction Date": pd.date_range("2024-01-01", periods=100, freq="h"),
        "Customer ID": [f"cust_{i % 20}" for i in range(100)],
        "Product Category": np.random.choice(["food", "tech", "health"], 100),
        "Is Fraudulent": np.random.choice([0, 1], 100),
    })


@pytest.fixture
def minimal_numeric_df():
    """Dataset with only numeric columns and no recognisable names."""
    return pd.DataFrame({
        "col_a": np.random.randn(50),
        "col_b": np.random.randn(50),
        "col_c": np.random.randn(50),
    })


# ---------------------------------------------------------------------------
# Tests: Auto-detection
# ---------------------------------------------------------------------------

class TestAutoDetection:

    def test_detects_amount_column(self, detector, standard_fraud_df):
        roles = detector.detect(standard_fraud_df)
        assert roles.amount_col == "amount"

    def test_detects_timestamp_column(self, detector, standard_fraud_df):
        roles = detector.detect(standard_fraud_df)
        assert roles.timestamp_col == "timestamp"

    def test_detects_user_column(self, detector, standard_fraud_df):
        roles = detector.detect(standard_fraud_df)
        assert roles.user_col == "user_id"

    def test_detects_target_column(self, detector, standard_fraud_df):
        roles = detector.detect(standard_fraud_df)
        assert roles.target_col == "is_fraud"

    def test_detects_id_column(self, detector, standard_fraud_df):
        roles = detector.detect(standard_fraud_df)
        assert "transaction_id" in roles.id_cols

    def test_detects_category_columns(self, detector, standard_fraud_df):
        roles = detector.detect(standard_fraud_df)
        assert "merchant_category" in roles.category_cols
        assert "device_type" in roles.category_cols

    def test_nonstandard_amount(self, detector, nonstandard_df):
        roles = detector.detect(nonstandard_df)
        assert roles.amount_col == "Transaction Amount"

    def test_nonstandard_timestamp(self, detector, nonstandard_df):
        roles = detector.detect(nonstandard_df)
        assert roles.timestamp_col == "Transaction Date"

    def test_nonstandard_user(self, detector, nonstandard_df):
        roles = detector.detect(nonstandard_df)
        assert roles.user_col == "Customer ID"

    def test_nonstandard_target(self, detector, nonstandard_df):
        roles = detector.detect(nonstandard_df)
        assert roles.target_col == "Is Fraudulent"

    def test_minimal_numeric_no_crash(self, detector, minimal_numeric_df):
        """A dataset with no recognisable roles should not crash."""
        roles = detector.detect(minimal_numeric_df)
        assert roles.amount_col is None
        assert roles.timestamp_col is None
        assert roles.user_col is None
        assert roles.target_col is None
        # All columns should end up as numeric
        assert len(roles.numeric_cols) == 3

    def test_string_timestamp_detection(self, detector):
        """Timestamp stored as string should still be detected."""
        df = pd.DataFrame({
            "txn_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "val": [10, 20, 30],
        })
        roles = detector.detect(df)
        assert roles.timestamp_col == "txn_date"


# ---------------------------------------------------------------------------
# Tests: User-provided column_mapping overrides
# ---------------------------------------------------------------------------

class TestUserOverrides:

    def test_override_amount(self, detector):
        df = pd.DataFrame({
            "money": [100, 200, 300],
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        })
        roles = detector.detect(df, column_mapping={"money": "amount"})
        assert roles.amount_col == "money"

    def test_override_timestamp(self, detector):
        df = pd.DataFrame({
            "created": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "val": [10, 20, 30],
        })
        roles = detector.detect(df, column_mapping={"created": "timestamp"})
        assert roles.timestamp_col == "created"

    def test_override_user(self, detector):
        df = pd.DataFrame({
            "acct": ["a1", "a2", "a3"],
            "val": [10, 20, 30],
        })
        roles = detector.detect(df, column_mapping={"acct": "user_id"})
        assert roles.user_col == "acct"

    def test_override_target(self, detector, standard_fraud_df):
        roles = detector.detect(standard_fraud_df, column_mapping={"is_fraud": "target"})
        assert roles.target_col == "is_fraud"

    def test_missing_mapping_column_warns(self, detector, standard_fraud_df, caplog):
        """Mapping a column that doesn't exist should log a warning, not crash."""
        roles = detector.detect(
            standard_fraud_df,
            column_mapping={"nonexistent_col": "amount"},
        )
        assert "not found in DataFrame" in caplog.text
        # Auto-detection should still work for other roles
        assert roles.amount_col == "amount"

    def test_partial_override(self, detector, standard_fraud_df):
        """Override only amount; other roles should still auto-detect."""
        roles = detector.detect(
            standard_fraud_df,
            column_mapping={"amount": "amount"},
        )
        assert roles.amount_col == "amount"
        assert roles.timestamp_col == "timestamp"  # auto-detected
        assert roles.user_col == "user_id"  # auto-detected
        assert roles.target_col == "is_fraud"  # auto-detected


# ---------------------------------------------------------------------------
# Tests: ColumnRoles dataclass
# ---------------------------------------------------------------------------

class TestColumnRoles:

    def test_summary(self):
        roles = ColumnRoles(
            amount_col="amt",
            timestamp_col="ts",
            user_col="uid",
            target_col="label",
            category_cols=["cat1"],
            numeric_cols=["n1", "n2"],
            id_cols=["pk"],
        )
        s = roles.summary()
        assert s["amount_col"] == "amt"
        assert s["timestamp_col"] == "ts"
        assert s["user_col"] == "uid"
        assert s["target_col"] == "label"
        assert s["category_cols"] == ["cat1"]
        assert len(s["numeric_cols"]) == 2
        assert s["id_cols"] == ["pk"]

    def test_default_empty(self):
        roles = ColumnRoles()
        assert roles.amount_col is None
        assert roles.category_cols == []
        assert roles.numeric_cols == []
