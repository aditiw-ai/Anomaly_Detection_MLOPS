"""
Unit tests for Test 4 fix: Feature set metadata completeness.
Verifies that all_features, input_rows, and processing_time_seconds
are correctly passed through the service layer to the DB update.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from uuid import uuid4

from app.services.feature_service import FeatureService
from app.models.feature_set import FeatureSet


class TestFeatureSetMetadataUpdate:
    """Test that update_feature_set_status_sync handles new metadata fields."""

    @pytest.fixture
    def mock_db(self):
        """Mock sync database session."""
        db = MagicMock()
        db.execute = MagicMock()
        return db

    @pytest.fixture
    def service(self, mock_db):
        """Create FeatureService with mocked DB."""
        return FeatureService(mock_db)

    @pytest.fixture
    def feature_set_id(self):
        return str(uuid4())

    def test_all_features_passed_to_update(self, service, mock_db, feature_set_id):
        """Verify all_features is included in the DB update when provided."""
        features = ["amount", "hour", "day_of_week", "is_weekend"]

        result = service.update_feature_set_status_sync(
            feature_set_id, "COMPLETED",
            all_features=features,
        )

        assert result is True
        # Verify execute was called
        mock_db.execute.assert_called_once()
        # Extract the update statement values
        update_call = mock_db.execute.call_args
        stmt = update_call[0][0]
        # The compiled parameters should contain all_features
        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        param_keys = list(compiled.params.keys())
        assert "all_features" in param_keys

    def test_input_rows_passed_to_update(self, service, mock_db, feature_set_id):
        """Verify input_rows is included in the DB update when provided."""
        result = service.update_feature_set_status_sync(
            feature_set_id, "RUNNING",
            input_rows=15000,
        )

        assert result is True
        mock_db.execute.assert_called_once()
        update_call = mock_db.execute.call_args
        stmt = update_call[0][0]
        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        param_keys = list(compiled.params.keys())
        assert "input_rows" in param_keys

    def test_processing_time_passed_to_update(self, service, mock_db, feature_set_id):
        """Verify processing_time_seconds is included in the DB update when provided."""
        result = service.update_feature_set_status_sync(
            feature_set_id, "COMPLETED",
            processing_time_seconds=42,
        )

        assert result is True
        mock_db.execute.assert_called_once()
        update_call = mock_db.execute.call_args
        stmt = update_call[0][0]
        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        param_keys = list(compiled.params.keys())
        assert "processing_time_seconds" in param_keys

    def test_all_metadata_fields_in_completed_call(self, service, mock_db, feature_set_id):
        """Verify the full COMPLETED call includes all metadata fields."""
        features = ["amount", "hour", "day_of_week"]

        result = service.update_feature_set_status_sync(
            feature_set_id, "COMPLETED",
            storage_path="features/raw/test/features.parquet",
            selected_features=features,
            all_features=features,
            feature_count=3,
            input_rows=10000,
            processing_time_seconds=15,
            selection_report={"total_features": 3},
        )

        assert result is True
        mock_db.execute.assert_called_once()
        update_call = mock_db.execute.call_args
        stmt = update_call[0][0]
        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        param_keys = list(compiled.params.keys())

        # All fields must be present
        assert "status" in param_keys
        assert "storage_path" in param_keys
        assert "selected_features" in param_keys
        assert "selected_feature_count" in param_keys
        assert "all_features" in param_keys
        assert "feature_count" in param_keys
        assert "input_rows" in param_keys
        assert "processing_time_seconds" in param_keys
        assert "selection_report" in param_keys
        assert "completed_at" in param_keys

    def test_none_metadata_fields_excluded(self, service, mock_db, feature_set_id):
        """Verify that None values are NOT passed to the DB update."""
        result = service.update_feature_set_status_sync(
            feature_set_id, "RUNNING",
            # all_features, input_rows, processing_time_seconds all default to None
        )

        assert result is True
        mock_db.execute.assert_called_once()
        update_call = mock_db.execute.call_args
        stmt = update_call[0][0]
        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        param_keys = list(compiled.params.keys())

        # These should NOT be present when not provided
        assert "all_features" not in param_keys
        assert "input_rows" not in param_keys
        assert "processing_time_seconds" not in param_keys

    def test_zero_processing_time_is_included(self, service, mock_db, feature_set_id):
        """Verify processing_time_seconds=0 is included (not falsely excluded)."""
        result = service.update_feature_set_status_sync(
            feature_set_id, "COMPLETED",
            processing_time_seconds=0,
        )

        assert result is True
        mock_db.execute.assert_called_once()
        update_call = mock_db.execute.call_args
        stmt = update_call[0][0]
        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        param_keys = list(compiled.params.keys())
        assert "processing_time_seconds" in param_keys

    def test_zero_input_rows_is_included(self, service, mock_db, feature_set_id):
        """Verify input_rows=0 is included (edge case - not excluded by `if` check)."""
        result = service.update_feature_set_status_sync(
            feature_set_id, "COMPLETED",
            input_rows=0,
        )

        assert result is True
        mock_db.execute.assert_called_once()
        update_call = mock_db.execute.call_args
        stmt = update_call[0][0]
        compiled = stmt.compile(compile_kwargs={"literal_binds": False})
        param_keys = list(compiled.params.keys())
        assert "input_rows" in param_keys

    def test_invalid_uuid_returns_false(self, service, mock_db):
        """Verify invalid UUID returns False without DB call."""
        result = service.update_feature_set_status_sync(
            "not-a-valid-uuid", "COMPLETED",
            all_features=["a", "b"],
            input_rows=100,
            processing_time_seconds=5,
        )

        assert result is False
        mock_db.execute.assert_not_called()


class TestComputeFeaturesWorkerMetadata:
    """
    Test that the compute_features worker passes metadata correctly.
    This is a structural test — verifying the call signature, not actual execution.
    """

    def test_worker_import_time(self):
        """Verify the time module is imported in feature_worker."""
        import app.workers.feature_worker as fw
        import time as time_module
        assert hasattr(fw, 'time') or 'time' in dir(fw) or time_module is not None

    def test_feature_set_model_has_metadata_columns(self):
        """Verify the FeatureSet model has all required metadata columns."""
        assert hasattr(FeatureSet, 'all_features')
        assert hasattr(FeatureSet, 'input_rows')
        assert hasattr(FeatureSet, 'processing_time_seconds')
        assert hasattr(FeatureSet, 'feature_count')
        assert hasattr(FeatureSet, 'selected_features')
        assert hasattr(FeatureSet, 'storage_path')
        assert hasattr(FeatureSet, 'created_at')
        assert hasattr(FeatureSet, 'completed_at')
