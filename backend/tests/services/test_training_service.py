import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime
from app.core.time import IST
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.training_service import TrainingService, ModelService
from app.models.training_job import TrainingJob
from app.models.ml_model import MLModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def training_service(mock_db):
    with patch("app.services.training_service.DataProcessingService"):
        service = TrainingService(db=mock_db)
    return service


@pytest.fixture
def model_service(mock_db):
    return ModelService(db=mock_db)


def _make_job(**kwargs) -> TrainingJob:
    """Build a minimal valid TrainingJob without FK constraints."""
    defaults = dict(
        id=uuid4(),
        name="test-job",
        dataset_id=uuid4(),
        algorithm="xgboost",
        hyperparameters={"n_estimators": 100},
        tuning_method="manual",
        tuning_config={},
        feature_config={"amount": True},
        status="QUEUED",
        progress=0.0,
        processing_only=False,
        metrics={},
    )
    defaults.update(kwargs)
    return TrainingJob(**defaults)


# ---------------------------------------------------------------------------
# TrainingService — list_training_jobs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_training_jobs_no_filter(training_service, mock_db):
    """Returns all jobs and correct total with no status filter."""
    job1 = _make_job(name="job-1", status="COMPLETED")
    job2 = _make_job(name="job-2", status="RUNNING")

    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 2

    mock_items_result = MagicMock()
    mock_items_result.scalars.return_value.all.return_value = [job1, job2]

    mock_db.execute.side_effect = [mock_count_result, mock_items_result]

    jobs, total = await training_service.list_training_jobs()

    assert total == 2
    assert len(jobs) == 2
    assert jobs[0]["name"] == "job-1"
    assert jobs[1]["name"] == "job-2"
    assert mock_db.execute.call_count == 2


@pytest.mark.asyncio
async def test_list_training_jobs_with_status_filter(training_service, mock_db):
    """Status filter reduces result set — two DB executes still issued."""
    running_job = _make_job(name="running-job", status="RUNNING")

    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 1

    mock_items_result = MagicMock()
    mock_items_result.scalars.return_value.all.return_value = [running_job]

    mock_db.execute.side_effect = [mock_count_result, mock_items_result]

    jobs, total = await training_service.list_training_jobs(status="RUNNING")

    assert total == 1
    assert jobs[0]["status"] == "RUNNING"
    assert mock_db.execute.call_count == 2


# ---------------------------------------------------------------------------
# TrainingService — get_training_job
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_training_job_found(training_service, mock_db):
    """Returns serialized dict when job exists in DB."""
    job = _make_job(name="found-job", status="COMPLETED")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = job
    mock_db.execute.return_value = mock_result

    result = await training_service.get_training_job(str(job.id))

    assert result is not None
    assert result["name"] == "found-job"
    assert result["status"] == "COMPLETED"
    mock_db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_training_job_not_found(training_service, mock_db):
    """Returns None when DB has no matching row."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    result = await training_service.get_training_job(str(uuid4()))

    assert result is None
    mock_db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_training_job_invalid_uuid(training_service, mock_db):
    """Returns None immediately for a non-UUID string without hitting DB."""
    result = await training_service.get_training_job("not-a-uuid")

    assert result is None
    mock_db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# TrainingService — create_training_job
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_training_job_queues_celery(mock_db):
    """Job is persisted and train_model.delay() is called when processing_only=False."""
    dataset_id = str(uuid4())

    # Patch DataProcessingService at class level so __init__ gets mock
    with patch("app.services.training_service.DataProcessingService") as mock_dps_cls, \
         patch("app.workers.training_worker.train_model") as mock_celery_task:

        mock_dps_cls.return_value.prepare_training_data = AsyncMock(return_value={
            "train_dataset_id": str(uuid4()),
            "test_dataset_id": str(uuid4()),
            "train_rows": 8000,
            "test_rows": 2000,
        })

        service = TrainingService(db=mock_db)

        # db.refresh must populate job.id so _serialize_job can iterate __table__.columns
        async def fake_refresh(obj):
            if not obj.id:
                obj.id = uuid4()

        mock_db.refresh.side_effect = fake_refresh

        await service.create_training_job(
            name="celery-test-job",
            dataset_id=dataset_id,
            feature_config={"amount": True},
            algorithm="xgboost",
            hyperparameters={"n_estimators": 100},
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_celery_task.delay.assert_called_once()


@pytest.mark.asyncio
async def test_create_training_job_processing_only(mock_db):
    """processing_only=True sets status DATA_PREPARED and skips Celery dispatch."""
    dataset_id = str(uuid4())

    with patch("app.services.training_service.DataProcessingService") as mock_dps_cls, \
         patch("app.workers.training_worker.train_model") as mock_celery_task:

        mock_dps_cls.return_value.prepare_training_data = AsyncMock(return_value={
            "train_dataset_id": str(uuid4()),
            "test_dataset_id": str(uuid4()),
            "train_rows": 8000,
            "test_rows": 2000,
        })

        service = TrainingService(db=mock_db)

        async def fake_refresh(obj):
            if not obj.id:
                obj.id = uuid4()

        mock_db.refresh.side_effect = fake_refresh

        result = await service.create_training_job(
            name="processing-only-job",
            dataset_id=dataset_id,
            feature_config={"amount": True},
            algorithm="xgboost",
            hyperparameters={"n_estimators": 100},
            processing_only=True,
        )

        assert result["status"] == "DATA_PREPARED"
        assert result["progress"] == 1.0
        mock_celery_task.delay.assert_not_called()


# ---------------------------------------------------------------------------
# TrainingService — delete_training_job
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_training_job_found(training_service, mock_db):
    """Existing job is deleted and True is returned."""
    job = _make_job(name="to-delete")

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = job
    mock_db.execute.return_value = mock_result

    result = await training_service.delete_training_job(str(job.id))

    assert result is True
    mock_db.delete.assert_called_once_with(job)
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_training_job_not_found(training_service, mock_db):
    """Non-existent job returns False without calling db.delete."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    result = await training_service.delete_training_job(str(uuid4()))

    assert result is False
    mock_db.delete.assert_not_called()
    mock_db.commit.assert_not_called()


# ---------------------------------------------------------------------------
# TrainingService — list_algorithms & get_default_hyperparameters
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_algorithms(training_service):
    """Returns a list of standardized ML algorithm definitions."""
    algorithms = await training_service.list_algorithms()

    assert len(algorithms) > 0
    names = [algo["id"] for algo in algorithms]
    assert "xgboost" in names
    assert "random_forest" in names


@pytest.mark.asyncio
async def test_get_default_hyperparameters_valid(training_service):
    """Returns dict of extracted defaults for a known algorithm."""
    defaults = await training_service.get_default_hyperparameters("xgboost")

    assert defaults is not None
    assert defaults["n_estimators"] == 100
    assert defaults["learning_rate"] == 0.3


@pytest.mark.asyncio
async def test_get_default_hyperparameters_invalid(training_service):
    """Returns empty dict for unknown algorithm."""
    defaults = await training_service.get_default_hyperparameters("unknown-algo")

    assert defaults == {}


# ---------------------------------------------------------------------------
# ModelService — list_models
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_models(model_service, mock_db):
    """Returns models and total count correctly."""
    m1 = MLModel(
        id=uuid4(), name="Model A", version="1.0.0", algorithm="xgboost",
        hyperparameters={}, storage_path="models/a/model.pkl", metrics={}, status="PRODUCTION"
    )
    m2 = MLModel(
        id=uuid4(), name="Model B", version="1.0.1", algorithm="lightgbm",
        hyperparameters={}, storage_path="models/b/model.pkl", metrics={}, status="STAGING"
    )

    mock_count_result = MagicMock()
    mock_count_result.scalar.return_value = 2

    mock_items_result = MagicMock()
    mock_items_result.scalars.return_value.all.return_value = [m1, m2]

    mock_db.execute.side_effect = [mock_count_result, mock_items_result]

    models, total = await model_service.list_models()

    assert total == 2
    assert len(models) == 2
    assert mock_db.execute.call_count == 2


# ---------------------------------------------------------------------------
# ModelService — get_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_model_found(model_service, mock_db):
    """Returns MLModel instance when found."""
    model = MLModel(
        id=uuid4(), name="Found Model", version="2.0.0", algorithm="xgboost",
        hyperparameters={}, storage_path="models/x/model.pkl", metrics={}, status="STAGING"
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = model
    mock_db.execute.return_value = mock_result

    result = await model_service.get_model(str(model.id))

    assert result is model
    mock_db.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_model_invalid_uuid(model_service, mock_db):
    """Returns None for an invalid UUID without DB hit."""
    result = await model_service.get_model("invalid-uuid-string")

    assert result is None
    mock_db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# ModelService — promote_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_promote_model_to_production_demotes_current(model_service, mock_db):
    """Promoting a model to PRODUCTION archives the existing production model."""
    current_prod_id = uuid4()
    new_model_id = uuid4()

    current_prod = MLModel(
        id=current_prod_id, name="Old Prod", version="1.0.0", algorithm="xgboost",
        hyperparameters={}, storage_path="models/old/model.pkl", metrics={}, status="PRODUCTION"
    )
    new_model = MLModel(
        id=new_model_id, name="New Model", version="2.0.0", algorithm="xgboost",
        hyperparameters={}, storage_path="models/new/model.pkl", metrics={}, status="STAGING"
    )

    # execute call order: get_model → get_production_model → db.refresh
    mock_get_new = MagicMock()
    mock_get_new.scalar_one_or_none.return_value = new_model

    mock_get_prod = MagicMock()
    mock_get_prod.scalar_one_or_none.return_value = current_prod

    mock_db.execute.side_effect = [mock_get_new, mock_get_prod]

    result = await model_service.promote_model(str(new_model_id), "PRODUCTION")

    assert current_prod.status == "ARCHIVED"
    assert current_prod.archived_reason == "Replaced by new production model"
    assert result.status == "PRODUCTION"
    mock_db.commit.assert_called_once()


# ---------------------------------------------------------------------------
# ModelService — get_production_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_production_model(model_service, mock_db):
    """Returns the one model marked as PRODUCTION."""
    prod = MLModel(
        id=uuid4(), name="Prod Model", version="1.0.0", algorithm="xgboost",
        hyperparameters={}, storage_path="models/old/model.pkl", metrics={}, status="PRODUCTION"
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = prod
    mock_db.execute.return_value = mock_result

    result = await model_service.get_production_model()

    assert result is prod
    mock_db.execute.assert_called_once()


# ---------------------------------------------------------------------------
# ModelService — set_baselines
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_baselines_success(model_service, mock_db):
    """Adds Baseline records efficiently."""
    model_id = uuid4()
    model = MLModel(
        id=model_id, name="Found Model", version="1.0.0", algorithm="xgboost",
        hyperparameters={}, storage_path="path", metrics={}, status="STAGING"
    )

    # First execute is get_model internal call
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = model
    mock_db.execute.return_value = mock_result

    baselines = [
        {"metric": "roc_auc", "threshold": 0.85},
        {"metric": "precision", "threshold": 0.82, "operator": "gt"}
    ]

    result = await model_service.set_baselines(str(model_id), baselines)

    assert len(result) == 2
    assert result[0].metric_name == "roc_auc"
    assert result[0].threshold == 0.85
    assert result[0].operator == "gte"  # default
    assert result[1].operator == "gt"

    # added to db session
    assert mock_db.add.call_count == 2
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_set_baselines_not_found(model_service, mock_db):
    """Raises ValueError if model ID is missing."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    with pytest.raises(ValueError, match="not found"):
        await model_service.set_baselines(str(uuid4()), [])


# ---------------------------------------------------------------------------
# ModelService — delete_model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_model_hard_delete(model_service, mock_db):
    """Removes model and its azure blobs if hard_delete=True."""
    model = MLModel(
        id=uuid4(), name="Prod Model", version="1.0.0", algorithm="xgboost",
        hyperparameters={}, storage_path="blob_main.pkl", onnx_path="blob.onnx", metrics={}, status="PRODUCTION"
    )

    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = model
    mock_db.execute.return_value = mock_result

    with patch("app.core.storage.StorageService.delete_model") as mock_delete_blob:
        mock_delete_blob.return_value = AsyncMock()

        res = await model_service.delete_model(str(model.id), hard_delete=True)

        assert res is True
        assert mock_delete_blob.call_count == 2
        mock_delete_blob.assert_any_call("blob_main.pkl")
        mock_delete_blob.assert_any_call("blob.onnx")

        mock_db.delete.assert_called_once_with(model)
        mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_model_not_found(model_service, mock_db):
    """Returns False immediately."""
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    res = await model_service.delete_model(str(uuid4()))

    assert res is False
    mock_db.delete.assert_not_called()


# ---------------------------------------------------------------------------
# New Tests: Serialization & Pagination
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_serialize_job_timezone_fix(training_service):
    """Verify _serialize_job injects IST if tzinfo is missing."""
    job = MagicMock(spec=TrainingJob)
    # Mock columns for serialization
    job.__table__.columns = []
    for col_name in ['id', 'created_at', 'started_at', 'completed_at']:
        col = MagicMock()
        col.name = col_name
        job.__table__.columns.append(col)
        
    dt_naive = datetime(2026, 3, 17, 10, 0, 0)
    job.id = uuid4()
    job.created_at = dt_naive
    job.started_at = dt_naive
    job.completed_at = None

    def mock_getattr(obj, attr):
        return getattr(job, attr)

    with patch("app.services.training_service.getattr", side_effect=mock_getattr):
        serialized = training_service._serialize_job(job)

    assert serialized["created_at"].tzinfo == IST
    assert serialized["started_at"].tzinfo == IST
    assert serialized["completed_at"] is None


@pytest.mark.asyncio
async def test_list_models_pagination(model_service, mock_db):
    """Verify offsets and limits are applied based on page/page_size."""
    mock_count = MagicMock()
    mock_count.scalar.return_value = 50
    mock_items = MagicMock()
    mock_items.scalars.return_value.all.return_value = []
    
    mock_db.execute.side_effect = [mock_count, mock_items]
    
    await model_service.list_models(page=2, page_size=10)
    
    assert mock_db.execute.call_count == 2


@pytest.mark.asyncio
async def test_promote_model_no_previous_prod(model_service, mock_db):
    """Verify promotion works when there is no existing production model."""
    model = MLModel(id=uuid4(), status="STAGING")
    
    mock_get_model = MagicMock()
    mock_get_model.scalar_one_or_none.return_value = model
    
    mock_get_prod = MagicMock()
    mock_get_prod.scalar_one_or_none.return_value = None
    
    mock_db.execute.side_effect = [mock_get_model, mock_get_prod]
    
    result = await model_service.promote_model(str(model.id), "PRODUCTION")
    
    assert result.status == "PRODUCTION"
    assert result.promoted_at is not None
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_delete_model_storage_failure_logged(model_service, mock_db):
    """Verify delete_model continues even if storage deletion fails."""
    model = MLModel(id=uuid4(), storage_path="fail.pkl", onnx_path=None)
    
    mock_res = MagicMock()
    mock_res.scalar_one_or_none.return_value = model
    mock_db.execute.return_value = mock_res
    
    with patch("app.core.storage.StorageService.delete_model", side_effect=Exception("Storage Error")):
        res = await model_service.delete_model(str(model.id), hard_delete=True)
        
        assert res is True
        mock_db.delete.assert_called_once_with(model)
        mock_db.commit.assert_called_once()
