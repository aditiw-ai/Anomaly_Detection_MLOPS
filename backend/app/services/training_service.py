"""
Training Service
Business logic for model training operations.
"""
from app.core.time import IST, now_ist
from typing import Optional, Tuple, List, Dict, Any
from uuid import UUID
from datetime import datetime, timezone
import logging
import hashlib
import json

from sqlalchemy import select, func, update, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.ml_model import MLModel, Baseline
from app.models.feature_set import FeatureSet
from app.models.training_job import TrainingJob
from app.services.data_processing_service import DataProcessingService

logger = logging.getLogger(__name__)


def _safe_uuid_user_id(user_id: Optional[str], context: str) -> Optional[str]:
    """
    Validate user_id for UUID-backed created_by columns.
    Returns normalized UUID string or None when invalid.
    """
    if not user_id:
        return None
    try:
        return str(UUID(str(user_id)))
    except (ValueError, TypeError):
        logger.warning(
            "Skipping created_by user filter in %s due to non-UUID user_id=%r",
            context,
            user_id,
        )
        return None


class TrainingService:
    """Service for model training operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.data_processing = DataProcessingService(db)
    
    async def list_training_jobs(
        self,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[str] = None,
    ) -> Tuple[List[TrainingJob], int]:
        """List training jobs with pagination."""
        query = select(TrainingJob).order_by(desc(TrainingJob.created_at))
        count_query = select(func.count(TrainingJob.id))
        
        if status:
            query = query.where(TrainingJob.status == status)
            count_query = count_query.where(TrainingJob.status == status)
            
        safe_user_id = _safe_uuid_user_id(user_id, "list_training_jobs")
        if safe_user_id:
            query = query.where(TrainingJob.created_by == safe_user_id)
            count_query = count_query.where(TrainingJob.created_by == safe_user_id)
        
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        
        result = await self.db.execute(query)
        jobs = result.scalars().all()
        
        # Serialize and fix timestamps
        return [self._serialize_job(job) for job in jobs], total

    def _serialize_job(self, job: TrainingJob) -> Dict[str, Any]:
        """Convert job to dict and fix timestamps."""
        data = {c.name: getattr(job, c.name) for c in job.__table__.columns}
        
        # Add timezone info
        for field in ['created_at', 'started_at', 'completed_at']:
            val = data.get(field)
            if val and val.tzinfo is None:
                data[field] = val.replace(tzinfo=IST)
                
        return data
    
    async def create_training_job(
        self,
        name: str,
        dataset_id: str,
        feature_config: Dict[str, bool],
        algorithm: str,
        hyperparameters: Dict[str, Any],
        feature_set_id: Optional[str] = None,
        tuning_method: str = "manual",
        tuning_config: Dict[str, Any] = None,
        processing_only: bool = False,
        user_id: Optional[str] = None,
        enforce_preprocessing_commit: bool = False,
    ) -> TrainingJob:
        """Create a new training job."""
        preprocessing_snapshot = None
        preprocessing_checksum = None
        preprocessing_version = None

        if not processing_only and enforce_preprocessing_commit:
            if not feature_set_id:
                raise ValueError(
                    "A committed preprocessing plan is required before training. "
                    "Generate and commit a Feature Set preprocessing plan first."
                )
            try:
                fs_uuid = UUID(feature_set_id)
            except ValueError:
                raise ValueError("Invalid feature_set_id")

            fs_result = await self.db.execute(select(FeatureSet).where(FeatureSet.id == fs_uuid))
            feature_set = fs_result.scalar_one_or_none()
            if not feature_set:
                raise ValueError(f"Feature set {feature_set_id} not found")
            if not feature_set.preprocessing_committed_at or int(feature_set.preprocessing_plan_version or 0) <= 0:
                raise ValueError(
                    "Preprocessing plan is not committed. Please commit preprocessing decisions before training."
                )

            preprocessing_snapshot = (
                (feature_set.config or {}).get("preprocessing_plan_snapshot")
                or {
                    "feature_set_id": str(feature_set.id),
                    "plan_version": int(feature_set.preprocessing_plan_version or 0),
                    "profile": feature_set.preprocessing_profile or [],
                    "recommendations": feature_set.preprocessing_recommendations or {},
                    "decisions": feature_set.preprocessing_decisions or {},
                    "committed_at": feature_set.preprocessing_committed_at.isoformat(),
                    "committed_by": feature_set.preprocessing_committed_by,
                }
            )
            preprocessing_checksum = hashlib.sha256(
                json.dumps(preprocessing_snapshot, sort_keys=True).encode("utf-8")
            ).hexdigest()
            preprocessing_version = int(feature_set.preprocessing_plan_version or 0)
        
        # 1. Prepare Data (Split & Save)
        # This is blocking/awaitable - if this fails, we don't create the job
        processed_data = await self.data_processing.prepare_training_data(
            dataset_id=dataset_id,
            feature_config=feature_config,
            test_size=hyperparameters.get("test_size", 0.2),
        )
        
        status = "QUEUED"
        progress = 0.0
        
        if processing_only:
            status = "DATA_PREPARED"
            progress = 1.0
        
        # 2. Create Job Record
        job = TrainingJob(
            name=name,
            dataset_id=UUID(dataset_id),
            feature_set_id=UUID(feature_set_id) if feature_set_id else None,
            feature_config=feature_config,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            tuning_method=tuning_method,
            tuning_config=tuning_config or {},
            status=status,
            progress=progress,
            processing_only=processing_only,
            metrics={
                "train_dataset_id": processed_data.get("train_dataset_id"),
                "test_dataset_id": processed_data.get("test_dataset_id"),
                "train_rows": processed_data.get("train_rows"),
                "test_rows": processed_data.get("test_rows"),
                "preprocessing_plan_snapshot": preprocessing_snapshot,
                "preprocessing_plan_checksum": preprocessing_checksum,
                "preprocessing_plan_version": preprocessing_version,
                "preprocessing_feature_set_id": feature_set_id,
            },
            created_by=_safe_uuid_user_id(user_id, "create_training_job"),
            created_at=now_ist()
        )
        
        self.db.add(job)
        try:
            await self.db.commit()
            await self.db.refresh(job)
        except Exception as e:
            await self.db.rollback()
            raise ValueError(f"Failed to create training job record: {e}")
        
        
        # 3. Trigger Async Training (if not just processing)
        if not processing_only:
            from app.workers.training_worker import train_model
            # Pass the DB ID to the worker
            train_model.delay(str(job.id))
        else:
            logger.info(f"Skipping training worker for job {job.id} (processing_only=True)")
        
        
        logger.info(f"Created training job {job.id}")
        
        return self._serialize_job(job)
    
    async def get_training_job(self, job_id: str, user_id: Optional[str] = None) -> Optional[TrainingJob]:
        """Get training job status."""
        try:
            uuid_id = UUID(job_id)
        except ValueError:
            return None
            
        stmt = select(TrainingJob).where(TrainingJob.id == uuid_id)
        safe_user_id = _safe_uuid_user_id(user_id, "get_training_job")
        if safe_user_id:
            stmt = stmt.where(TrainingJob.created_by == safe_user_id)
            
        result = await self.db.execute(stmt)
        job = result.scalar_one_or_none()

        if job:
            return self._serialize_job(job)
                
        return None
    
    async def list_algorithms(self) -> List[Dict]:
        """List available ML algorithms."""
        return [
            {
                "id": "xgboost",
                "name": "XGBoost",
                "description": "Gradient boosting optimized for tabular data. Best for fraud detection.",
                "hyperparameters": [
                    # ── Learning ──────────────────────────────────────────────
                    {
                        "name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000,
                        "group": "Learning",
                        "description": "Number of boosting rounds (trees). Higher = more capacity but slower and may overfit.",
                    },
                    {
                        "name": "learning_rate", "type": "float", "default": 0.3, "min": 0.0, "max": 1.0,
                        "group": "Learning",
                        "description": "Step size shrinkage (eta). Shrinks feature weights after each step to prevent overfitting.",
                    },
                    {
                        "name": "base_score", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                        "group": "Learning",
                        "description": "Initial prediction score of all instances (global bias).",
                    },
                    {
                        "name": "objective", "type": "select",
                        "default": "binary:logistic",
                        "options": [
                            {"label": "binary:logistic (binary classification)", "value": "binary:logistic"},
                            {"label": "binary:logitraw (raw score, no sigmoid)", "value": "binary:logitraw"},
                            {"label": "multi:softmax (multiclass labels)", "value": "multi:softmax"},
                            {"label": "multi:softprob (multiclass probs)", "value": "multi:softprob"},
                            {"label": "reg:squarederror (regression, MSE)", "value": "reg:squarederror"},
                            {"label": "reg:logistic (logistic regression)", "value": "reg:logistic"},
                            {"label": "reg:tweedie (Tweedie regression)", "value": "reg:tweedie"},
                            {"label": "rank:pairwise (pairwise ranking)", "value": "rank:pairwise"},
                        ],
                        "group": "Learning",
                        "description": "Specifies the learning task and the corresponding learning objective.",
                    },
                    {
                        "name": "eval_metric", "type": "select",
                        "default": "logloss",
                        "options": [
                            {"label": "logloss (negative log-likelihood)", "value": "logloss"},
                            {"label": "auc (area under ROC curve)", "value": "auc"},
                            {"label": "aucpr (area under PR curve)", "value": "aucpr"},
                            {"label": "error (binary classification error)", "value": "error"},
                            {"label": "rmse (root mean square error)", "value": "rmse"},
                            {"label": "mae (mean absolute error)", "value": "mae"},
                            {"label": "map (mean average precision)", "value": "map"},
                        ],
                        "group": "Learning",
                        "description": "Evaluation metric for validation data. Defaults: rmse for regression, error for classification.",
                    },
                    {
                        "name": "early_stopping_rounds", "type": "int", "default": 0, "min": 0, "max": 200,
                        "group": "Learning",
                        "description": "Stop training when validation score doesn't improve for this many rounds. 0 = disabled.",
                    },
                    {
                        "name": "seed", "type": "int", "default": 0, "min": 0, "max": 99999,
                        "group": "Learning",
                        "description": "Random number seed for reproducibility.",
                    },
                    {
                        "name": "verbosity", "type": "int", "default": 1, "min": 0, "max": 3,
                        "group": "Learning",
                        "description": "Verbosity of printing messages. 0=silent, 1=warning, 2=info, 3=debug.",
                    },

                    # ── Tree Structure ────────────────────────────────────────
                    {
                        "name": "booster", "type": "select",
                        "default": "gbtree",
                        "options": [
                            {"label": "gbtree (tree-based model)", "value": "gbtree"},
                            {"label": "gblinear (linear model)", "value": "gblinear"},
                            {"label": "dart (dropout trees)", "value": "dart"},
                        ],
                        "group": "Tree Structure",
                        "description": "Which booster to use. gbtree and dart use tree-based models; gblinear uses a linear function.",
                    },
                    {
                        "name": "tree_method", "type": "select",
                        "default": "hist",
                        "options": [
                            {"label": "auto (automatic selection)", "value": "auto"},
                            {"label": "exact (exact greedy)", "value": "exact"},
                            {"label": "approx (approximate greedy)", "value": "approx"},
                            {"label": "hist (histogram-based)", "value": "hist"},
                            {"label": "gpu_hist (GPU histogram)", "value": "gpu_hist"},
                        ],
                        "group": "Tree Structure",
                        "description": "Tree construction algorithm. hist is fastest for large datasets.",
                    },
                    {
                        "name": "max_depth", "type": "int", "default": 6, "min": 0, "max": 20,
                        "group": "Tree Structure",
                        "description": "Maximum depth of a tree. Increasing makes model more complex and likely to overfit. 0 = no limit.",
                    },
                    {
                        "name": "grow_policy", "type": "select",
                        "default": "depthwise",
                        "options": [
                            {"label": "depthwise (grow level by level)", "value": "depthwise"},
                            {"label": "lossguide (grow nodes with max loss change)", "value": "lossguide"},
                        ],
                        "group": "Tree Structure",
                        "description": "Controls how new nodes are added to the tree. Only supported when tree_method=hist.",
                    },
                    {
                        "name": "max_leaves", "type": "int", "default": 0, "min": 0, "max": 256,
                        "group": "Tree Structure",
                        "description": "Maximum number of nodes to add. Relevant only when grow_policy=lossguide. 0 = no limit.",
                    },
                    {
                        "name": "max_bin", "type": "int", "default": 256, "min": 32, "max": 1024,
                        "group": "Tree Structure",
                        "description": "Maximum number of discrete bins to bucket continuous features. Used only if tree_method=hist.",
                    },
                    {
                        "name": "max_delta_step", "type": "int", "default": 0, "min": 0, "max": 10,
                        "group": "Tree Structure",
                        "description": "Maximum delta step for each tree's weight estimation. Useful for logistic regression. Set to 1-10 to control updates.",
                    },

                    # ── Sampling ──────────────────────────────────────────────
                    {
                        "name": "subsample", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0,
                        "group": "Sampling",
                        "description": "Subsample ratio of the training instance. 0.5 = random half of data per tree. Prevents overfitting.",
                    },
                    {
                        "name": "colsample_bytree", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0,
                        "group": "Sampling",
                        "description": "Subsample ratio of columns when constructing each tree.",
                    },
                    {
                        "name": "colsample_bylevel", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0,
                        "group": "Sampling",
                        "description": "Subsample ratio of columns for each split, at each tree level.",
                    },
                    {
                        "name": "colsample_bynode", "type": "float", "default": 1.0, "min": 0.0, "max": 1.0,
                        "group": "Sampling",
                        "description": "Subsample ratio of columns from each tree node (split).",
                    },

                    # ── Regularization ────────────────────────────────────────
                    {
                        "name": "gamma", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0,
                        "group": "Regularization",
                        "description": "Minimum loss reduction to make a further partition on a leaf node. Higher = more conservative.",
                    },
                    {
                        "name": "alpha", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0,
                        "group": "Regularization",
                        "description": "L1 regularization term on weights. Increasing makes models more conservative.",
                    },
                    {
                        "name": "lambda", "type": "float", "default": 1.0, "min": 0.0, "max": 10.0,
                        "group": "Regularization",
                        "description": "L2 regularization term on weights. Increasing makes models more conservative.",
                    },
                    {
                        "name": "min_child_weight", "type": "float", "default": 1.0, "min": 0.0, "max": 20.0,
                        "group": "Regularization",
                        "description": "Minimum sum of instance weight (hessian) needed in a child. Higher = more conservative.",
                    },
                    {
                        "name": "scale_pos_weight", "type": "float", "default": 1.0, "min": 0.1, "max": 100.0,
                        "group": "Regularization",
                        "description": "Controls balance of positive and negative weights. Useful for unbalanced classes. Typical: sum(negatives)/sum(positives).",
                    },

                    # ── DART ──────────────────────────────────────────────────
                    {
                        "name": "rate_drop", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
                        "group": "DART",
                        "description": "Dropout rate — fraction of previous trees to drop during each DART boosting round. Only applies when booster=dart.",
                    },
                    {
                        "name": "skip_drop", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
                        "group": "DART",
                        "description": "Probability of skipping the dropout procedure during a DART boosting iteration. Only applies when booster=dart.",
                    },
                    {
                        "name": "normalize_type", "type": "select",
                        "default": "tree",
                        "options": [
                            {"label": "tree (normalize by trees)", "value": "tree"},
                            {"label": "forest (normalize by forest)", "value": "forest"},
                        ],
                        "group": "DART",
                        "description": "Type of normalization algorithm for DART. Only applies when booster=dart.",
                    },
                    {
                        "name": "sample_type", "type": "select",
                        "default": "uniform",
                        "options": [
                            {"label": "uniform (uniform random)", "value": "uniform"},
                            {"label": "weighted (weighted by dropped trees' weight)", "value": "weighted"},
                        ],
                        "group": "DART",
                        "description": "Type of sampling algorithm for DART. Only applies when booster=dart.",
                    },
                ],
            },
            {
                "id": "lightgbm",
                "name": "LightGBM",
                "description": "Fast gradient boosting with leaf-wise tree growth.",
                "hyperparameters": [
                    # ── Learning ──────────────────────────────────────────────
                    {
                        "name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000,
                        "group": "Learning",
                        "description": "Number of boosting iterations (num_boost_round). Higher = more capacity but slower.",
                    },
                    {
                        "name": "learning_rate", "type": "float", "default": 0.1, "min": 0.001, "max": 1.0,
                        "group": "Learning",
                        "description": "Rate at which model weights are updated after each iteration. Lower = slower but more accurate.",
                    },
                    {
                        "name": "early_stopping_rounds", "type": "int", "default": 10, "min": 0, "max": 200,
                        "group": "Learning",
                        "description": "Stop if validation metric does not improve in this many rounds. 0 = disabled.",
                    },
                    {
                        "name": "metric", "type": "select",
                        "default": "auto",
                        "options": [
                            {"label": "auto (automatic based on task)", "value": "auto"},
                            {"label": "binary_logloss (binary classification)", "value": "binary_logloss"},
                            {"label": "binary_error (binary error rate)", "value": "binary_error"},
                            {"label": "auc (area under ROC curve)", "value": "auc"},
                            {"label": "average_precision (PR-AUC)", "value": "average_precision"},
                            {"label": "multi_logloss (multiclass log loss)", "value": "multi_logloss"},
                            {"label": "multi_error (multiclass error)", "value": "multi_error"},
                            {"label": "rmse (root mean square error)", "value": "rmse"},
                            {"label": "l1 (mean absolute error)", "value": "l1"},
                            {"label": "l2 (mean squared error)", "value": "l2"},
                            {"label": "huber (Huber loss)", "value": "huber"},
                            {"label": "fair (Fair loss)", "value": "fair"},
                            {"label": "cross_entropy", "value": "cross_entropy"},
                        ],
                        "group": "Learning",
                        "description": "Evaluation metric for validation data. 'auto' picks based on task type.",
                    },
                    {
                        "name": "boosting", "type": "select",
                        "default": "gbdt",
                        "options": [
                            {"label": "gbdt (gradient boosted decision tree)", "value": "gbdt"},
                            {"label": "rf (random forest)", "value": "rf"},
                            {"label": "dart (dropout meets additive regression trees)", "value": "dart"},
                            {"label": "goss (gradient-based one-side sampling)", "value": "goss"},
                        ],
                        "group": "Learning",
                        "description": "Boosting type / algorithm to use.",
                    },
                    {
                        "name": "verbosity", "type": "int", "default": 1, "min": -1, "max": 2,
                        "group": "Learning",
                        "description": "Verbosity. <0: fatal only, 0: errors+warnings, 1: info, >1: debug.",
                    },

                    # ── Tree Structure ────────────────────────────────────────
                    {
                        "name": "num_leaves", "type": "int", "default": 64, "min": 2, "max": 131072,
                        "group": "Tree Structure",
                        "description": "Maximum number of leaves in one tree. Main parameter to control model complexity.",
                    },
                    {
                        "name": "max_depth", "type": "int", "default": 6, "min": -1, "max": 50,
                        "group": "Tree Structure",
                        "description": "Maximum tree depth. -1 = no limit. Used to control overfitting on small datasets.",
                    },
                    {
                        "name": "min_data_in_leaf", "type": "int", "default": 3, "min": 0, "max": 500,
                        "group": "Tree Structure",
                        "description": "Minimum number of samples in a leaf. Higher = more conservative (prevents overfitting).",
                    },
                    {
                        "name": "max_delta_step", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0,
                        "group": "Tree Structure",
                        "description": "Limits max leaf output. Final max output = learning_rate * max_delta_step. 0 = no limit.",
                    },
                    {
                        "name": "min_gain_to_split", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0,
                        "group": "Tree Structure",
                        "description": "Minimum gain required to perform a split. Higher = fewer splits (faster, more conservative).",
                    },
                    {
                        "name": "max_bin", "type": "int", "default": 255, "min": 2, "max": 1024,
                        "group": "Tree Structure",
                        "description": "Max number of bins to bucket feature values. Smaller = faster but less accurate.",
                    },
                    {
                        "name": "tree_learner", "type": "select",
                        "default": "serial",
                        "options": [
                            {"label": "serial (single machine)", "value": "serial"},
                            {"label": "feature (feature parallel)", "value": "feature"},
                            {"label": "data (data parallel)", "value": "data"},
                            {"label": "voting (voting parallel)", "value": "voting"},
                        ],
                        "group": "Tree Structure",
                        "description": "Tree learning parallelism strategy.",
                    },

                    # ── Sampling ──────────────────────────────────────────────
                    {
                        "name": "feature_fraction", "type": "float", "default": 0.9, "min": 0.01, "max": 1.0,
                        "group": "Sampling",
                        "description": "Fraction of features selected per iteration (tree). Must be < 1.0 to activate.",
                    },
                    {
                        "name": "feature_fraction_bynode", "type": "float", "default": 1.0, "min": 0.01, "max": 1.0,
                        "group": "Sampling",
                        "description": "Fraction of features selected at each tree node split. Helps with overfitting.",
                    },
                    {
                        "name": "bagging_fraction", "type": "float", "default": 0.9, "min": 0.01, "max": 1.0,
                        "group": "Sampling",
                        "description": "Fraction of data used per bagging iteration (without replacement). Used with bagging_freq > 0.",
                    },
                    {
                        "name": "bagging_freq", "type": "int", "default": 1, "min": 0, "max": 50,
                        "group": "Sampling",
                        "description": "Frequency to perform bagging. 0 = disable bagging. Every k iterations, randomly select bagging_fraction of data.",
                    },

                    # ── Regularization ────────────────────────────────────────
                    {
                        "name": "lambda_l1", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0,
                        "group": "Regularization",
                        "description": "L1 regularization term. Increasing makes the model more conservative.",
                    },
                    {
                        "name": "lambda_l2", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0,
                        "group": "Regularization",
                        "description": "L2 regularization term. Increasing makes the model more conservative.",
                    },
                    {
                        "name": "scale_pos_weight", "type": "float", "default": 1.0, "min": 0.01, "max": 100.0,
                        "group": "Regularization",
                        "description": "Weight for positive class labels. Useful for imbalanced datasets. Cannot be used with is_unbalance=True.",
                    },
                    {
                        "name": "is_unbalance", "type": "select",
                        "default": "False",
                        "options": [
                            {"label": "False (balanced or manual weight)", "value": "False"},
                            {"label": "True (auto-rebalance positive class)", "value": "True"},
                        ],
                        "group": "Regularization",
                        "description": "Set True if training data is unbalanced (binary only). Cannot be used with scale_pos_weight.",
                    },
                    {
                        "name": "tweedie_variance_power", "type": "float", "default": 1.5, "min": 1.0, "max": 1.99,
                        "group": "Regularization",
                        "description": "Tweedie distribution variance power. 1.0 = Poisson, 2.0 = Gamma. Regression only.",
                    },

                    # ── System ────────────────────────────────────────────────
                    {
                        "name": "num_threads", "type": "int", "default": 0, "min": 0, "max": 64,
                        "group": "System",
                        "description": "Number of parallel threads. 0 = use OpenMP default (all available cores).",
                    },
                ],
            },

            {
                "id": "random_forest",
                "name": "Random Forest",
                "description": "Ensemble of decision trees with bagging.",
                "hyperparameters": [
                    # ── Core ──────────────────────────────────────────────────
                    {
                        "name": "n_estimators", "type": "int", "default": 200, "min": 100, "max": 1000,
                        "group": "Core",
                        "description": "Number of trees in the forest. Recommended: 200–1000 for production, 100–400 for small datasets.",
                    },
                    {
                        "name": "criterion", "type": "select",
                        "default": "gini",
                        "options": [
                            {"label": "gini (Gini impurity)", "value": "gini"},
                            {"label": "entropy (Shannon information gain)", "value": "entropy"},
                            {"label": "log_loss (Shannon information gain)", "value": "log_loss"},
                        ],
                        "group": "Core",
                        "description": "Function to measure the quality of a split.",
                    },
                    {
                        "name": "max_depth", "type": "int", "default": 10, "min": -1, "max": 40,
                        "group": "Core",
                        "description": "Maximum tree depth. Recommended: 5–40. Deep trees (20+) risk overfitting. -1 = no limit.",
                    },
                    {
                        "name": "max_features", "type": "select",
                        "default": "sqrt",
                        "options": [
                            {"label": "sqrt (default — √n_features)", "value": "sqrt"},
                            {"label": "log2 (log₂ n_features)", "value": "log2"},
                            {"label": "None (all features)", "value": "None"},
                        ],
                        "group": "Core",
                        "description": "Number of features to consider at each split. 'sqrt' is recommended.",
                    },
                    {
                        "name": "warm_start", "type": "select",
                        "default": "False",
                        "options": [
                            {"label": "False (fit fresh each time)", "value": "False"},
                            {"label": "True (reuse previous fit, add more trees)", "value": "True"},
                        ],
                        "group": "Core",
                        "description": "If True, reuses the previous fit and adds more estimators to the ensemble.",
                    },

                    # ── Leaf / Split ──────────────────────────────────────────
                    {
                        "name": "min_samples_split", "type": "int", "default": 2, "min": 2, "max": 20,
                        "group": "Leaf / Split",
                        "description": "Minimum samples required to split a node. Recommended: 2–20.",
                    },
                    {
                        "name": "min_samples_leaf", "type": "int", "default": 1, "min": 1, "max": 10,
                        "group": "Leaf / Split",
                        "description": "Minimum samples required at a leaf node. Recommended: 1–10. Higher = smoother, less overfitting.",
                    },
                    {
                        "name": "min_weight_fraction_leaf", "type": "float", "default": 0.0, "min": 0.0, "max": 0.5,
                        "group": "Leaf / Split",
                        "description": "Minimum weighted fraction of total sample weights required at a leaf node.",
                    },
                    {
                        "name": "max_leaf_nodes", "type": "int", "default": -1, "min": -1, "max": 200,
                        "group": "Leaf / Split",
                        "description": "Limit number of leaf nodes. Recommended range: 10–200. -1 = unlimited.",
                    },
                    {
                        "name": "min_impurity_decrease", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
                        "group": "Leaf / Split",
                        "description": "A node is split only if it decreases impurity by at least this value.",
                    },
                    {
                        "name": "ccp_alpha", "type": "float", "default": 0.0, "min": 0.0, "max": 1.0,
                        "group": "Leaf / Split",
                        "description": "Complexity parameter for Minimal Cost-Complexity Pruning. 0 = no pruning.",
                    },

                    # ── Bootstrap / OOB ───────────────────────────────────────
                    {
                        "name": "bootstrap", "type": "select",
                        "default": "True",
                        "options": [
                            {"label": "True (use bootstrap samples)", "value": "True"},
                            {"label": "False (use whole dataset per tree)", "value": "False"},
                        ],
                        "group": "Bootstrap / OOB",
                        "description": "Whether to bootstrap samples when building trees.",
                    },
                    {
                        "name": "oob_score", "type": "select",
                        "default": "False",
                        "options": [
                            {"label": "False (no out-of-bag scoring)", "value": "False"},
                            {"label": "True (estimate generalization via OOB)", "value": "True"},
                        ],
                        "group": "Bootstrap / OOB",
                        "description": "Use out-of-bag samples to estimate generalization accuracy. Requires bootstrap=True.",
                    },
                    {
                        "name": "max_samples", "type": "int", "default": -1, "min": -1, "max": 10000,
                        "group": "Bootstrap / OOB",
                        "description": "Number of samples to draw per tree when bootstrap=True. -1 = use all samples.",
                    },

                    # ── System ────────────────────────────────────────────────
                    {
                        "name": "n_jobs", "type": "int", "default": -1, "min": -1, "max": 64,
                        "group": "System",
                        "description": "Number of parallel jobs. -1 = use all processors. None = 1.",
                    },
                    {
                        "name": "verbose", "type": "int", "default": 0, "min": 0, "max": 3,
                        "group": "System",
                        "description": "Controls verbosity when fitting and predicting.",
                    },
                    {
                        "name": "class_weight", "type": "select",
                        "default": "balanced",
                        "options": [
                            {"label": "balanced (auto-weight by class frequency)", "value": "balanced"},
                            {"label": "balanced_subsample (per-tree balanced)", "value": "balanced_subsample"},
                            {"label": "None (all classes equal weight)", "value": "None"},
                        ],
                        "group": "System",
                        "description": "Class weights. 'balanced' adjusts weights inversely proportional to class frequency.",
                    },
                ],
            },

            {
                "id": "logistic_regression",
                "name": "Logistic Regression",
                "description": "Linear model for binary classification with probabilistic outputs.",
                "hyperparameters": [
                    # ── Regularization ────────────────────────────────────────
                    {
                        "name": "penalty", "type": "select",
                        "default": "l2",
                        "options": [
                            {"label": "l2 (Ridge — default, works with all solvers)", "value": "l2"},
                            {"label": "l1 (Lasso — requires liblinear or saga)", "value": "l1"},
                            {"label": "elasticnet (L1+L2 mix — requires saga)", "value": "elasticnet"},
                            {"label": "None (no regularization)", "value": "None"},
                        ],
                        "group": "Regularization",
                        "description": "Regularization type. l2 prevents large weights (Ridge). l1 produces sparse weights (Lasso). elasticnet is a mix. None = no regularization.",
                    },
                    {
                        "name": "C", "type": "float", "default": 1.0, "min": 0.001, "max": 100.0,
                        "group": "Regularization",
                        "description": "Inverse of regularization strength. Smaller C = stronger regularization. Larger C = weaker regularization. Typical tuning range: 0.001–100.",
                    },
                    {
                        "name": "l1_ratio", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0,
                        "group": "Regularization",
                        "description": "ElasticNet mixing parameter. Only used when penalty='elasticnet' and solver='saga'. 0 = L2 only, 1 = L1 only, 0.5 = equal mix.",
                    },

                    # ── Solver ────────────────────────────────────────────────
                    {
                        "name": "solver", "type": "select",
                        "default": "lbfgs",
                        "options": [
                            {"label": "lbfgs (default — good for small/medium datasets, l2/None)", "value": "lbfgs"},
                            {"label": "liblinear (small datasets, supports l1 & l2)", "value": "liblinear"},
                            {"label": "newton-cg (large datasets, l2/None only)", "value": "newton-cg"},
                            {"label": "sag (large datasets, l2/None, fast)", "value": "sag"},
                            {"label": "saga (large datasets, all penalties, elasticnet)", "value": "saga"},
                        ],
                        "group": "Solver",
                        "description": "Optimization algorithm. liblinear = small datasets. lbfgs/saga = large datasets. saga is the only solver supporting elasticnet.",
                    },
                    {
                        "name": "max_iter", "type": "int", "default": 100, "min": 50, "max": 2000,
                        "group": "Solver",
                        "description": "Maximum number of iterations for the solver to converge. Increase to 200–1000 if convergence warnings appear.",
                    },
                    {
                        "name": "tol", "type": "float", "default": 1e-4, "min": 1e-6, "max": 1e-2,
                        "group": "Solver",
                        "description": "Stopping tolerance for the optimizer. Lower = more precise but slower training. Default 1e-4 works well for most cases.",
                    },
                    {
                        "name": "dual", "type": "select",
                        "default": "False",
                        "options": [
                            {"label": "False (primal formulation — default)", "value": "False"},
                            {"label": "True (dual formulation — only liblinear + l2)", "value": "True"},
                        ],
                        "group": "Solver",
                        "description": "Dual or primal formulation. Only applies when solver='liblinear' and penalty='l2'. Prefer dual=False when n_samples > n_features.",
                    },

                    # ── Intercept ─────────────────────────────────────────────
                    {
                        "name": "fit_intercept", "type": "select",
                        "default": "True",
                        "options": [
                            {"label": "True (include intercept/bias term — default)", "value": "True"},
                            {"label": "False (no intercept — use if data is already centered)", "value": "False"},
                        ],
                        "group": "Intercept",
                        "description": "Whether to include an intercept (bias) term in the model. Set False only if data is pre-centered.",
                    },
                    {
                        "name": "intercept_scaling", "type": "float", "default": 1.0, "min": 0.1, "max": 10.0,
                        "group": "Intercept",
                        "description": "Scaling factor for the synthetic intercept feature. Only used when solver='liblinear' and fit_intercept=True.",
                    },

                    # ── Class Weighting ───────────────────────────────────────
                    {
                        "name": "class_weight", "type": "select",
                        "default": "balanced",
                        "options": [
                            {"label": "balanced (auto-weight inversely by class frequency)", "value": "balanced"},
                            {"label": "None (all classes equal weight)", "value": "None"},
                        ],
                        "group": "Class Weighting",
                        "description": "Handles class imbalance. 'balanced' adjusts weights inversely to class frequency — recommended for fraud detection.",
                    },

                    # ── System ────────────────────────────────────────────────
                    {
                        "name": "random_state", "type": "int", "default": 42, "min": 0, "max": 99999,
                        "group": "System",
                        "description": "Controls randomness for solvers that use it (sag, saga, liblinear). Set for reproducibility.",
                    },
                    {
                        "name": "n_jobs", "type": "int", "default": -1, "min": -1, "max": 64,
                        "group": "System",
                        "description": "CPU cores for parallel computation over classes (only for multi-class ovr). -1 = use all cores.",
                    },
                    {
                        "name": "verbose", "type": "int", "default": 0, "min": 0, "max": 2,
                        "group": "System",
                        "description": "Verbosity level for liblinear and lbfgs solvers. 0 = silent.",
                    },
                    {
                        "name": "warm_start", "type": "select",
                        "default": "False",
                        "options": [
                            {"label": "False (fit from scratch each time — default)", "value": "False"},
                            {"label": "True (reuse previous solution as starting point)", "value": "True"},
                        ],
                        "group": "System",
                        "description": "Reuse previous solution as initial fit. Useful for incremental training or tuning n_iter.",
                    },
                ],
            },

            {
                "id": "decision_tree",
                "name": "Decision Tree",
                "description": "Single decision tree classifier with interpretable rules.",
                "hyperparameters": [
                    {
                        "name": "criterion", "type": "select",
                        "default": "gini",
                        "options": [
                            {"label": "gini (Gini impurity)", "value": "gini"},
                            {"label": "entropy (Information gain)", "value": "entropy"},
                            {"label": "log_loss (same as entropy)", "value": "log_loss"},
                        ],
                        "group": "Tree Structure",
                        "description": "Function used to measure split quality.",
                    },
                    {
                        "name": "splitter", "type": "select",
                        "default": "best",
                        "options": [
                            {"label": "best (chooses the best split)", "value": "best"},
                            {"label": "random (chooses the best random split)", "value": "random"},
                        ],
                        "group": "Tree Structure",
                        "description": "Strategy used to choose split.",
                    },
                    {
                        "name": "max_depth", "type": "int", "default": 10, "min": 3, "max": 50,
                        "group": "Tree Size",
                        "description": "Maximum depth of the tree. None = grow until pure.",
                    },
                    {
                        "name": "min_samples_split", "type": "float", "default": 2, "min": 2, "max": 20,
                        "group": "Tree Size",
                        "description": "Minimum samples required to split a node (int -> number, float -> percentage).",
                    },
                    {
                        "name": "min_samples_leaf", "type": "float", "default": 1, "min": 1, "max": 10,
                        "group": "Tree Size",
                        "description": "Minimum samples required at leaf node.",
                    },
                    {
                        "name": "min_weight_fraction_leaf", "type": "float", "default": 0.0, "min": 0.0, "max": 0.5,
                        "group": "Tree Size",
                        "description": "Minimum weighted fraction required at leaf.",
                    },
                    {
                        "name": "max_features", "type": "select",
                        "default": "None",
                        "options": [
                            {"label": "None (use all features)", "value": "None"},
                            {"label": "sqrt (square root of features)", "value": "sqrt"},
                            {"label": "log2 (log2 of features)", "value": "log2"},
                        ],
                        "group": "Feature Selection",
                        "description": "Number of features considered for split.",
                    },
                    {
                        "name": "random_state", "type": "int", "default": 42, "min": 0, "max": 99999,
                        "group": "System",
                        "description": "Controls randomness.",
                    },
                    {
                        "name": "max_leaf_nodes", "type": "int", "default": 20, "min": 2, "max": 500,
                        "group": "Tree Size",
                        "description": "Maximum number of leaf nodes. None = unlimited.",
                    },
                    {
                        "name": "min_impurity_decrease", "type": "float", "default": 0.0, "min": 0.0, "max": 0.1,
                        "group": "Tree Split Requirement",
                        "description": "Node will split only if impurity decreases more than this value.",
                    },
                    {
                        "name": "class_weight", "type": "select",
                        "default": "balanced",
                        "options": [
                            {"label": "balanced (auto-weight inversely by class frequency)", "value": "balanced"},
                            {"label": "None (all classes equal weight)", "value": "None"},
                        ],
                        "group": "Class Weighting",
                        "description": "Handles class imbalance.",
                    },
                ],
            },
            {
                "id": "svm",
                "name": "Support Vector Machine",
                "description": "SVM classifier with kernel trick for non-linear boundaries. Note: High training time on large datasets.",
                "hyperparameters": [
                    {
                        "name": "C", "type": "float", "default": 1.0, "min": 0.001, "max": 1000.0,
                        "group": "Regularization",
                        "description": "Regularization parameter. Small=more regularization, Large=less.",
                    },
                    {
                        "name": "kernel", "type": "select",
                        "default": "rbf",
                        "options": [
                            {"label": "rbf (Radial Basis Function)", "value": "rbf"},
                            {"label": "linear", "value": "linear"},
                            {"label": "poly (Polynomial)", "value": "poly"},
                            {"label": "sigmoid", "value": "sigmoid"},
                            {"label": "precomputed", "value": "precomputed"},
                        ],
                        "group": "Kernel",
                        "description": "Kernel type used for transformation.",
                    },
                    {
                        "name": "degree", "type": "int", "default": 3, "min": 2, "max": 5,
                        "group": "Kernel",
                        "description": "Polynomial degree. Used only when kernel='poly'.",
                    },
                    {
                        "name": "gamma", "type": "select",
                        "default": "scale",
                        "options": [
                            {"label": "scale", "value": "scale"},
                            {"label": "auto", "value": "auto"},
                        ],
                        "group": "Kernel",
                        "description": "Kernel coefficient. Can also be a float value.",
                    },
                    {
                        "name": "coef0", "type": "float", "default": 0.0, "min": 0.0, "max": 10.0,
                        "group": "Kernel",
                        "description": "Independent term used in 'poly' and 'sigmoid' kernels.",
                    },
                    {
                        "name": "shrinking", "type": "select",
                        "default": "True",
                        "options": [
                            {"label": "True", "value": "True"},
                            {"label": "False", "value": "False"},
                        ],
                        "group": "Optimization",
                        "description": "Whether to use the shrinking heuristic.",
                    },
                    {
                        "name": "probability", "type": "select",
                        "default": "True",
                        "options": [
                            {"label": "True", "value": "True"},
                            {"label": "False", "value": "False"},
                        ],
                        "group": "Output",
                        "description": "Enables probability estimation. Setting False breaks ROC-AUC evaluation.",
                    },
                    {
                        "name": "tol", "type": "float", "default": 1e-3, "min": 1e-6, "max": 1e-1,
                        "group": "Optimization",
                        "description": "Tolerance for stopping criterion.",
                    },
                    {
                        "name": "cache_size", "type": "int", "default": 200, "min": 100, "max": 2000,
                        "group": "System",
                        "description": "Kernel cache size in MB.",
                    },
                    {
                        "name": "class_weight", "type": "select",
                        "default": "balanced",
                        "options": [
                            {"label": "balanced", "value": "balanced"},
                            {"label": "None", "value": "None"},
                        ],
                        "group": "Class Weighting",
                        "description": "Used for imbalanced datasets.",
                    },
                    {
                        "name": "verbose", "type": "select",
                        "default": "False",
                        "options": [
                            {"label": "False", "value": "False"},
                            {"label": "True", "value": "True"},
                        ],
                        "group": "System",
                        "description": "Enable training logs.",
                    },
                    {
                        "name": "max_iter", "type": "int", "default": -1, "min": -1, "max": 10000,
                        "group": "Optimization",
                        "description": "Maximum iterations. -1 = no limit.",
                    },
                    {
                        "name": "decision_function_shape", "type": "select",
                        "default": "ovr",
                        "options": [
                            {"label": "ovr (One vs Rest)", "value": "ovr"},
                            {"label": "ovo (One vs One)", "value": "ovo"},
                        ],
                        "group": "Optimization",
                        "description": "Multi-class strategy.",
                    },
                    {
                        "name": "break_ties", "type": "select",
                        "default": "False",
                        "options": [
                            {"label": "False", "value": "False"},
                            {"label": "True", "value": "True"},
                        ],
                        "group": "Optimization",
                        "description": "Break ties when decision_function_shape='ovr' and classes > 2.",
                    },
                    {
                        "name": "random_state", "type": "int", "default": 42, "min": 0, "max": 99999,
                        "group": "System",
                        "description": "Controls randomness (used for probability=True).",
                    },
                ],
            },
            {
                "id": "knn",
                "name": "K-Nearest Neighbors",
                "description": "Instance-based learning using k nearest neighbors for classification.",
                "hyperparameters": [
                    {
                        "name": "n_neighbors", "type": "int", "default": 5, "min": 1, "max": 50,
                        "group": "Model Complexity",
                        "description": "Number of nearest neighbors used for prediction.",
                    },
                    {
                        "name": "weights", "type": "select",
                        "default": "uniform",
                        "options": [
                            {"label": "uniform (all neighbors equal weight)", "value": "uniform"},
                            {"label": "distance (closer neighbors get higher weight)", "value": "distance"},
                        ],
                        "group": "Model Complexity",
                        "description": "Weight function used for prediction.",
                    },
                    {
                        "name": "algorithm", "type": "select",
                        "default": "auto",
                        "options": [
                            {"label": "auto", "value": "auto"},
                            {"label": "ball_tree", "value": "ball_tree"},
                            {"label": "kd_tree", "value": "kd_tree"},
                            {"label": "brute", "value": "brute"},
                        ],
                        "group": "Algorithm",
                        "description": "Algorithm used to compute nearest neighbors.",
                    },
                    {
                        "name": "leaf_size", "type": "int", "default": 30, "min": 10, "max": 100,
                        "group": "Algorithm",
                        "description": "Leaf size passed to BallTree or KDTree.",
                    },
                    {
                        "name": "p", "type": "int", "default": 2, "min": 1, "max": 5,
                        "group": "Distance Metric",
                        "description": "Power parameter for Minkowski metric (1=Manhattan, 2=Euclidean).",
                    },
                    {
                        "name": "metric", "type": "select",
                        "default": "minkowski",
                        "options": [
                            {"label": "minkowski", "value": "minkowski"},
                            {"label": "euclidean", "value": "euclidean"},
                            {"label": "manhattan", "value": "manhattan"},
                            {"label": "chebyshev", "value": "chebyshev"},
                            {"label": "hamming", "value": "hamming"},
                        ],
                        "group": "Distance Metric",
                        "description": "Distance metric to use.",
                    },
                    {
                        "name": "metric_params", "type": "select",
                        "default": "None",
                        "options": [
                            {"label": "None", "value": "None"}
                        ],
                        "group": "Distance Metric",
                        "description": "Additional parameters for distance metric (rarely used).",
                    },
                    {
                        "name": "n_jobs", "type": "int", "default": -1, "min": -1, "max": 64,
                        "group": "System",
                        "description": "Cores to use for neighbor search (-1 = all).",
                    },
                ],
            },
            {
                "id": "naive_bayes",
                "name": "Naive Bayes",
                "description": "Probabilistic classifier based on Bayes theorem with feature independence.",
                "hyperparameters": [
                    {"name": "var_smoothing", "type": "float", "default": 1e-9, "min": 1e-10, "max": 1e-7},
                ],
            },
            {
                "id": "neural_network",
                "name": "Neural Network (MLP)",
                "description": "Multi-Layer Perceptron classifier. Good for complex patterns but requires tuning and scaling.",
                "hyperparameters": [
                    {
                        "name": "hidden_layer_sizes", "type": "int", "default": 100, "min": 1, "max": 2000,
                        "group": "Architecture",
                        "description": "Number of neurons in the single hidden layer (for MLP simple representation).",
                    },
                    {
                        "name": "activation", "type": "select",
                        "default": "relu",
                        "options": [
                            {"label": "relu (Rectified Linear Unit)", "value": "relu"},
                            {"label": "tanh (Hyperbolic Tangent)", "value": "tanh"},
                            {"label": "logistic (Sigmoid)", "value": "logistic"},
                            {"label": "identity", "value": "identity"},
                        ],
                        "group": "Architecture",
                        "description": "Activation function used in hidden layers.",
                    },
                    {
                        "name": "solver", "type": "select",
                        "default": "adam",
                        "options": [
                            {"label": "adam (Default, good for large datasets)", "value": "adam"},
                            {"label": "sgd (Stochastic Gradient Descent)", "value": "sgd"},
                            {"label": "lbfgs (Good for small datasets)", "value": "lbfgs"},
                        ],
                        "group": "Training",
                        "description": "Optimization algorithm used for weight updates.",
                    },
                    {
                        "name": "max_iter", "type": "int", "default": 200, "min": 10, "max": 5000,
                        "group": "Training",
                        "description": "Maximum number of epochs (full training passes).",
                    },
                    {
                        "name": "batch_size", "type": "select",
                        "default": "auto",
                        "options": [
                            {"label": "auto (min(200, n_samples))", "value": "auto"},
                            {"label": "32", "value": "32"},
                            {"label": "64", "value": "64"},
                            {"label": "128", "value": "128"},
                            {"label": "256", "value": "256"},
                            {"label": "512", "value": "512"},
                        ],
                        "group": "Training",
                        "description": "Samples processed per iteration. Only applies to adam or sgd.",
                    },
                    {
                        "name": "learning_rate_init", "type": "float", "default": 0.001, "min": 1e-6, "max": 1.0,
                        "group": "Training",
                        "description": "Initial learning rate (step size for weight updates).",
                    },
                    {
                        "name": "shuffle", "type": "select",
                        "default": "True",
                        "options": [
                            {"label": "True", "value": "True"},
                            {"label": "False", "value": "False"},
                        ],
                        "group": "Training",
                        "description": "Shuffle samples in each iteration. Only used when solver='sgd' or 'adam'.",
                    },
                    {
                        "name": "alpha", "type": "float", "default": 0.0001, "min": 0.0, "max": 10.0,
                        "group": "Regularization",
                        "description": "L2 regularization term (weight decay).",
                    },
                    {
                        "name": "early_stopping", "type": "select",
                        "default": "False",
                        "options": [
                            {"label": "False", "value": "False"},
                            {"label": "True", "value": "True"},
                        ],
                        "group": "Regularization",
                        "description": "Stop training when validation score isn't improving.",
                    },
                    {
                        "name": "validation_fraction", "type": "float", "default": 0.1, "min": 0.01, "max": 0.5,
                        "group": "Regularization",
                        "description": "Portion of data set aside as early stopping validation. (Only if early_stopping=True).",
                    },
                    {
                        "name": "learning_rate", "type": "select",
                        "default": "constant",
                        "options": [
                            {"label": "constant", "value": "constant"},
                            {"label": "invscaling (inverse scaling)", "value": "invscaling"},
                            {"label": "adaptive", "value": "adaptive"},
                        ],
                        "group": "Learning Rate Control",
                        "description": "LR schedule. Only used when solver='sgd'.",
                    },
                    {
                        "name": "power_t", "type": "float", "default": 0.5, "min": 0.01, "max": 2.0,
                        "group": "Learning Rate Control",
                        "description": "Exponent for inverse scaling. Only used when solver='sgd' and learning_rate='invscaling'.",
                    },
                    {
                        "name": "momentum", "type": "float", "default": 0.9, "min": 0.0, "max": 1.0,
                        "group": "Advanced",
                        "description": "Momentum for gradient descent. Only used when solver='sgd'.",
                    },
                    {
                        "name": "beta_1", "type": "float", "default": 0.9, "min": 0.0, "max": 0.9999,
                        "group": "Advanced",
                        "description": "1st moment parameter. Only used when solver='adam'.",
                    },
                    {
                        "name": "beta_2", "type": "float", "default": 0.999, "min": 0.0, "max": 0.99999,
                        "group": "Advanced",
                        "description": "2nd moment parameter. Only used when solver='adam'.",
                    },
                    {
                        "name": "epsilon", "type": "float", "default": 1e-8, "min": 1e-10, "max": 1e-1,
                        "group": "Advanced",
                        "description": "Numerical stability parameter. Only used when solver='adam'.",
                    },
                    {
                        "name": "random_state", "type": "int", "default": 42, "min": 0, "max": 99999,
                        "group": "System",
                        "description": "Controls random weight initialization.",
                    },
                ],
            },
        ]
    
    async def get_default_hyperparameters(self, algorithm: str) -> Dict[str, Any]:
        """Get default hyperparameters for an algorithm."""
        algorithms = await self.list_algorithms()
        for algo in algorithms:
            if algo["id"] == algorithm:
                return {hp["name"]: hp["default"] for hp in algo["hyperparameters"]}
        return {}

    async def delete_training_job(self, job_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a training job."""
        try:
            uuid_id = UUID(job_id)
        except ValueError:
            return False
            
        stmt = select(TrainingJob).where(TrainingJob.id == uuid_id)
        safe_user_id = _safe_uuid_user_id(user_id, "delete_training_job")
        if safe_user_id:
            stmt = stmt.where(TrainingJob.created_by == safe_user_id)
            
        result = await self.db.execute(stmt)
        job = result.scalar_one_or_none()
        
        if not job:
            return False
            
        await self.db.delete(job)
        await self.db.commit()
        return True


class ModelService:
    """Service for model registry operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_models(
        self,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[str] = None,
    ) -> Tuple[List[MLModel], int]:
        """List models with pagination."""
        query = select(MLModel).order_by(MLModel.created_at.desc())
        count_query = select(func.count(MLModel.id))
        
        if status:
            query = query.where(MLModel.status == status)
            count_query = count_query.where(MLModel.status == status)
            
        safe_user_id = _safe_uuid_user_id(user_id, "list_models")
        if safe_user_id:
            query = query.where(MLModel.created_by == safe_user_id)
            count_query = count_query.where(MLModel.created_by == safe_user_id)
        
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)
        
        result = await self.db.execute(query)
        models = result.scalars().all()
        
        return list(models), total
    
    async def get_model(self, model_id: str, user_id: Optional[str] = None) -> Optional[MLModel]:
        """Get a single model by ID."""
        try:
            uuid_id = UUID(model_id)
        except ValueError:
            return None
        
        stmt = select(MLModel).where(MLModel.id == uuid_id)
        safe_user_id = _safe_uuid_user_id(user_id, "get_model")
        if safe_user_id:
            stmt = stmt.where(MLModel.created_by == safe_user_id)
            
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_production_model(self) -> Optional[MLModel]:
        """Get the current production model."""
        result = await self.db.execute(
            select(MLModel).where(MLModel.status == "PRODUCTION")
        )
        return result.scalar_one_or_none()
    
    async def promote_model(
        self,
        model_id: str,
        target_status: str,
        ) -> Optional[MLModel]:
        """Promote a model to a new status."""
        model = await self.get_model(model_id)
        if not model:
            return None
        
        # If promoting to PRODUCTION, demote current production
        if target_status == "PRODUCTION":
            current_prod = await self.get_production_model()
            if current_prod and str(current_prod.id) != model_id:
                current_prod.status = "ARCHIVED"
                current_prod.archived_at = now_ist()
                current_prod.archived_reason = "Replaced by new production model"
        
        model.status = target_status
        if target_status == "PRODUCTION":
            model.promoted_at = now_ist()
        
        await self.db.commit()
        await self.db.refresh(model)
        
        logger.info(f"Model {model_id} promoted to {target_status}")
        return model
    
    async def set_baselines(
        self,
        model_id: str,
        baselines: List[Dict],
    ) -> List[Baseline]:
        """Set baseline thresholds for a model."""
        model = await self.get_model(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        created_baselines = []
        for b in baselines:
            baseline = Baseline(
                model_id=UUID(model_id),
                metric_name=b["metric"],
                threshold=b["threshold"],
                operator=b.get("operator", "gte"),
            )
            self.db.add(baseline)
            created_baselines.append(baseline)
        
        await self.db.commit()
        return created_baselines
    
    async def delete_model(self, model_id: str, hard_delete: bool = True, user_id: Optional[str] = None) -> bool:
        """
        Delete a model from the registry.
        
        Args:
            model_id: Model ID to delete
            hard_delete: If True, also delete from blob storage
            user_id: The ID of the current user
            
        Returns:
            True if deleted, False if not found
        """
        model = await self.get_model(model_id, user_id=user_id)
        if not model:
            return False
        
        # Delete from blob storage if hard delete
        if hard_delete and model.storage_path:
            from app.core.storage import StorageService
            storage = StorageService()
            
            # Delete main model file
            try:
                await storage.delete_model(model.storage_path)
            except Exception as e:
                logger.warning(f"Failed to delete model storage: {e}")
            
            # Delete ONNX file if exists
            if model.onnx_path:
                try:
                    await storage.delete_model(model.onnx_path)
                except Exception as e:
                    logger.warning(f"Failed to delete ONNX storage: {e}")
        
        # Delete from database (cascade will delete baselines)
        await self.db.delete(model)
        await self.db.commit()
        
        logger.info(f"Deleted model {model_id}")
        return True
