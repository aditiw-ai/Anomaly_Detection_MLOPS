from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.classification_model import ClassificationModel


# Canonical list of classification models
SEED_CLASSIFICATION_MODELS = [
    {
        "algorithm_id": "xgboost",
        "name": "XGBoost",
        "description": "Extreme Gradient Boosting - highly efficient and flexible gradient boosting framework",
        "model_type": "supervised",
        "hyperparameters_schema": [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000},
            {"name": "max_depth", "type": "int", "default": 6, "min": 1, "max": 20},
            {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.01, "max": 1.0},
            {"name": "subsample", "type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
            {"name": "colsample_bytree", "type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
        ],
    },
    {
        "algorithm_id": "lightgbm",
        "name": "LightGBM",
        "description": "Light Gradient Boosting Machine - fast, distributed, high-performance gradient boosting framework",
        "model_type": "supervised",
        "hyperparameters_schema": [
            {"name": "n_estimators", "type": "int", "default": 100, "min": 10, "max": 1000},
            {"name": "max_depth", "type": "int", "default": -1, "min": -1, "max": 20},
            {"name": "learning_rate", "type": "float", "default": 0.1, "min": 0.01, "max": 1.0},
            {"name": "num_leaves", "type": "int", "default": 31, "min": 2, "max": 256},
            {"name": "min_child_samples", "type": "int", "default": 20, "min": 1, "max": 100},
        ],
    },
    {
        "algorithm_id": "random_forest",
        "name": "Random Forest",
        "description": "Ensemble learning method using multiple decision trees",
        "model_type": "supervised",
        "hyperparameters_schema": [
            {"name": "n_estimators",    "type": "int",   "default": 200, "min": 100, "max": 1000},
            {"name": "max_depth",       "type": "int",   "default": 10,  "min": -1,  "max": 40},
            {"name": "min_samples_split","type": "int",  "default": 2,   "min": 2,   "max": 20},
            {"name": "min_samples_leaf", "type": "int",  "default": 1,   "min": 1,   "max": 10},
            {"name": "max_leaf_nodes",   "type": "int",  "default": -1,  "min": -1,  "max": 200},
        ],
    },
]


class ClassificationModelService:
    """Service for managing classification model types (algorithms)."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_classification_models(
        self,
        active_only: bool = True,
        model_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[ClassificationModel], int]:
        """List all classification model types with optional filtering."""
        query = select(ClassificationModel)
        
        # Apply filters
        filters = []
        if active_only:
            filters.append(ClassificationModel.is_active == True)
        if model_type:
            filters.append(ClassificationModel.model_type == model_type)
        
        if filters:
            query = query.where(and_(*filters))
        
        # Get total count
        count_query = select(func.count(ClassificationModel.id))
        if filters:
            count_query = count_query.where(and_(*filters))
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Apply pagination
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size).order_by(ClassificationModel.name)
        
        result = await self.db.execute(query)
        rows = result.scalars().all()
        
        return list(rows), total
    
    async def get_by_algorithm_id(self, algorithm_id: str) -> Optional[ClassificationModel]:
        """Get a classification model by its algorithm_id."""
        result = await self.db.execute(
            select(ClassificationModel).where(ClassificationModel.algorithm_id == algorithm_id)
        )
        return result.scalar_one_or_none()
    
    async def seed_from_registry(self) -> int:
        """
        Seed the classification_models table from the canonical list.
        Only inserts if the algorithm_id doesn't already exist.
        Returns the number of new rows inserted.
        """
        inserted = 0
        for model_data in SEED_CLASSIFICATION_MODELS:
            existing = await self.get_by_algorithm_id(model_data["algorithm_id"])
            if not existing:
                model = ClassificationModel(**model_data)
                self.db.add(model)
                inserted += 1
        
        await self.db.flush()
        return inserted
