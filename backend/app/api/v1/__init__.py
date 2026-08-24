"""
API v1 Router
Aggregates all API endpoints for version 1.
"""
from fastapi import APIRouter

from app.api.v1.datasets import router as datasets_router
from app.api.v1.features import router as features_router
from app.api.v1.training import router as training_router
from app.api.v1.models import router as models_router
from app.api.v1.inference import router as inference_router
from app.api.v1.monitoring import router as monitoring_router
from app.api.v1.alerts import router as alerts_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.retraining import router as retraining_router
from app.api.v1.ab_testing import router as ab_testing_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.auth import router as auth_router
from app.api.v1.admin import router as admin_router
from app.api.v1.role_requests import router as role_requests_router

api_router = APIRouter()

# Include all routers
api_router.include_router(datasets_router)
api_router.include_router(features_router)
api_router.include_router(training_router)
api_router.include_router(models_router)
api_router.include_router(inference_router)
api_router.include_router(monitoring_router)
api_router.include_router(alerts_router)
api_router.include_router(jobs_router)
api_router.include_router(retraining_router)
api_router.include_router(ab_testing_router)
api_router.include_router(dashboard_router)
api_router.include_router(auth_router, prefix="/auth", tags=["auth"])
api_router.include_router(admin_router, prefix="/admin", tags=["admin"])
api_router.include_router(role_requests_router, prefix="/role-requests", tags=["role-requests"])



