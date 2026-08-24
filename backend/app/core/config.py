"""
Application Configuration
Uses Pydantic Settings for environment variable management.
"""
from functools import lru_cache
from pathlib import Path
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_ENV_PATH = _BACKEND_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Application
    APP_NAME: str = "Shadow Hubble"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://localhost:5432/shadowhubble"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 5
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_FEATURE_TTL: int = 86400  # 24 hours
    
    # Azure Blob Storage
    # Azure Blob Storage - All 8 Containers
    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_STORAGE_CONTAINER_DATASETS: str = "datasets"
    AZURE_STORAGE_CONTAINER_MODELS: str = "models"
    AZURE_STORAGE_CONTAINER_FEATURES: str = "features"
    AZURE_STORAGE_CONTAINER_MONITORING: str = "monitoring"
    AZURE_STORAGE_CONTAINER_AUDIT_LOGS: str = "audit-logs"
    AZURE_STORAGE_CONTAINER_EXPERIMENTS: str = "experiments"
    AZURE_STORAGE_CONTAINER_BACKUPS: str = "backups"
    AZURE_STORAGE_CONTAINER_TEMP_PROCESSING: str = "temp-processing"
    
    # Azure AD B2C
    AZURE_AD_B2C_TENANT: str = ""
    AZURE_AD_B2C_CLIENT_ID: str = ""
    AZURE_AD_B2C_POLICY: str = "B2C_1_signupsignin"
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    
    # ML Settings
    MAX_FEATURES: int = 30
    DEFAULT_TRAIN_TEST_SPLIT: float = 0.8
    PSI_WARNING_THRESHOLD: float = 0.1
    PSI_CRITICAL_THRESHOLD: float = 0.25
    
    # Security
    SECRET_KEY: str  # required — no default; startup fails if not set in environment
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Notifications
    SLACK_WEBHOOK_URL: str | None = None

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_value(cls, value):
        """Accept common deployment-style DEBUG values like 'release'/'production'."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production"}:
                return False
            if normalized in {"debug", "dev", "development"}:
                return True
        return value


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
