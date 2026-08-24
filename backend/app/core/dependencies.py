"""
Authentication Dependencies
FastAPI dependencies for protected routes.
"""
from typing import Optional, List
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.auth import (
    get_auth,
    User,
    Role,
    Permission,
    AzureADB2CAuth,
    normalize_roles,
)

# Bearer token security scheme
security = HTTPBearer(auto_error=False)


from app.core.security import ALGORITHM, SECRET_KEY
from app.core.database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User as DBUser
from jose import jwt, JWTError

def _to_role(role_raw: object) -> Optional[Role]:
    """Normalize role strings from DB/token into Role enum values."""
    normalized = normalize_roles(role_raw, default=None)
    return normalized[0] if normalized else None

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current authenticated user. First checks local JWT, falls back to Azure B2C.
    
    Returns None if no token provided (for public endpoints).
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    is_local_token = False
    try:
        is_local_token = jwt.get_unverified_header(token).get("alg") == ALGORITHM
        # Try local JWT decoding first
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id:
            result = await db.execute(select(DBUser).where(DBUser.id == user_id))
            db_user = result.scalar_one_or_none()
            if db_user and db_user.is_active:
                roles = normalize_roles(db_user.roles, default=Role.VIEWER)
                
                # Compute permissions
                from app.core.auth import ROLE_PERMISSIONS
                permissions = set()
                for role in roles:
                    permissions.update(ROLE_PERMISSIONS.get(role, []))
                
                return User(
                    id=db_user.id,
                    email=db_user.email,
                    name=db_user.name,
                    roles=roles,
                    permissions=list(permissions),
                    tenant_id="local",
                    issued_at=datetime.now(timezone.utc),
                    expires_at=datetime.now(timezone.utc) + timedelta(days=1),
                )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid local user",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except JWTError:
        if is_local_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid local token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        pass  # Not a valid local token, try B2C
    
    auth = get_auth()
    user = await auth.validate_token(token)
    
    return user


async def require_auth(
    user: User = Depends(get_current_user),
) -> User:
    """
    Require authentication.
    
    Raises 401 if not authenticated.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_permission(permission: Permission):
    """
    Factory for permission-checking dependencies.
    
    Usage:
        @router.get("/admin", dependencies=[Depends(require_permission(Permission.USERS_MANAGE))])
    """
    async def check_permission(user: User = Depends(require_auth)) -> User:
        auth = get_auth()
        if not auth.has_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: {permission.value}",
            )
        return user
    
    return check_permission


def require_role(role: Role):
    """
    Factory for role-checking dependencies.
    
    Usage:
        @router.get("/admin", dependencies=[Depends(require_role(Role.ADMIN))])
    """
    async def check_role(user: User = Depends(require_auth)) -> User:
        auth = get_auth()
        if not auth.has_role(user, role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role.value}",
            )
        return user
    
    return check_role


def require_any_role(roles: List[Role]):
    """
    Factory for checking if user has any of the specified roles.
    
    Usage:
        @router.get("/ml", dependencies=[Depends(require_any_role([Role.DATA_SCIENTIST, Role.ML_ENGINEER]))])
    """
    async def check_roles(user: User = Depends(require_auth)) -> User:
        auth = get_auth()
        if not any(auth.has_role(user, role) for role in roles):
            role_names = ", ".join(r.value for r in roles)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"One of these roles required: {role_names}",
            )
        return user
    
    return check_roles


# ── Convenience dependencies ──────────────────────────────────────────────────
require_admin        = require_role(Role.ADMIN)
require_data_write   = require_any_role([Role.ADMIN, Role.DATA_ENGINEER])
require_ml_engineer  = require_any_role([Role.ADMIN, Role.ML_ENGINEER])
require_deployer     = require_any_role([Role.ADMIN, Role.DEPLOYER])

# Permission-based dependencies
can_train_models       = require_permission(Permission.MODEL_TRAIN)
can_deploy_models      = require_permission(Permission.MODEL_DEPLOY)
can_run_inference      = require_permission(Permission.INFERENCE_RUN)
can_manage_jobs        = require_permission(Permission.JOBS_MANAGE)
can_configure_monitoring = require_permission(Permission.MONITORING_CONFIGURE)
can_manage_users       = require_permission(Permission.USERS_MANAGE)
