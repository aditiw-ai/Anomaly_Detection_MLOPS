from datetime import timedelta
import uuid
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel

from app.core.database import get_db
from app.core.security import verify_password, get_password_hash, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from app.models.user import User
from app.schemas.user import UserProfile
from app.core.dependencies import require_auth
from app.core.auth import User as AuthUser

router = APIRouter()
logger = logging.getLogger(__name__)


class UserCreate(BaseModel):
    email: str
    password: str
    name: str


@router.post("/signup", status_code=status.HTTP_201_CREATED)
async def signup(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user."""
    result = await db.execute(select(User).where(User.email == user_data.email))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    new_user = User(
        id=str(uuid.uuid4()),
        email=user_data.email,
        hashed_password=get_password_hash(user_data.password),
        name=user_data.name,
        roles=["VIEWER"],
        is_active=True
    )

    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(subject=new_user.id, expires_delta=access_token_expires)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": new_user.id,
        "name": new_user.name,
        "roles": new_user.roles
    }


@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
):
    def parse_roles(raw_roles) -> list[str]:
        if raw_roles is None:
            return ["VIEWER"]

        if isinstance(raw_roles, list):
            normalized = [str(role).strip().upper() for role in raw_roles if str(role).strip()]
            return normalized or ["VIEWER"]

        if isinstance(raw_roles, str):
            normalized_text = raw_roles.strip()
            if not normalized_text:
                return ["VIEWER"]

            try:
                parsed = json.loads(normalized_text)
                if isinstance(parsed, list):
                    normalized = [str(role).strip().upper() for role in parsed if str(role).strip()]
                    return normalized or ["VIEWER"]
                if isinstance(parsed, str):
                    normalized = parsed.strip().upper()
                    return [normalized] if normalized else ["VIEWER"]
            except json.JSONDecodeError:
                pass

            if normalized_text.startswith("{") and normalized_text.endswith("}"):
                normalized_text = normalized_text[1:-1]
            normalized = [part.strip().strip('"').upper() for part in normalized_text.split(",") if part.strip().strip('"')]
            return normalized or ["VIEWER"]

        return ["VIEWER"]

    try:
        email = (form_data.username or "").strip().lower()
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        query = text(
            """
            SELECT
                id::text AS id,
                email,
                hashed_password,
                name,
                COALESCE(to_json(roles)::text, '["VIEWER"]') AS roles_json,
                COALESCE(is_active, TRUE) AS is_active
            FROM users
            WHERE lower(email) = :email
            LIMIT 1
            """
        )
        result = await db.execute(query, {"email": email})
        user = result.mappings().first()

        if not user or not verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not user["is_active"]:
            raise HTTPException(status_code=400, detail="Inactive user")

        roles = parse_roles(user["roles_json"])
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(subject=user["id"], expires_delta=access_token_expires)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user["id"],
            "name": user["name"],
            "roles": roles
        }
    except HTTPException:
        raise
    except SQLAlchemyError:
        logger.exception("Database error during login for email=%s", form_data.username)
        raise HTTPException(status_code=500, detail="Authentication service unavailable")
    except Exception:
        logger.exception("Unexpected error during login for email=%s", form_data.username)
        raise HTTPException(status_code=500, detail="Login failed due to internal error")

@router.get("/me", response_model=UserProfile)
async def get_my_profile(
    current_user: AuthUser = Depends(require_auth),
    db: AsyncSession = Depends(get_db)
):
    """Fetch the full profile of the currently logged-in user."""
    result = await db.execute(select(User).where(User.id == current_user.id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
