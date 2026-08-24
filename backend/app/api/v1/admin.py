"""
Admin API — user management (ADMIN role only).
"""
import uuid
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.core.database import get_db
from app.core.dependencies import can_manage_users
from app.core.security import get_password_hash
from app.models.user import User

router = APIRouter()

VALID_ROLES = {"ADMIN", "DATA_ENGINEER", "ML_ENGINEER", "DEPLOYER", "VIEWER"}


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    roles: List[str]
    is_active: bool

    class Config:
        from_attributes = True


class CreateUserRequest(BaseModel):
    name: str
    email: str
    password: str
    roles: List[str] = ["VIEWER"]


class RoleUpdateRequest(BaseModel):
    roles: List[str]


@router.post("/users", response_model=UserResponse, status_code=201, dependencies=[Depends(can_manage_users)])
async def create_user(body: CreateUserRequest, db: AsyncSession = Depends(get_db)):
    """Admin creates a new user with a specified role."""
    for role in body.roles:
        if role not in VALID_ROLES:
            raise HTTPException(status_code=400, detail=f"Invalid role: {role}")

    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        id=str(uuid.uuid4()),
        email=body.email,
        name=body.name,
        hashed_password=get_password_hash(body.password),
        roles=body.roles,
        is_active=True,
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user


@router.get("/users", response_model=List[UserResponse], dependencies=[Depends(can_manage_users)])
async def list_users(db: AsyncSession = Depends(get_db)):
    """List all users. ADMIN only."""
    result = await db.execute(select(User))
    users = result.scalars().all()
    return users


@router.patch("/users/{user_id}/role", response_model=UserResponse, dependencies=[Depends(can_manage_users)])
async def update_user_role(
    user_id: str,
    body: RoleUpdateRequest,
    db: AsyncSession = Depends(get_db)
):
    """Change a user's roles. ADMIN only."""
    valid_roles = {"ADMIN", "DATA_ENGINEER", "ML_ENGINEER", "DEPLOYER", "VIEWER"}
    for role in body.roles:
        if role not in valid_roles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role: {role}. Choose from: {', '.join(valid_roles)}"
            )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.roles = body.roles
    await db.commit()
    await db.refresh(user)
    return user


@router.patch("/users/{user_id}/status", response_model=UserResponse, dependencies=[Depends(can_manage_users)])
async def toggle_user_status(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Toggle a user's active/inactive status. ADMIN only."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = not user.is_active
    await db.commit()
    await db.refresh(user)
    return user


@router.delete("/users/{user_id}", status_code=204, dependencies=[Depends(can_manage_users)])
async def delete_user(user_id: str, db: AsyncSession = Depends(get_db)):
    """Delete a user. ADMIN only."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await db.delete(user)
    await db.commit()
