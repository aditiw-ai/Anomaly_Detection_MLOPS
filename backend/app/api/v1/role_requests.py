from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from typing import List, Optional

from app.core.database import get_db
from app.core.dependencies import get_current_user, require_admin
from app.models.user import User
from app.models.role_request import RoleRequest, RequestStatus
from app.schemas.role_request import RoleRequestCreate, RoleRequestResponse, RoleRequestUpdate
from app.core.auth import User as AuthUser

router = APIRouter()

VALID_ROLES = {"ADMIN", "DATA_ENGINEER", "ML_ENGINEER", "DEPLOYER", "VIEWER"}

@router.post("/", response_model=RoleRequestResponse)
async def create_request(
    body: RoleRequestCreate,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    if body.requested_role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {body.requested_role}")
        
    new_request = RoleRequest(
        id=str(uuid4()),
        user_id=current_user.id,
        requested_role=body.requested_role,
        reason=body.reason,
        status=RequestStatus.PENDING
    )
    db.add(new_request)
    await db.commit()
    await db.refresh(new_request)
    return new_request

@router.get("/me", response_model=List[RoleRequestResponse])
async def list_my_requests(
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    result = await db.execute(
        select(RoleRequest)
        .where(RoleRequest.user_id == current_user.id)
        .order_by(RoleRequest.created_at.desc())
    )
    return result.scalars().all()

@router.get("/admin/list", response_model=List[RoleRequestResponse], dependencies=[Depends(require_admin)])
async def list_all_requests(
    db: AsyncSession = Depends(get_db)
):
    # Join with User to get email/name
    result = await db.execute(
        select(RoleRequest, User.email, User.name)
        .join(User, RoleRequest.user_id == User.id)
        .order_by(RoleRequest.created_at.desc())
    )
    
    requests = []
    for row in result.all():
        req = row[0]
        req.user_email = row[1]
        req.user_name = row[2]
        requests.append(req)
        
    return requests

@router.patch("/admin/{request_id}/approve", response_model=RoleRequestResponse, dependencies=[Depends(require_admin)])
async def approve_request(
    request_id: str,
    admin_notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(RoleRequest).where(RoleRequest.id == request_id))
    req = result.scalar_one_or_none()
    
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
        
    if req.status != RequestStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Request is already {req.status}")
        
    # Update request status
    req.status = RequestStatus.APPROVED
    req.admin_notes = admin_notes
    
    # Update user role
    await db.execute(
        update(User)
        .where(User.id == req.user_id)
        .values(roles=[req.requested_role])
    )
    
    await db.commit()
    await db.refresh(req)
    return req

@router.patch("/admin/{request_id}/reject", response_model=RoleRequestResponse, dependencies=[Depends(require_admin)])
async def reject_request(
    request_id: str,
    admin_notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(RoleRequest).where(RoleRequest.id == request_id))
    req = result.scalar_one_or_none()
    
    if not req:
        raise HTTPException(status_code=404, detail="Request not found")
        
    if req.status != RequestStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Request is already {req.status}")
        
    req.status = RequestStatus.REJECTED
    req.admin_notes = admin_notes
    
    await db.commit()
    await db.refresh(req)
    return req
