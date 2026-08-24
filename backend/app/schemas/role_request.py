from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict
import enum

class RequestStatus(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class RoleRequestBase(BaseModel):
    requested_role: str
    reason: Optional[str] = None

class RoleRequestCreate(RoleRequestBase):
    pass

class RoleRequestUpdate(BaseModel):
    status: RequestStatus
    admin_notes: Optional[str] = None

class RoleRequestResponse(RoleRequestBase):
    id: str
    user_id: str
    status: RequestStatus
    admin_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    # Extra fields for UI convenience
    user_email: Optional[str] = None
    user_name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
