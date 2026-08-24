from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

class UserProfile(BaseModel):
    id: str
    email: str
    name: str
    roles: List[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
