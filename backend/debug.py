import asyncio
import os
from app.core.database import async_session_maker
from app.models.user import User
from sqlalchemy import select
from app.core.security import verify_password

async def main():
    email = "admin@example.com"
    password = "password123"
    
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user:
            print("HASH IN DB:", user.hashed_password)
            is_valid = verify_password(password, user.hashed_password)
            print("IS VALID:", is_valid)
            print("IS ACTIVE:", user.is_active)
        else:
            print("USER NOT FOUND")

if __name__ == "__main__":
    asyncio.run(main())
