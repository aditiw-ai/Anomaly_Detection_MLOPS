import asyncio
import sys
import os

# Add the parent directory to sys.path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import async_session_maker
from app.models.user import User
from app.core.security import get_password_hash
from sqlalchemy import select

async def main():
    email = "admin@example.com"
    password = "password123"
    name = "Admin User"
    
    async with async_session_maker() as session:
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalar_one_or_none()
        
        if user:
            print(f"User {email} already exists!")
            return
            
        print(f"Creating admin user {email}...")
        new_user = User(
            id="admin-12345",
            email=email,
            hashed_password=get_password_hash(password),
            name=name,
            roles=["ADMIN"],
            is_active=True
        )
        session.add(new_user)
        await session.commit()
        print(f"Admin user created successfully! Email: {email}, Password: {password}")

if __name__ == "__main__":
    asyncio.run(main())
