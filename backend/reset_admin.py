import asyncio
from sqlalchemy import select

from app.core.database import async_session_maker
from app.models.user import User
from app.core.security import get_password_hash


async def reset_admin():
    async with async_session_maker() as session:
        result = await session.execute(
            select(User).where(User.email == "admin@example.com")
        )

        user = result.scalar_one_or_none()

        if not user:
            print("User not found!")
            return

        user.roles = ["ADMIN"]
        user.hashed_password = get_password_hash("password123")

        await session.commit()

        print("ADMIN ACCOUNT RESET SUCCESSFULLY")
        print("Email: admin@example.com")
        print("Password: password123")
        print("Role: ADMIN")


if __name__ == "__main__":
    asyncio.run(reset_admin())