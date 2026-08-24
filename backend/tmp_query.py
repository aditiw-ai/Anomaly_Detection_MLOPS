import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.core.database import async_session_maker
from app.models.user import User
from sqlalchemy import select

async def run():
    async with async_session_maker() as db:
        res = await db.execute(select(User))
        users = res.scalars().all()
        for u in users:
            print(f"User: {u.email}, Roles: {u.roles}")
            # If lowercase admin, fix it!
            new_roles = [r.upper() for r in u.roles]
            if new_roles != u.roles:
                print(f"Fixing roles for {u.email} to {new_roles}")
                u.roles = new_roles
        await db.commit()

if __name__ == "__main__":
    asyncio.run(run())
