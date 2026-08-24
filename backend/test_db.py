import asyncio
import asyncpg

async def test():
    try:
        conn = await asyncpg.connect('postgresql://admin123:Akshronix%40123@frauddetection.postgres.database.azure.com:5432/postgres?ssl=require')
        print('Connection successful')
        await conn.close()
    except Exception as e:
        print(f'Connection failed: {e}')

asyncio.run(test())
