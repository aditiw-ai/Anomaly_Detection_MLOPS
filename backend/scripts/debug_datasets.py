import os
import psycopg2
import json

def debug_db():
    url = os.environ.get("DATABASE_URL", "").replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("ssl=require", "sslmode=require")
    
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, row_count, storage_path, file_size_bytes FROM datasets ORDER BY created_at DESC LIMIT 5")
            rows = cur.fetchall()
            for r in rows:
                print(f"ID: {r[0]}")
                print(f"Name: {r[1]}")
                print(f"Row Count: {r[2]}")
                print(f"Storage Path: {r[3]}")
                print(f"File Size: {r[4]} bytes")
                print("-" * 40)
    finally:
        conn.close()

if __name__ == "__main__":
    debug_db()
