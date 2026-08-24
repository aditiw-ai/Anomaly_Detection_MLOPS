import os
import psycopg2

def check_db():
    url = os.environ.get("DATABASE_URL", "").replace("+asyncpg", "").replace("ssl=require", "sslmode=require")
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    cur.execute("SELECT id, name, row_count, storage_path, file_size_bytes FROM datasets ORDER BY created_at DESC LIMIT 1")
    row = cur.fetchone()
    print(f"ID: {row[0]}")
    print(f"Name: {row[1]}")
    print(f"Row count: {row[2]}")
    print(f"Storage path: {row[3]}")
    print(f"File size: {row[4]}")
    conn.close()

if __name__ == "__main__":
    check_db()
