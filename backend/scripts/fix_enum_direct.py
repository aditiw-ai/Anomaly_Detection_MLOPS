import os
import psycopg2

def migrate():
    # Convert asyncpg to psycopg2 style URL
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("❌ DATABASE_URL not set")
        return
        
    # Replace asyncpg driver and SSL parameters
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    url = url.replace("ssl=require", "sslmode=require")
    
    try:
        print(f"🚀 Connecting to database to update 'alerttype' enum...")
        conn = psycopg2.connect(url)
        conn.autocommit = True
        with conn.cursor() as cur:
            try:
                cur.execute("ALTER TYPE alerttype ADD VALUE 'TRAINING'")
                print("✅ Successfully added 'TRAINING' to alerttype enum.")
            except psycopg2.errors.DuplicateObject:
                print("ℹ️  'TRAINING' already exists in alerttype enum.")
            except Exception as e:
                print(f"❌ SQL Execution failed: {e}")
        conn.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    migrate()
