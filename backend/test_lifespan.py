import asyncio
import os
import sys

# Add backend to PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from app.main import app

def run_test():
    # Using 'with' triggers the lifespan context (startup/shutdown events)
    try:
        with TestClient(app) as client:
            print("Lifespan started successfully!")
            response = client.post("/api/v1/auth/login", data={"username": "admin@example.com", "password": "password123"})
            print("STATUS", response.status_code)
    except Exception as e:
        print("LIFESPAN FAILED:", type(e), e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
