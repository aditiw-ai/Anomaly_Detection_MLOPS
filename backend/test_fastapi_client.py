import asyncio
import os
import sys

# Add backend to PYTHONPATH
sys.path.insert(0, os.path.abspath("."))

from fastapi.testclient import TestClient
from app.main import app

def run_test():
    client = TestClient(app)
    response = client.post("/api/v1/auth/login", data={"username": "admin@example.com", "password": "password123"})
    print("STATUS", response.status_code)
    print("DATA", response.text)

if __name__ == "__main__":
    run_test()
