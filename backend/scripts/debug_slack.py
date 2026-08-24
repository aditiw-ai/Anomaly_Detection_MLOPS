import os
import requests
from dotenv import load_dotenv

load_dotenv(".env")
url = os.environ.get("SLACK_WEBHOOK_URL")

if not url:
    print("❌ SLACK_WEBHOOK_URL is missing from .env")
else:
    print(f"URL loaded: {url[:30]}...")
    try:
        resp = requests.post(url, json={"text": "Hello! Testing webhook manually."})
        print(f"HTTP Status Code: {resp.status_code}")
        print(f"Response Body: {resp.text}")
    except Exception as e:
        print(f"Failed to connect: {e}")
