import requests
import traceback

def test_login():
    url = "http://127.0.0.1:8000/api/v1/auth/login"
    data = {"username": "admin@example.com", "password": "password123"}
    try:
        response = requests.post(url, data=data)
        print(f"Status Code: {response.status_code}")
        print("Response headers:", response.headers)
        print("Response text:", response.text)
    except Exception as e:
        print("Error:", e)

test_login()
