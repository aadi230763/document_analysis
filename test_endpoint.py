import requests
import json

# Test the local endpoint
url = "http://localhost:5001/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer b57bd62a8ac6975e085fe323f226a67b4cf72557d1b87eeb5c8daef5a1df1ecd"
}

# Load the test data
with open("test_request.json", "r") as f:
    data = json.load(f)

print("Testing local endpoint...")
print(f"URL: {url}")
print(f"Data: {json.dumps(data, indent=2)}")

try:
    response = requests.post(url, headers=headers, json=data, timeout=60)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}") 