import requests
import json

# Test the Azure endpoint (replace with your actual Azure URL)
azure_url = "https://bajaj-cjb6e0hqhcgrg7cd.spaincentral-01.azurewebsites.net/hackrx/run"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer b57bd62a8ac6975e085fe323f226a67b4cf72557d1b87eeb5c8daef5a1df1ecd"
}

# Load the test data
with open("test_request.json", "r") as f:
    data = json.load(f)

print("Testing Azure endpoint...")
print(f"URL: {azure_url}")
print(f"Data: {json.dumps(data, indent=2)}")

try:
    response = requests.post(azure_url, headers=headers, json=data, timeout=120)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ Azure deployment is working!")
    else:
        print("\n❌ Azure deployment still has issues")
        
except Exception as e:
    print(f"Error: {e}") 