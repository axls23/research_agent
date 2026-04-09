import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Test the proxy on 8001
url = "http://127.0.0.1:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": os.getenv("RLM_PRIMARY_MODEL", "Qwen/Qwen2.5-0.5B"),
    "messages": [{"role": "user", "content": "hi"}]
}

print(f"Testing Fast-RLM Proxy at {url}...")
try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
