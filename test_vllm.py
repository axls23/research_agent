import os
import requests
from dotenv import load_dotenv
load_dotenv()

url = f"{os.getenv('OPENAI_API_BASE')}/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
}
data = {
    "model": "Qwen/Qwen2.5-0.5B",
    "messages": [{"role": "user", "content": "hi"}]
}

try:
    response = requests.post(url, headers=headers, json=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
