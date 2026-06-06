import urllib.request
import json
import os

url = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    
download_url = None
for asset in data.get("assets", []):
    if "ubuntu-x64.tar.gz" in asset.get("name", ""):
        download_url = asset["browser_download_url"]
        break

if download_url:
    print(f"Downloading from {download_url}")
    os.system(f"wget {download_url} -O llama.tar.gz")
    os.system("tar -xzf llama.tar.gz")
    os.system("chmod +x build/bin/llama-server build/bin/llama-cli 2>/dev/null || chmod +x llama-server llama-cli 2>/dev/null")
    os.system("cp build/bin/llama-* . 2>/dev/null || true")
    os.system("cp llama-server /usr/local/bin/ 2>/dev/null || sudo cp llama-server /usr/local/bin/ 2>/dev/null || true")
    print("Done")
else:
    print("Could not find ubuntu-x64.tar.gz release.")
