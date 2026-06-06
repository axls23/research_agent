#!/bin/bash
wget $(curl -s https://api.github.com/repos/ggerganov/llama.cpp/releases/latest | grep 'browser_download_url.*ubuntu-x64.zip' | cut -d '"' -f 4) -O llama.zip
unzip -o llama.zip
chmod +x llama-server llama-cli
