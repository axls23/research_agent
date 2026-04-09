#!/bin/bash
# ========================================================
# vLLM Server Launcher for WSL2 (Ubuntu)
# ========================================================
# Run this script INSIDE your WSL2 terminal to host the 
# high-speed subagent model for the Deep Agents SDK.
# ========================================================

echo "Setting up vLLM environment in WSL2..."

# 1. Ensure pip and python are installed
sudo apt update
sudo apt install -y python3-pip python3-venv

# 2. Create a virtual environment for the vLLM server
python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate

# 3. Install vLLM (Linux native)
echo "Installing vLLM... (This may take a few minutes if compilation is required)"
pip install vllm

# 4. Start the server
echo "Starting vLLM OpenAI-Compatible Server..."
echo "Model: Qwen/Qwen2.5-1.5B-Instruct"
echo "Port: 8000"
echo "--------------------------------------------------------"
echo "Your Windows host and Deep Agents orchestrator"
echo "will automatically connect to this server at:"
echo "http://localhost:8000/v1"
echo "--------------------------------------------------------"

# Leave 20% VRAM free for the Windows AirLLM orchestrator
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --host 0.0.0.0 \
    --port 8000
