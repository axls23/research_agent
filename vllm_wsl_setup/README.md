# vLLM Server Setup for WSL

This folder contains utilities for setting up and running a local **vLLM** inference server inside your WSL environment. This server acts as the centralized brain for the NEXUS orchestrator and its asynchronous subagents, serving the `google/gemma-4-26B-A4B-it` model.

## Prerequisites
- A WSL 2 environment (Ubuntu recommended).
- **NVIDIA CUDA support in WSL:** You must install the NVIDIA Container Toolkit or ensure CUDA is passed through to WSL.
- A **HuggingFace** account with access to the gated Gemma 4 repository.

## Setup Instructions

### 1. Install Dependencies
Run the setup script inside WSL to install Python 3, pip, and configure an isolated virtual environment (`~/vllm-env`):
```bash
bash setup_wsl_vllm.sh
```

### 2. Authenticate with HuggingFace
Gemma 4 is a gated model. You need to authenticate your WSL environment with HuggingFace to download the weights:
```bash
source ~/vllm-env/bin/activate
pip install -U "huggingface_hub[cli]"
huggingface-cli login
```

### 3. Start the Server
Run the startup script to launch the OpenAI-compatible vLLM API server:
```bash
bash start_gemma4_vllm.sh
```

By default, the server runs on `http://localhost:8000/v1` and expects the API key `vllm`. 
The `nexus.py` framework is already configured to route traffic to this endpoint when you run it in Windows.

## Connecting from Windows
Once the server is running inside WSL, it binds to `0.0.0.0:8000`. Windows automatically proxies `localhost` to WSL, so you can simply run your pipeline in your Windows PowerShell environment:
```powershell
python nexus.py run --topic "Quantum Machine Learning"
```
The central orchestrator and all agents will pipe their reasoning requests straight into the WSL vLLM server.
