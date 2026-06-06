#!/bin/bash
# setup_wsl_vllm.sh
# Sets up a clean vLLM environment in WSL 2 (Ubuntu)

set -e

echo "================================================="
echo "   NEXUS Engine: WSL vLLM Server Setup           "
echo "================================================="

# 1. Update system packages
echo "==> Updating dnf packages..."
sudo dnf upgrade --refresh -y

# 2. Install Python 3.10+ (vLLM requires modern Python) and pip
echo "==> Installing Python and pip..."
sudo dnf install -y python3 python3-pip python3-virtualenv

# 3. Create an isolated virtual environment for the vLLM server
echo "==> Creating virtual environment 'vllm-env'..."
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate

# 4. Install PyTorch & vLLM
# Note: Ensure your WSL has the NVIDIA Container Toolkit and CUDA toolkit installed
echo "==> Installing vLLM and dependencies..."
pip install --upgrade pip
pip install vllm

# 5. Authenticate with HuggingFace (Gemma 4 is gated, you need an HF token)
echo "==> HuggingFace Authentication..."
echo "To download google/gemma-4-26B-A4B-it, you must authenticate with HuggingFace."
echo "If you haven't already, run: huggingface-cli login"

echo "================================================="
echo "Setup Complete! Activate your environment via:"
echo "  source ~/vllm-env/bin/activate"
echo "================================================="
