#!/bin/bash
# start_gemma4_vllm.sh
# Starts the vLLM OpenAI-compatible server with Gemma 4 26B

set -e

# Make sure the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating vllm-env..."
    source ~/vllm-env/bin/activate
fi

MODEL_NAME="sjakek/gemma4-12b-mtp-assistantclear"
# ===============================================================
# Hugging Face Authentication
# Replace the string below with your actual HF Token from:
# https://huggingface.co/settings/tokens
# ===============================================================
export HF_TOKEN="${HF_TOKEN:-your_hf_token_here}"

echo "================================================="
echo "   Starting vLLM Server on port 8000             "
echo "   Model: $MODEL_NAME                            "
echo "================================================="

# Start the OpenAI compatible API server
# Added flags for 6GB VRAM / 16GB RAM offloading:
# --cpu-offload-gb 10: Offload up to 10GB of model weights to system RAM
# --gpu-memory-utilization 0.80: Use 80% of the 6GB VRAM for weights & KV cache
# --enforce-eager: Disable CUDA graphs to save memory
# --max-model-len 2048: Limit context window to save KV cache VRAM
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key vllm \
    --max-model-len 2048 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.80 \
    --cpu-offload-gb 10 \
    --enforce-eager \
    --trust-remote-code \
    --spec-model google/gemma-4-26B-A4B-it-assistant \
    --spec-tokens 4 \
    --spec-method gemma4_mtp
