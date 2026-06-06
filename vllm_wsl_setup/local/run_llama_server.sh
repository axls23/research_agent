#!/bin/bash
export LD_LIBRARY_PATH="$(pwd)/../llama-b9500:$LD_LIBRARY_PATH"
../llama-b9500/llama-server \
  -m ./gemma-4-12b-it-UD-Q6_K_XL.gguf \
  --model-draft ./gemma-4-12B-it-assistant-Q8_0.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-type-k q8_0 \
  --spec-draft-type-v q8_0 \
  --ctx-size 131072 \
  --batch-size 4096 \
  --ubatch-size 512 \
  --flash-attn on \
  --n-gpu-layers 999 \
  --n-gpu-layers-draft 999 \
  --fit off \
  --jinja
