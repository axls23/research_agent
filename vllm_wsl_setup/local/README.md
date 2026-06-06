---
license: gemma
base_model:
- google/gemma-4-12B-it-assistant
- google/gemma-4-12B-it
pipeline_tag: text-generation
library_name: gguf
tags:
- gguf
- gemma4
- mtp
- speculative-decoding
- draft-model
- llama.cpp
---

# Gemma 4 12B IT MTP Assistant GGUF

GGUF conversion of Google's `google/gemma-4-12B-it-assistant` draft / MTP assistant model for pairing with `google/gemma-4-12B-it`-compatible Gemma 4 12B instruction checkpoints.

These files are generated for the current Gemma 4 MTP llama.cpp work, not stock mainline builds that reject the assistant architecture. Local runtime validation used `am17an/llama.cpp` branch `gemma4-mtp` at commit `b8e703e`, with a converter compatibility patch so the official `Gemma4UnifiedAssistantForCausalLM` config registers as the Gemma 4 assistant converter.

Requested repo was `atx/gemma4-12b-mtp-assistant`; the active token is for `sjakek` and has no `atx` org write rights, so this package was uploaded under `sjakek/gemma4-12b-mtp-assistant`.

## Files

- `gemma-4-12B-it-assistant-BF16.gguf`: BF16 GGUF, 861,520,128 bytes.
- `gemma-4-12B-it-assistant-Q8_0.gguf`: Q8_0 GGUF from branch `llama-quantize`, 465,109,248 bytes.
- `logs/`: conversion, quantization, load, generation, and long-context validation logs.

## Source

- Source model: `google/gemma-4-12B-it-assistant`
- Source revision: `723932f88886ab714522468b94f9c7ee48d8c9a8`
- Source config architecture: `Gemma4UnifiedAssistantForCausalLM`
- Source model type: `gemma4_unified_assistant`
- GGUF architecture emitted here: `gemma4-assistant`

## GGUF Metadata

Structural audit passed on both files:

- `general.architecture = gemma4-assistant`
- `gemma4-assistant.context_length = 131072`
- `gemma4-assistant.embedding_length = 1024`
- `gemma4-assistant.embedding_length_out = 3840`
- `gemma4-assistant.block_count = 4`
- `gemma4-assistant.feed_forward_length = 8192`
- `gemma4-assistant.attention.head_count = 16`
- `gemma4-assistant.attention.head_count_kv = [8, 8, 8, 1]`
- `gemma4-assistant.attention.shared_kv_layers = 4`
- `gemma4-assistant.nextn_predict_layers = 4`
- `gemma4-assistant.attention.sliding_window_pattern = [true, true, true, false]`
- `gemma4-assistant.rope.dimension_count = 512`
- `gemma4-assistant.rope.dimension_count_swa = 256`
- `tokenizer.ggml.bos_token_id = 2`
- `tokenizer.ggml.eos_token_id = 1`
- Tensor count: 49
- Required tensors present: `rope_freqs.weight`, `nextn.pre_projection.weight`, `nextn.post_projection.weight`

## Runtime Validation

Validated locally against the target GGUF:

```text
gemma-4-12b-it-UD-Q6_K_XL.gguf
```

Test results:

- BF16 assistant load and generation: pass.
- Q8_0 assistant load and generation: pass.
- Speculative mode: `--spec-type draft-mtp`.
- Draft KV lane for Q8_0 test: `--spec-draft-type-k q8_0 --spec-draft-type-v q8_0`.
- Serving shape: `--ctx-size 131072 --batch-size 4096 --ubatch-size 512 --flash-attn on`.
- Target-only vs MTP deterministic check at `temperature=0`: decoded output matched.
- BF16 acceptance probe: 6 accepted / 6 generated draft tokens.
- Q8_0 acceptance probe: 6 accepted / 6 generated draft tokens.
- Long-context smoke: 126,009 prompt tokens plus 8 generated tokens, `truncated=false`, no OOM or context error.
- OpenAI-compatible chat endpoint: pass with draft activity observed.

Relevant retained logs:

- `logs/convert-branch-bf16.log`
- `logs/quantize-branch-q8_0.log`
- `logs/completion-target-baseline.json`
- `logs/completion-bf16-mtp.json`
- `logs/completion-q8-mtp.json`
- `logs/completion-q8-mtp-long-context-summary.json`
- `logs/server-final-q8-mtp-canonical.log`
- `logs/openai-chat-final-q8-canonical.json`

## Benchmarks

Additional local benchmark reports are included in this repository:

- `benchmarks/gemma4_q4xl_mtp_2k_64k`: single-stream Q4_K_XL target-only vs Q8 MTP, draft max 3, temperature 0.
- `benchmarks/gemma4_q4xl_mtp_2k_64k_draft2_temp06_topk1`: single-stream Q4_K_XL target-only vs Q8 MTP, draft max 2, drafter top-k 1, accepter temperature 0.6.
- `benchmarks/gemma4_q4xl_mtp_2k_64k_draft2_temp06_topk1_parallel3`: three concurrent 2K-in/2K-out streams, `--parallel 3`, draft max 2, drafter top-k 1, accepter temperature 0.6. This was the first local run where MTP beat target-only on aggregate generation throughput:
  - coding: 47.14 tok/s target-only vs 50.57 tok/s MTP
  - general: 48.42 tok/s target-only vs 50.60 tok/s MTP

## Checksums

```text
be2ff6cf6dc9f4d753be846efb990606a5fec1b9c758c7f200112d2431f5e248  gemma-4-12B-it-assistant-BF16.gguf
cb9b46d9ff820b2b9b0d53cc911a2bc27eb2faf84700284047244d8f28883794  gemma-4-12B-it-assistant-Q8_0.gguf
```

## llama.cpp Example

Use a Gemma 4 MTP-capable llama.cpp branch/build:

```bash
llama-server \
  -m gemma-4-12b-it-UD-Q6_K_XL.gguf \
  --model-draft gemma-4-12B-it-assistant-Q8_0.gguf \
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
```

Stock llama.cpp builds that do not include Gemma 4 assistant support may fail with `unknown model architecture: 'gemma4-assistant'`.
