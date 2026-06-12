# mlx-local-server

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/built%20with-Rust-orange.svg)](https://www.rust-lang.org)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%2FM2%2FM3%2FM4-black.svg)](https://www.apple.com/mac/)

OpenAI-compatible inference servers for Apple Silicon — LLM, image generation, and audio — written in Rust with Python (MLX) for inference via [PyO3](https://pyo3.rs).

**8 MB idle RAM. 16 ms cold start. Single binary.**

| Server | Capability | Port |
|---|---|---|
| `mlx-lm-server` | Chat completions, embeddings, vision, LoRA, fine-tuning | `8080` |
| `mlx-audio-server` | TTS, STT, audio translation, source separation | `8001` |
| `mlx-image-server` | FLUX.2-klein text-to-image, 9s/image on M-series | `8002` |

> Showcased at [WWDC 2025 — Build local AI agents on Mac with MLX](https://developer.apple.com/videos/play/wwdc2025/).

---

## Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4/M5)
- macOS 13+
- Python 3.12 or 3.13
- Rust 1.75+ — install via `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

```bash
# Clone and set up Python env
git clone https://github.com/Ar9av/mlx-lm-server
cd mlx-lm-server
python3 -m venv .venv
source .venv/bin/activate
pip install mlx mlx-lm mflux mlx-audio
```

---

## Quick start

```bash
# LLM — chat completions on :8080
./run.sh lm

# Image generation — FLUX.2 on :8002
./run.sh image

# Audio (TTS/STT) on :8001
./run.sh audio
```

Models are downloaded automatically on first use. Force a rebuild after Rust changes with `BUILD=1 ./run.sh lm`.

### Try it

```bash
# Load a model
curl -X POST http://localhost:8080/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}'

# Chat
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Generate an image (downloads ~4 GB on first run, then cached)
curl http://localhost:8002/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a red apple on a table, photorealistic", "size": "1024x1024"}' \
  | python3 -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

Works as a drop-in replacement for the OpenAI API — point any OpenAI SDK at `http://localhost:8080/v1`.

---

## Why not `python -m mlx_lm.server`?

mlx-lm ships a built-in Python server. This wraps it in a Rust HTTP layer with significantly more surface area.

### Runtime

| | `mlx_lm.server` | mlx-local-server |
|---|---|---|
| Language | Python (uvicorn/starlette) | Rust (tokio + axum) |
| Idle RSS | ~60–100 MB | **8 MB** |
| Cold start | ~3–5 s | **16 ms** |
| Concurrency | asyncio + GIL contention | tokio async, GIL only during inference |
| Deployment | needs Python env in PATH | single self-contained binary |

### API surface

| Feature | `mlx_lm.server` | mlx-local-server |
|---|---|---|
| OpenAI chat completions + streaming | ✅ | ✅ |
| Text completions | ✅ | ✅ |
| Embeddings | ❌ | ✅ |
| Reranking (`/v1/rerank`) | ❌ | ✅ |
| Anthropic Messages API | ❌ | ✅ |
| Vision routing (`mlx_vlm`) | ❌ | ✅ auto-detects `image_url` |
| TTS / STT / source separation | ❌ | ✅ |
| Image generation (FLUX.2) | ❌ | ✅ |
| Tokenize endpoint | ❌ | ✅ |
| Built-in benchmarking | ❌ | ✅ |
| HuggingFace model search | ❌ | ✅ |
| Logprobs | ❌ | ✅ |
| Seed | ❌ | ✅ |
| Prompt cache (KV reuse) | ❌ | ✅ |
| Reasoning field separation | ❌ | ✅ `reasoning_content` delta |
| SSE keep-alive during prefill | ❌ | ✅ |

### Model lifecycle

| Feature | `mlx_lm.server` | mlx-local-server |
|---|---|---|
| Runtime load/unload without restart | ❌ | ✅ |
| LoRA adapter hot-swap | single at startup | ✅ multiple, per-request |
| Speculative decoding | ❌ | ✅ per-request `num_draft_tokens` |
| MoE top-k override | ❌ | ✅ `MLX_MOE_TOP_K` |
| Warm-up KV cache at startup | ❌ | ✅ `MLX_WARM_PROMPTS` |
| RAM guard | ❌ | ✅ |

### Sampler parameters

`mlx_lm.server` exposes `temperature` and `top_p`. This server exposes all 10:

`temperature` · `top_p` · `top_k` · `min_p` · `repetition_penalty` · `presence_penalty` · `frequency_penalty` · `num_draft_tokens` · `xtc_probability` · `xtc_threshold`

---

## mlx-lm-server

### Features

- **Chat completions** (`POST /v1/chat/completions`) — streaming SSE + sync
- **Anthropic messages** (`POST /v1/messages`)
- **Text completions** (`POST /v1/completions`)
- **Embeddings** (`POST /v1/embeddings`)
- **Reranking** (`POST /v1/rerank`) — cosine similarity scoring for RAG pipelines
- **Vision** — auto-routes `image_url` messages to `mlx_vlm`
- **LoRA adapter hot-swap** — multiple adapters, per-request routing
- **Speculative decoding** — pass `drafter` on load, `num_draft_tokens` per request
- **KV-cache quantization** — `kv_bits` + `kv_group_size`
- **Prompt cache** — `session_id` for KV-cache reuse across turns
- **Tool use / function calling** — auto-detects model's parser (Llama-3, Qwen, Mistral, etc.)
- **Reasoning models** — `thinking_budget` caps `<think>` tokens; `reasoning_content` field in response
- **Logprobs** — `logprobs` + `top_logprobs`
- **Fine-tuning** (`POST /v1/train`) — LoRA/DoRA with SSE progress stream
- **Adapter fuse** (`POST /v1/adapters/:name/fuse`)
- **Model convert** (`POST /v1/convert`) — GGUF or HF → MLX

### Benchmarks

Tested on Apple M-series, `mlx-community/Llama-3.2-1B-Instruct-4bit`:

| Metric | Result |
|---|---|
| Cold start | 16 ms |
| Model load (cached) | 2.4 s |
| Idle RSS | 8 MB |
| Throughput (streaming) | 115–261 tok/s |
| Time to first token | 86–96 ms |
| 4× concurrent requests | 0.37 s wall, 0 errors |

### Request parameters

| Parameter | Type | Description |
|---|---|---|
| `temperature` | float | Sampling temperature (default: 0.7) |
| `top_p` | float | Nucleus sampling (default: 0.9) |
| `top_k` | int | Top-K sampling |
| `min_p` | float | Minimum probability threshold |
| `repetition_penalty` | float | Repeat penalty |
| `presence_penalty` | float | Presence penalty |
| `frequency_penalty` | float | Frequency penalty |
| `xtc_probability` | float | XTC sampler — drop probability for top tokens |
| `xtc_threshold` | float | XTC sampler — logit threshold |
| `num_draft_tokens` | int | Speculative decoding steps |
| `kv_bits` | int | KV-cache quantization (4 or 8) |
| `stop` | string \| array | Stop sequences |
| `seed` | int | Reproducible outputs |
| `logprobs` | bool | Per-token log probabilities |
| `top_logprobs` | int | Top-N alternatives per token |
| `session_id` | string | Prompt cache key |
| `thinking_budget` | int | Max reasoning tokens before forcing `</think>` (Qwen3/R1) |
| `chat_template_kwargs` | object | Extra kwargs forwarded to `apply_chat_template` (e.g. `{"enable_thinking": false}`) |

### API examples

```bash
# Load a model
curl -X POST http://localhost:8080/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}'

# Chat (streaming)
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Tool use
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama",
    "messages": [{"role":"user","content":"What is the weather in London?"}],
    "tools": [{"type":"function","function":{"name":"get_weather","description":"Get weather","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}]
  }'

# Prompt cache (reuse KV across turns)
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"My name is Alice."}],"session_id":"conv-1"}'

# Free session when done
curl -X DELETE http://localhost:8080/v1/sessions/conv-1

# Speculative decoding
curl -X POST http://localhost:8080/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","drafter":"mlx-community/Llama-3.2-1B-Instruct-4bit"}'

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"Write a poem"}],"num_draft_tokens":4}'

# Fine-tune (streams SSE progress)
curl http://localhost:8080/v1/train \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit","data":"./my-data","fine_tune_type":"lora","adapter_path":"./my-adapter","iters":500}'

# Fuse adapter into base model
curl -X POST http://localhost:8080/v1/adapters/my-lora/fuse \
  -H 'Content-Type: application/json' \
  -d '{"adapter":"my-lora","output":"./fused-model"}'
```

---

## mlx-image-server

Text-to-image via [mflux](https://github.com/filipstrand/mflux), OpenAI-compatible.

Default model: **FLUX.2-klein-4B** — 4-bit quantized, ~4 GB on disk, ~9 seconds per image on M-series.

Pre-quantized weights (no setup needed): [ar9av/FLUX.2-klein-4B-mflux-4bit](https://huggingface.co/ar9av/FLUX.2-klein-4B-mflux-4bit)

### Features

- **Text-to-image** (`POST /v1/images/generations`) — returns `b64_json` or data URI
- **FLUX.2-klein** (default) and **FLUX.1-schnell/dev**
- **Auto-quantize** — 4-bit on first load, no manual step
- **Auto-load** — loads model on first request if none is loaded
- **Full parameter control** — `size`, `steps`, `guidance`, `seed`, `n`

### Quick start

```bash
# Start the server
./run.sh image

# Generate — model downloads automatically (~4 GB, then cached)
curl http://localhost:8002/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "a futuristic city at night, cinematic", "size": "1024x1024"}' \
  | python3 -c "import sys,json,base64; open('out.png','wb').write(base64.b64decode(json.load(sys.stdin)['data'][0]['b64_json']))"
```

### Models

| Model | Size (quantized) | Steps | Notes |
|---|---|---|---|
| `black-forest-labs/FLUX.2-klein-4B` | ~4 GB | 4 | **Default. Fast, no gate.** |
| `black-forest-labs/FLUX.1-schnell` | ~9 GB | 4 | Requires HF gate acceptance |
| `black-forest-labs/FLUX.1-dev` | ~9 GB | 20–50 | Higher quality, requires gate |

For FLUX.1 models, accept the license at huggingface.co/black-forest-labs and run `huggingface-cli login`.

### OpenAI Python SDK

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8002/v1", api_key="local")
resp = client.images.generate(
    model="flux2-klein",
    prompt="a watercolor painting of a mountain lake at sunrise",
    size="1024x1024",
)
open("out.png", "wb").write(base64.b64decode(resp.data[0].b64_json))
```

### Request parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Text prompt |
| `model` | string | `flux2-klein-4b` | Model alias or HF repo ID |
| `size` | string | `1024x1024` | `WIDTHxHEIGHT` |
| `n` | int | `1` | Images to generate (max 4) |
| `response_format` | string | `b64_json` | `b64_json` or `url` (data URI) |
| `steps` | int | `4` | Inference steps |
| `guidance` | float | `1.0` | Guidance scale |
| `seed` | int | random | Reproducibility |
| `quantize` | int | `4` | Quantize on load: `4` or `8` |

---

## mlx-audio-server

OpenAI-compatible audio via [mlx-audio](https://github.com/Blaizzy/mlx-audio).

### Features

- **TTS** (`POST /v1/audio/speech`) — Kokoro, streaming chunked WAV or full file
- **STT** (`POST /v1/audio/transcriptions`) — Whisper-family, optional timestamps
- **Translation** (`POST /v1/audio/translations`) — STT with forced English output
- **Source separation** (`POST /v1/audio/separations`) — SAM-Audio, text-guided

### API examples

```bash
# TTS
curl http://localhost:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"Hello from Apple Silicon!","voice":"af_heart"}' \
  --output speech.wav

# STT
curl http://localhost:8001/v1/audio/transcriptions \
  -F file=@audio.wav -F model=whisper-large-v3

# Source separation
curl http://localhost:8001/v1/audio/separations \
  -F file=@mixed.wav -F description="speech"
```

### Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="local")

with client.audio.speech.with_streaming_response.create(
    model="kokoro", voice="af_heart", input="Hello!"
) as r:
    r.stream_to_file("out.wav")

with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper", file=f).text)
```

---

## Multi-Mac distributed inference

mlx-lm v0.30.6+ supports multi-rank distributed inference via `mx.distributed`. The Rust server talks to rank-0.

| Backend | Transport | Requirements |
|---|---|---|
| JACCL | Thunderbolt RDMA | macOS 26.2+, Thunderbolt 5 mesh |
| Ring | TCP/Ethernet | any macOS |
| MPI | any | MPI install |

```bash
# Generate hostfile
mlx.distributed_config --hosts mac1.local,mac2.local --over thunderbolt --backend jaccl --output hostfile.json

# Launch 70B across two Macs
MLX_METAL_FAST_SYNCH=1 mlx.launch --backend jaccl --hostfile hostfile.json \
  -- python -m mlx_lm chat --model mlx-community/Llama-3.1-70B-Instruct-4bit
```

> Reference: [WWDC 2025 — Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2025/)

---

## Building from source

```bash
# Check workspace
PYO3_PYTHON=.venv/bin/python cargo check --workspace

# Release binaries
PYO3_PYTHON=.venv/bin/python cargo build --release --workspace
# → target/release/mlx-lm-server
# → target/release/mlx-audio-server
# → target/release/mlx-image-server
```

## Repo layout

```
mlx-local-server/
├── run.sh                  Entry point: ./run.sh lm|audio|image
├── Cargo.toml              Workspace manifest
├── mlx-lm-server/          LLM inference server
├── mlx-audio-server/       Audio inference server
├── mlx-image-server/       Image generation server
├── tools/
│   └── convert_mflux_model.py   Convert old mflux community models to 0.18+ format
└── examples/
    ├── sts_demo.py          Live mic → source-separation demo
    └── bench_lm.py          Throughput / TTFT benchmark
```

## Related

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM inference
- [mflux](https://github.com/filipstrand/mflux) — FLUX image generation on MLX
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — Audio inference
- [FLUX.2-klein-4B-mflux-4bit](https://huggingface.co/ar9av/FLUX.2-klein-4B-mflux-4bit) — Pre-quantized image model

## License

MIT
