# mlx-local-server

A monorepo of OpenAI-compatible inference servers for Apple Silicon, written in Rust. Both servers embed Python via [PyO3](https://pyo3.rs) — Metal acceleration stays in Python, everything else (HTTP, concurrency, streaming) runs natively in Rust.

| Server | What it does | Default port |
|---|---|---|
| [`mlx-lm-server`](./mlx-lm-server) | LLM chat completions, Anthropic compat, LoRA adapters, vision | `8080` |
| [`mlx-audio-server`](./mlx-audio-server) | TTS, STT, audio translation, source separation | `8001` |
| [`mlx-image-server`](./mlx-image-server) | FLUX.1 image generation (text-to-image) | `8002` |

**Single binaries. ~8 MB idle RSS each. Drop-in replacement for OpenAI API.**

> Built with [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple's open-source ML framework for Apple Silicon. Showcased at [WWDC 2025 — Build local AI agents on Mac with MLX](https://developer.apple.com/videos/play/wwdc2025/).

---

## Why not `python -m mlx_lm.server`?

mlx-lm ships a built-in Python server. This project wraps it in a Rust HTTP layer and extends it significantly. Here's what's different:

### Runtime

| | `mlx_lm.server` | mlx-local-server |
|---|---|---|
| Language | Python (uvicorn/starlette) | Rust (tokio + axum) |
| Idle RSS | ~60–100 MB | **8 MB** |
| Cold start | ~3–5 s (Python imports) | **16 ms** |
| Concurrency | asyncio + GIL contention | tokio async, GIL only during inference |
| Deployment | needs Python env in PATH | single self-contained binary |

### API surface

| Feature | `mlx_lm.server` | mlx-local-server |
|---|---|---|
| OpenAI chat completions + streaming | ✅ | ✅ |
| Text completions | ✅ | ✅ |
| Embeddings | ❌ | ✅ |
| Anthropic Messages API | ❌ | ✅ |
| Vision routing (`mlx_vlm`) | ❌ | ✅ auto-detects `image_url` |
| TTS / STT / source separation | ❌ | ✅ (mlx-audio-server) |
| Tokenize endpoint | ❌ | ✅ |
| Built-in benchmarking (TTFT + tok/s percentiles) | ❌ | ✅ |
| HuggingFace Hub model search | ❌ | ✅ |
| Ollama `/api/ps` compat | ❌ | ✅ |
| Logprobs (`logprobs` + `top_logprobs`) | ❌ | ✅ |
| Seed (`seed`) | ❌ | ✅ |
| Stop sequences (`stop`) | ❌ | ✅ single string or array |
| Prompt cache (session-based KV reuse) | ❌ | ✅ `session_id` param |

### Model & adapter lifecycle

| Feature | `mlx_lm.server` | mlx-local-server |
|---|---|---|
| Runtime load/unload without restart | ❌ | ✅ |
| LoRA adapter hot-swap | single adapter at startup | ✅ multiple, per-request routing |
| Speculative decoding | ❌ server-level | ✅ drafter model + per-request `num_draft_tokens` |
| RAM guard (pre-load memory check) | ❌ | ✅ |
| Model allowlist / size limit | ❌ | ✅ env vars |
| Scan local HuggingFace cache | ❌ | ✅ |

### Sampler parameters

`mlx_lm.server` exposes `temperature` and `top_p`. This server exposes all 8 mlx-lm sampling parameters — including `presence_penalty` and `frequency_penalty`, which exist in mlx-lm but were never wired through its built-in server:

`temperature` · `top_p` · `top_k` · `min_p` · `repetition_penalty` · `presence_penalty` · `frequency_penalty` · `num_draft_tokens`

### Fine-tuning workflow

mlx-lm's CLI handles offline training via `mlx_lm.lora --train` and `mlx_lm.fuse`. This server wraps those same Python modules behind HTTP endpoints — train, fuse, and convert without leaving the API:

```bash
# 1. Train via API — streams SSE progress events while training runs
curl http://localhost:8080/v1/train \
  -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit",
       "data":"./my-data","iters":1000,"adapter_path":"./my-adapter"}'

# 2. Mount the trained adapter (no restart)
curl -X POST http://localhost:8080/v1/adapters/mount \
  -d '{"name":"my-lora","adapter_path":"./my-adapter"}'

# 3. Route specific requests to it (per-request, same server)
curl http://localhost:8080/v1/chat/completions \
  -d '{"messages":[...],"adapter_name":"my-lora"}'

# 4. Fuse adapter into base model when done iterating
curl -X POST http://localhost:8080/v1/adapters/my-lora/fuse \
  -d '{"adapter":"my-lora","output":"./fused-model"}'
```

Multiple adapters can be mounted simultaneously on the same base model, each reachable by name per request. `mlx_lm.lora --train` still works for offline training; the API is an additional option.

---

## Quick start

```bash
# LLM server  →  http://localhost:8080
./run.sh lm

# Audio server  →  http://localhost:8001
./run.sh audio

# Image server  →  http://localhost:8002
./run.sh image

# Force rebuild after Rust changes
BUILD=1 ./run.sh lm
```

---

## mlx-lm-server

OpenAI-compatible LLM inference powered by [mlx-lm](https://github.com/ml-explore/mlx-lm).

### Features

- **OpenAI chat completions** (`POST /v1/chat/completions`) — streaming SSE + sync
- **Anthropic messages** (`POST /v1/messages`) — streaming + sync
- **Text completions** (`POST /v1/completions`)
- **Embeddings** (`POST /v1/embeddings`)
- **Vision** — routes `image_url` messages to `mlx_vlm` automatically
- **LoRA adapter hot-swap** (`GET/POST/DELETE /v1/adapters`)
- **Speculative decoding** — pass `drafter` in load request, `num_draft_tokens` per request
- **KV-cache quantization** — `kv_bits` + `kv_group_size` per request
- **Full sampler control** — `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`
- **Stop sequences** — `stop` accepts a string or array; finish_reason reflects whether a stop string or length limit was hit
- **Logprobs** — `logprobs: true` + `top_logprobs: N` returns per-token log probabilities and top-N alternatives
- **Seed** — `seed` for reproducible outputs
- **Prompt cache** — `session_id` enables KV-cache reuse across multi-turn requests; `DELETE /v1/sessions/:id` frees it
- **Tool use / function calling** — `tools` + `tool_choice` in any chat request; auto-detects model's tool parser (Llama-3, Qwen, Mistral, Gemma, etc.)
- **Benchmarking** (`POST /v1/benchmark`) — TTFT/tps percentiles
- **Model info** (`GET /v1/models/:id/info`) — scans HF cache, reads config.json
- **RAM guard** — rejects loads that would exceed available memory
- **Fine-tuning** (`POST /v1/train`) — LoRA / DoRA / full fine-tuning with SSE progress stream
- **Adapter fuse** (`POST /v1/adapters/:name/fuse`) — merge adapter weights into base model
- **Model convert** (`POST /v1/convert`) — convert GGUF or HuggingFace models to MLX format

### Request parameters

All chat completion parameters:

| Parameter | Type | Description |
|---|---|---|
| `temperature` | float | Sampling temperature (default: 0.7) |
| `top_p` | float | Nucleus sampling (default: 0.9) |
| `top_k` | int | Top-K tokens to sample from |
| `min_p` | float | Minimum probability threshold |
| `repetition_penalty` | float | Penalty for repeated tokens |
| `presence_penalty` | float | Penalise tokens already in context |
| `frequency_penalty` | float | Penalise tokens by frequency |
| `num_draft_tokens` | int | Speculative decoding draft steps (requires drafter model) |
| `kv_bits` | int | KV-cache quantization bits (4 or 8) |
| `kv_group_size` | int | KV-cache quantization group size |
| `stop` | string \| array | Stop generation at this string (or first match in array) |
| `seed` | int | RNG seed for reproducible outputs |
| `logprobs` | bool | Return per-token log probabilities |
| `top_logprobs` | int | Number of top-token alternatives per position (requires `logprobs: true`) |
| `session_id` | string | Reuse KV cache across requests with the same ID (prompt cache) |

### Benchmarks

Tested on Apple M-series with `mlx-community/Llama-3.2-1B-Instruct-4bit`:

| Metric | Result |
|---|---|
| Cold start | 16 ms |
| Model load (cached) | 2.4 s |
| Idle memory (RSS) | 8 MB |
| Throughput (streaming) | 115–261 tok/s |
| Time to first token | 86–96 ms |
| 4× concurrent requests | 0.37 s wall, 0 errors |

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

# Tool use / function calling
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "llama",
    "messages": [{"role":"user","content":"What is the weather in London?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["location"]
        }
      }
    }]
  }'
# → {"choices":[{"finish_reason":"tool_calls","tool_calls":[{"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":"{\"location\":\"London\"}"}}]}]}

# Chat with sampler params
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}],
       "temperature":0.8,"top_k":50,"repetition_penalty":1.1}'

# Stop sequences — stop at first match, finish_reason reflects it
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"List items:"}],
       "stop":["3.","END"]}'

# Seed — reproducible outputs
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"Pick a number"}],
       "seed":42,"temperature":1.0}'

# Logprobs — per-token log probabilities + top-5 alternatives
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"Say hi"}],
       "logprobs":true,"top_logprobs":5}'
# → choices[0].logprobs.content[*].logprob  (chosen token)
# → choices[0].logprobs.content[*].top_logprobs  (top-5 alternatives)

# Prompt cache — reuse KV cache across a multi-turn session
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"My name is Alice."}],
       "session_id":"conv-1"}'

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"My name is Alice."},
       {"role":"assistant","content":"Hello Alice!"},
       {"role":"user","content":"What is my name?"}],
       "session_id":"conv-1"}'
# Second request skips re-encoding the shared prefix

# Free the session KV cache when done
curl -X DELETE http://localhost:8080/v1/sessions/conv-1

# Speculative decoding (load drafter first, then per-request control)
curl -X POST http://localhost:8080/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Llama-3.2-3B-Instruct-4bit",
       "drafter":"mlx-community/Llama-3.2-1B-Instruct-4bit"}'

curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[...],"num_draft_tokens":4}'

# Mount a LoRA adapter
curl -X POST http://localhost:8080/v1/adapters/mount \
  -H 'Content-Type: application/json' \
  -d '{"name":"my-lora","adapter_path":"/path/to/adapter","model":"mlx-community/Llama-3.2-3B-Instruct-4bit"}'

# Fine-tune (LoRA) — streams SSE progress events
curl http://localhost:8080/v1/train \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "data": "./my-data",
    "fine_tune_type": "lora",
    "adapter_path": "./my-adapter",
    "iters": 500,
    "batch_size": 4,
    "learning_rate": 1e-4
  }'
# data: {"event":"progress","message":"Training lora — 500 iters, lr=1e-4"}
# data: {"event":"done","adapter_path":"./my-adapter","message":"Training complete"}

# Fuse adapter into base model
curl -X POST http://localhost:8080/v1/adapters/my-lora/fuse \
  -H 'Content-Type: application/json' \
  -d '{"adapter":"my-lora","output":"./fused-model"}'

# Convert GGUF or HF model to MLX (with optional quantization)
curl -X POST http://localhost:8080/v1/convert \
  -H 'Content-Type: application/json' \
  -d '{"model":"bartowski/Llama-3.2-3B-Instruct-GGUF","output":"./mlx-llama3","quantize_bits":4}'
```

---

## mlx-image-server

OpenAI-compatible image generation powered by [mflux](https://github.com/filipstrand/mflux) (FLUX.1 on Apple Silicon).

### Features

- **Text-to-image** (`POST /v1/images/generations`) — OpenAI-compatible, returns `b64_json` or data URI
- **FLUX.1 schnell + dev** — fast 4-step schnell (default) or quality-focused dev
- **Custom model path** — use any public HuggingFace repo (including pre-quantized community models)
- **Quantization** — optional 4-bit or 8-bit weight quantization on load
- **Full parameter control** — `size`, `steps`, `guidance`, `seed`, `negative_prompt`, `n`
- **Auto-load** — model loads automatically on first generation request if not pre-loaded

### Models

The official FLUX.1 models (`black-forest-labs/FLUX.1-schnell`) require accepting the license at [huggingface.co/black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) and running `huggingface-cli login`.

Alternatively, use a public pre-quantized community model — no auth required:

| `model` | `model_path` | Size | Notes |
|---|---|---|---|
| `flux-schnell` | `madroid/flux.1-schnell-mflux-4bit` | ~3.4 GB | 4-bit, no auth required |
| `flux-schnell` | *(none, needs HF auth)* | ~34 GB | bf16, then quantize locally |
| `flux-dev` | *(none, needs HF auth)* | ~34 GB | higher quality, 20–50 steps |

### API examples

```bash
# Load a public pre-quantized model (no HF auth needed)
curl -X POST http://localhost:8002/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model": "flux-schnell", "model_path": "madroid/flux.1-schnell-mflux-4bit"}'

# Generate an image (b64_json response)
curl http://localhost:8002/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a red apple on a wooden table, photorealistic",
    "size": "1024x1024",
    "steps": 4,
    "seed": 42
  }' | python3 -c "
import sys, json, base64
r = json.load(sys.stdin)
open('out.png','wb').write(base64.b64decode(r['data'][0]['b64_json']))
print('Saved out.png')
"

# OpenAI Python SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8002/v1", api_key="local")
resp = client.images.generate(
    model="flux-schnell",
    prompt="a watercolor painting of a mountain lake at sunrise",
    size="1024x1024",
    n=1,
)
# resp.data[0].b64_json contains the PNG
```

### Request parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | required | Text prompt |
| `model` | string | `flux-schnell` | `flux-schnell` or `flux-dev` |
| `size` | string | `1024x1024` | `WIDTHxHEIGHT`, e.g. `512x512`, `1024x768` |
| `n` | int | `1` | Number of images (max 4) |
| `response_format` | string | `b64_json` | `b64_json` or `url` (data URI) |
| `steps` | int | `4` | Inference steps (4 for schnell, 20–50 for dev) |
| `guidance` | float | `4.0` | Classifier-free guidance scale |
| `seed` | int | random | RNG seed for reproducibility |
| `negative_prompt` | string | — | Negative prompt (dev only) |
| `quantize` | int | — | Quantize weights on load: `4` or `8` |

---

## mlx-audio-server

OpenAI-compatible audio inference powered by [mlx-audio](https://github.com/Blaizzy/mlx-audio).

### Features

- **TTS** (`POST /v1/audio/speech`) — Kokoro, streaming chunked WAV or full file
- **STT** (`POST /v1/audio/transcriptions`) — Whisper-family, optional segment timestamps
- **Translation** (`POST /v1/audio/translations`) — STT with forced English output
- **Source separation** (`POST /v1/audio/separations`) — SAM-Audio, text-guided target extraction
- **Model management** (`GET/POST/DELETE /v1/models`) — hot-load/unload any model at runtime

### API examples

```bash
# TTS — returns WAV
curl http://localhost:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"Hello from Apple Silicon!","voice":"af_heart"}' \
  --output speech.wav

# TTS streaming
curl http://localhost:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"Streaming...","stream":true}' \
  --output stream.wav

# STT
curl http://localhost:8001/v1/audio/transcriptions \
  -F file=@audio.wav -F model=whisper-large-v3

# Source separation — extract speech from a mixed recording
curl http://localhost:8001/v1/audio/separations \
  -F file=@mixed.wav -F description="speech"
```

### Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="local")

# TTS
with client.audio.speech.with_streaming_response.create(
    model="kokoro", voice="af_heart", input="Hello!"
) as r:
    r.stream_to_file("out.wav")

# STT
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper", file=f).text)
```

---

## Multi-Mac distributed inference

As of [mlx-lm v0.30.6](https://github.com/ml-explore/mlx-lm), mlx-lm's built-in server supports multi-rank distributed inference via `mx.distributed`. This is separate from the Rust servers but can be used alongside them — the Rust server talks to rank-0.

### Backends

| Backend | Transport | Requirements |
|---|---|---|
| **JACCL** | Thunderbolt RDMA | macOS 26.2+, `rdma_ctl enable` in recovery, Thunderbolt 5 mesh |
| **Ring** | TCP/Ethernet | any macOS, ring topology |
| **MPI** | any | MPI install |

### Setup (Thunderbolt, 2 Macs)

```bash
# Generate hostfile
mlx.distributed_config \
  --hosts mac1.local,mac2.local \
  --over thunderbolt \
  --backend jaccl \
  --auto-setup \
  --output hostfile.json

# Launch distributed inference
MLX_METAL_FAST_SYNCH=1 mlx.launch \
  --backend jaccl \
  --hostfile hostfile.json \
  -- python -m mlx_lm chat \
       --model mlx-community/Llama-3.1-70B-Instruct-4bit
```

For Ethernet (ring topology):

```bash
mlx.distributed_config --hosts mac1,mac2,mac3 --over ethernet --backend ring --output hostfile.json
mlx.launch --backend ring --hostfile hostfile.json -- python -m mlx_lm.server --model ...
```

> Reference: [WWDC 2025 — Explore distributed inference and training with MLX](https://developer.apple.com/videos/play/wwdc2025/)  
> Docs: [MLX distributed](https://ml-explore.github.io/mlx/build/html/usage/distributed.html)

---

## Examples

```
examples/
  sts_demo.py      Live mic → source-separation demo (needs mlx-audio-server on :8001)
  bench_lm.py      Throughput / TTFT / concurrency benchmark for mlx-lm-server
```

```bash
# Run the STS demo
pip install sounddevice soundfile scipy numpy requests
python examples/sts_demo.py --server http://localhost:8001

# Mix background music for best results
python examples/sts_demo.py --mix background.wav --duration 8

# Benchmark the LM server
python examples/bench_lm.py
```

---

## Building both

```bash
# Check both crates
PYO3_PYTHON=/path/to/.venv/bin/python cargo check --workspace

# Build both release binaries
PYO3_PYTHON=/path/to/.venv/bin/python cargo build --release --workspace
# → target/release/mlx-lm-server
# → target/release/mlx-audio-server
```

## Repo layout

```
mlx-local-server/
├── run.sh                  Unified entry: ./run.sh lm|audio [flags...]
├── Cargo.toml              Workspace manifest + shared deps
├── mlx-lm-server/          LLM inference server
│   ├── src/
│   ├── Cargo.toml
│   └── run.sh              Thin wrapper → root run.sh lm
├── mlx-audio-server/       Audio inference server
│   ├── src/
│   ├── Cargo.toml
│   └── run.sh              Thin wrapper → root run.sh audio
└── examples/
    ├── sts_demo.py
    └── bench_lm.py
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4/M5)
- Rust 1.75+
- Python 3.12 or 3.13
- M5 with macOS 26.2+: automatic Neural Accelerator (NAX) support via MLX Metal backend — no code changes needed

## Related resources

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — LLM inference Python library
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — Audio inference Python library
- [MLX Swift](https://github.com/ml-explore/mlx-swift) — Swift bindings for MLX
- [WWDC 2025: Build local AI agents on Mac with MLX](https://developer.apple.com/videos/play/wwdc2025/)
- [WWDC 2025: Get started with MLX for Apple silicon](https://developer.apple.com/videos/play/wwdc2025/)
- [WWDC 2025: Explore large language models on Apple silicon with MLX](https://developer.apple.com/videos/play/wwdc2025/)

## License

MIT
