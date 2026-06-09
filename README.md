# mlx-lm-server

A native Rust HTTP server that runs MLX LLM inference locally on Apple Silicon with an OpenAI-compatible API.

- **Axum** web framework — async, zero-copy, tokio-native
- **PyO3** bridge to `mlx_lm` — keeps Metal/MLX acceleration without rewriting the inference stack
- Inference runs in `spawn_blocking` threads; streaming tokens flow through `mpsc` channels into SSE — the async runtime is never blocked
- Single ~8 MB binary, 8 MB idle memory footprint

## Benchmarks

Tested on Apple M-series with `mlx-community/Llama-3.2-1B-Instruct-4bit`:

| Metric | Result |
|---|---|
| Server cold-start | **16 ms** |
| Model load (cached) | **2.4 s** |
| Idle memory (RSS) | **8 MB** |
| Loaded memory (RSS) | **380 MB** |
| Throughput — short prompt | **115–161 tok/s** |
| Throughput — long prompt | **206–234 tok/s** |
| Streaming TTFT | **86–96 ms** |
| 4× concurrent requests | **0.37 s wall time, 0 errors** |

Streaming consistently outperforms non-streaming (~260 tok/s peak) because tokens are yielded directly from the Python generator without waiting for the full response to be assembled.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain (`rustup`)
- Python 3.13 (`brew install python@3.13`) + `uv`

## Setup

```bash
git clone <repo>
cd mlx-lm-server

# Create venv and install mlx-lm
uv venv .venv --python python3.13
uv pip install mlx-lm --python .venv/bin/python

# Build release binary
PYO3_PYTHON="$(pwd)/.venv/bin/python" cargo build --release
```

## Running

```bash
./run.sh          # handles venv + build automatically
```

Or manually:

```bash
PYTHONPATH=".venv/lib/python3.13/site-packages" \
  VIRTUAL_ENV=".venv" \
  ./target/release/mlx-lm-server
```

Server listens on `http://localhost:8000` by default.

## API

### Chat completions

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Streaming (SSE)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Count to 10"}],
    "stream": true
  }'
```

The model is auto-loaded on the first request. You can also load/unload explicitly:

```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit"}'

curl -X DELETE "http://localhost:8000/v1/models/mlx-community/Mistral-7B-Instruct-v0.3-4bit"
```

### Model discovery

```bash
# List loaded model
curl http://localhost:8000/v1/models

# List locally cached HuggingFace models
curl http://localhost:8000/api/models/local

# Search mlx-community on HuggingFace
curl "http://localhost:8000/api/huggingface/models?search=mistral&limit=10"

# Delete a cached model from disk
curl -X DELETE http://localhost:8000/api/models/local/mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

### Health

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"current_model":"mlx-community/..."}
```

## Configuration

All config via environment variables with `MLX_` prefix:

| Variable | Default | Description |
|---|---|---|
| `MLX_PORT` | `8000` | Listen port |
| `MLX_HOST` | `0.0.0.0` | Listen address |
| `MLX_DEFAULT_MODEL` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | Model to auto-load |
| `MLX_DEFAULT_MAX_TOKENS` | `2048` | Generation token limit |
| `MLX_DEFAULT_TEMPERATURE` | `0.7` | Sampling temperature |
| `MLX_DEFAULT_TOP_P` | `0.9` | Top-p sampling |
| `MLX_STREAM_TIMEOUT` | `120.0` | SSE stream timeout (s) |
| `MLX_MAX_MODEL_SIZE_GB` | — | Reject models larger than N GB |
| `MLX_ALLOWED_MODELS` | — | Comma-separated model allowlist |
| `MLX_CORS_ORIGINS` | `http://localhost:5173,…` | CORS allowed origins |
| `MLX_DEBUG` | `false` | Verbose logging |

## Benchmarking

```bash
python3 bench.py
```

Runs cold-start, model load, throughput (streaming + non-streaming), TTFT, and 4× concurrency test.

## Project structure

```
src/
├── main.rs          — axum router, startup
├── config.rs        — env-var config
├── error.rs         — MlxError → HTTP response mapping
├── models.rs        — OpenAI-compatible request/response types
├── state.rs         — shared AppState (config + service)
├── mlx_service.rs   — PyO3 bridge: load / generate / stream_generate
└── routes/
    ├── chat.rs      — POST /v1/chat/completions (sync + SSE)
    ├── health.rs    — GET /health /status /
    └── models.rs    — model management + HF search + local cache
```

## Using with OpenAI SDK

Because the API is OpenAI-compatible, you can point any OpenAI client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```
