# mlx-lm-server

OpenAI-compatible LLM inference server for Apple Silicon, written in Rust.

Inspired by [vishalnagda1/mlx-lm-server](https://github.com/vishalnagda1/mlx-lm-server). This version replaces the Python/FastAPI backend with a native Rust binary (axum) while keeping MLX inference via PyO3 bindings — giving you fast cold-start, lower memory overhead, and a single deployable binary.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Rust toolchain (`rustup`)
- Python 3.13 (`brew install python@3.13`) + `uv`

## Setup

```bash
# Create venv and install mlx-lm
uv venv .venv --python python3.13
uv pip install mlx-lm --python .venv/bin/python

# Build (release)
PYO3_PYTHON="$(pwd)/.venv/bin/python" cargo build --release
```

## Running

```bash
./run.sh
# or manually:
PYTHONPATH=".venv/lib/python3.13/site-packages" \
  VIRTUAL_ENV=".venv" \
  ./target/release/mlx-lm-server
```

Server starts on `http://localhost:8000` by default.

## API

### OpenAI-compatible

```bash
# Chat completions (auto-loads model on first request)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit", "messages": [{"role":"user","content":"Count to 5"}], "stream": true}'

# List loaded models
curl http://localhost:8000/v1/models

# Load a model explicitly
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit"}'

# Unload
curl -X DELETE "http://localhost:8000/v1/models/mlx-community/Mistral-7B-Instruct-v0.3-4bit"
```

### Model management

```bash
# List locally cached HuggingFace models
curl http://localhost:8000/api/models/local

# Delete a cached model
curl -X DELETE http://localhost:8000/api/models/local/mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Search mlx-community models on HuggingFace
curl "http://localhost:8000/api/huggingface/models?search=mistral&limit=10"
```

### Health

```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
```

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `MLX_PORT` | `8000` | Listen port |
| `MLX_HOST` | `0.0.0.0` | Listen address |
| `MLX_DEFAULT_MODEL` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | Auto-loaded model |
| `MLX_DEFAULT_MAX_TOKENS` | `2048` | Default generation limit |
| `MLX_DEFAULT_TEMPERATURE` | `0.7` | Default temperature |
| `MLX_DEFAULT_TOP_P` | `0.9` | Default top-p |
| `MLX_STREAM_TIMEOUT` | `120.0` | SSE stream timeout (seconds) |
| `MLX_MAX_MODEL_SIZE_GB` | — | Reject models larger than this |
| `MLX_ALLOWED_MODELS` | — | Comma-separated allowlist |
| `MLX_CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | CORS origins |
| `MLX_DEBUG` | `false` | Debug logging |

## Architecture

```
src/
├── main.rs          — axum router + startup
├── config.rs        — env-var config
├── error.rs         — MlxError → HTTP response
├── models.rs        — OpenAI-compatible types
├── state.rs         — shared AppState
├── mlx_service.rs   — PyO3 bridge to mlx_lm (load/generate/stream)
└── routes/
    ├── chat.rs      — POST /v1/chat/completions
    ├── health.rs    — GET /health, /status, /
    └── models.rs    — model load/unload + HF search + local cache
```

MLX runs on Python via PyO3. Inference happens in `tokio::task::spawn_blocking` threads so the async runtime is never blocked. Streaming uses an `mpsc` channel — the blocking thread feeds tokens into it, the axum handler drains it into SSE chunks.
