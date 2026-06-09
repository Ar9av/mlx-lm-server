# mlx-lm-server

An OpenAI-compatible inference server for Apple Silicon, written in Rust. Bridges `mlx_lm` Python inference via [PyO3](https://pyo3.rs) — Metal acceleration stays in Python, everything else (HTTP, concurrency, streaming) runs natively in Rust.

**Single ~8 MB binary. 8 MB idle RSS. Drop-in replacement for the OpenAI API.**

## Benchmarks

Tested on Apple M-series with `mlx-community/Llama-3.2-1B-Instruct-4bit`:

| Metric | Result |
|---|---|
| Cold start | 16 ms |
| Model load (cached) | 2.4 s |
| Idle memory (RSS) | 8 MB |
| Loaded memory (RSS) | 380 MB |
| Throughput (streaming) | 115–261 tok/s |
| Time to first token | 86–96 ms |
| 4× concurrent requests | 0.37 s wall, 0 errors |

## Requirements

- macOS + Apple Silicon (M1 or later)
- Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Python 3.13: `brew install python@3.13`
- [uv](https://github.com/astral-sh/uv): `brew install uv`

## Quick start

```bash
git clone https://github.com/Ar9av/mlx-lm-server
cd mlx-lm-server
./run.sh
```

`run.sh` creates the venv, installs `mlx-lm`, builds the release binary, and starts the server. First run takes ~2 minutes.

Or manually:

```bash
uv venv .venv --python python3.13
uv pip install mlx-lm --python .venv/bin/python

PYO3_PYTHON="$(pwd)/.venv/bin/python" cargo build --release

PYTHONPATH=".venv/lib/python3.13/site-packages" \
  VIRTUAL_ENV=".venv" \
  ./target/release/mlx-lm-server
```

Default: `http://localhost:8000`

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MLX_HOST` | `0.0.0.0` | Listen address |
| `MLX_PORT` | `8000` | Listen port |
| `MLX_DEFAULT_MODEL` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | Auto-load on first request |
| `MLX_DEFAULT_MAX_TOKENS` | `2048` | Max tokens to generate |
| `MLX_DEFAULT_TEMPERATURE` | `0.7` | Sampling temperature |
| `MLX_DEFAULT_TOP_P` | `0.9` | Top-p nucleus sampling |
| `MLX_MAX_CONCURRENT` | `1` | Max parallel inference requests |
| `MLX_STREAM_TIMEOUT` | `120.0` | SSE stream timeout (seconds) |
| `MLX_MAX_MODEL_SIZE_GB` | _(none)_ | Reject model loads above this size |
| `MLX_ALLOWED_MODELS` | _(none)_ | Comma-separated model ID allowlist |
| `MLX_CORS_ORIGINS` | `http://localhost:3000,…` | Allowed CORS origins |
| `MLX_DEBUG` | `false` | Verbose request logging |

## API

### Inference

#### `POST /v1/chat/completions`
OpenAI-compatible chat. Supports streaming (`"stream": true` → SSE).

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "What is MLX?"}],
    "stream": true
  }'
```

Extra fields beyond the OpenAI spec:

| Field | Type | Description |
|---|---|---|
| `kv_bits` | int | KV-cache quantization (4 or 8) — up to 62% faster decode at long context |
| `kv_group_size` | int | KV quant group size (default 64) |
| `adapter_name` | string | Use a named mounted LoRA adapter for this request |

#### `POST /v1/completions`
Legacy text completion. Same extra fields as chat.

#### `POST /v1/embeddings`
Mean-pooled, L2-normalized embeddings from the loaded model's embedding layer.

```bash
curl http://localhost:8000/v1/embeddings \
  -d '{"model": "...", "input": ["hello world", "foo bar"]}'
```

#### `POST /v1/messages`
Anthropic Messages API. Supports `stream: true` with Anthropic SSE event sequence.

```bash
curl http://localhost:8000/v1/messages \
  -d '{"model": "...", "messages": [...], "max_tokens": 256}'
```

#### `POST /v1/tokenize`
```bash
curl http://localhost:8000/v1/tokenize -d '{"prompt": "Hello, world!"}'
# → {"tokens": [9906, 11, 1917, 0], "count": 4}
```

#### `POST /v1/benchmark`
Run N inference passes and return latency statistics.

```bash
curl -X POST http://localhost:8000/v1/benchmark \
  -d '{"prompt": "Write a poem.", "runs": 5, "max_tokens": 64}'
# → {"ttft_ms_p50": 88.2, "tokens_per_sec_mean": 217.4, ...}
```

### Model management

#### `GET /v1/models`
List all locally cached models. Currently loaded model appears first.

#### `POST /v1/models/load`
Load a model (downloads from HuggingFace if not cached). Checks available RAM before loading.

```bash
# Basic load
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model": "mlx-community/Llama-3.2-3B-Instruct-4bit"}'

# With LoRA adapter
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model": "...", "adapter": "/path/to/adapter"}'

# With speculative decoding drafter (+1.4× throughput)
curl -X POST http://localhost:8000/v1/models/load \
  -d '{"model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
       "drafter": "mlx-community/Llama-3.2-1B-Instruct-4bit"}'
```

#### `GET /v1/models/:id/info`
Disk size, quantization, architecture, context length, vision capability, and load status for a cached model.

#### `DELETE /v1/models/*model_id`
Unload the current model and clear the MLX cache.

### LoRA adapters

Hot-swap up to 4 co-resident LoRA adapters per-request without reloading the base model.

```bash
# Mount an adapter
curl -X POST http://localhost:8000/v1/adapters/mount \
  -d '{"name": "coding", "adapter_path": "/path/to/lora"}'

# List mounted adapters
curl http://localhost:8000/v1/adapters

# Use an adapter for a specific request
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"messages": [...], "adapter_name": "coding"}'

# Unmount
curl -X DELETE http://localhost:8000/v1/adapters/coding
```

### Vision (multimodal)

Pass images via `image_url` content parts. Requires `pip install mlx-vlm` and a vision-capable model.

```json
{
  "model": "mlx-community/llava-1.5-7b-mlx",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "What's in this image?"},
      {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
    ]
  }]
}
```

`data:image/jpeg;base64,...` URLs are also accepted. Returns HTTP 501 with install instructions if `mlx_vlm` is not installed.

### Discovery & utilities

```bash
# Local HuggingFace cache
curl http://localhost:8000/api/models/local

# Delete a cached model from disk
curl -X DELETE http://localhost:8000/api/models/local/mlx-community/Llama-3.2-3B-Instruct-4bit

# Search mlx-community on HuggingFace Hub
curl "http://localhost:8000/api/huggingface/models?search=mistral&limit=10"

# Process info (model, memory, pid)
curl http://localhost:8000/api/ps

# Machine-readable API reference (for AI assistants)
curl http://localhost:8000/llms.txt
```

## Client examples

### OpenAI SDK (Python)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

for chunk in client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Anthropic SDK (Python)

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="unused")

with client.messages.stream(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}],
) as stream:
    print(stream.get_final_message().content[0].text)
```

### Claude Code

```bash
export ANTHROPIC_BASE_URL="http://localhost:8000"
claude  # uses your local model instead of Anthropic's API
```

## Architecture

```
src/
├── main.rs           axum router, middleware, startup
├── config.rs         env-var configuration
├── error.rs          MlxError → HTTP response mapping
├── models.rs         request/response types (OpenAI + Anthropic)
├── state.rs          AppState (Arc<Config> + MlxService + Semaphore)
├── mlx_service.rs    PyO3 bridge: load, generate, stream, embeddings, vision
└── routes/
    ├── adapters.rs   LoRA adapter registry
    ├── anthropic.rs  POST /v1/messages
    ├── benchmark.rs  POST /v1/benchmark
    ├── chat.rs       POST /v1/chat/completions, /v1/tokenize
    ├── completions.rs POST /v1/completions
    ├── embeddings.rs  POST /v1/embeddings
    ├── health.rs     GET /health /status /llms.txt
    └── models.rs     model management + HF search + local cache
```

Inference runs in `tokio::task::spawn_blocking` threads (Python GIL held). Tokens stream from Python generator → `mpsc` channel → `ReceiverStream` → SSE body without blocking the async runtime. A `Semaphore` gates concurrent inference up to `MLX_MAX_CONCURRENT`.

## License

MIT
