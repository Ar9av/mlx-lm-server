# mlx-local-server

A monorepo of OpenAI-compatible inference servers for Apple Silicon, written in Rust. Both servers embed Python via [PyO3](https://pyo3.rs) — Metal acceleration stays in Python, everything else (HTTP, concurrency, streaming) runs natively in Rust.

| Server | What it does | Default port |
|---|---|---|
| [`mlx-lm-server`](./mlx-lm-server) | LLM chat completions, Anthropic compat, LoRA adapters, vision | `8080` |
| [`mlx-audio-server`](./mlx-audio-server) | TTS, STT, audio translation, source separation | `8001` |

**Single binaries. ~8 MB idle RSS each. Drop-in replacement for OpenAI API.**

---

## mlx-lm-server

OpenAI-compatible LLM inference powered by [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms).

### Quick start

```bash
cd mlx-lm-server
./run.sh
```

Or:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install mlx-lm mlx-vlm

export PYO3_PYTHON="$(pwd)/.venv/bin/python"
cargo build --release -p mlx-lm-server
./target/release/mlx-lm-server
```

### Features

- **OpenAI chat completions** (`POST /v1/chat/completions`) — streaming SSE + sync
- **Anthropic messages** (`POST /v1/messages`) — streaming + sync
- **Text completions** (`POST /v1/completions`)
- **Embeddings** (`POST /v1/embeddings`)
- **Vision** — routes image_url messages to `mlx_vlm` automatically
- **LoRA adapter hot-swap** (`GET/POST/DELETE /v1/adapters`)
- **Speculative decoding** — pass `drafter` in load request
- **KV-cache quantization** — `kv_bits` + `kv_group_size` per request
- **Benchmarking** (`POST /v1/benchmark`) — TTFT/tps percentiles
- **Model info** (`GET /v1/models/:id/info`) — scans HF cache, reads config.json
- **RAM guard** — rejects loads that would exceed available memory

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

# Chat
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Mount a LoRA adapter
curl -X POST http://localhost:8080/v1/adapters/mount \
  -H 'Content-Type: application/json' \
  -d '{"name":"my-lora","adapter_path":"/path/to/adapter","model":"mlx-community/Llama-3.2-3B-Instruct-4bit"}'
```

---

## mlx-audio-server

OpenAI-compatible audio inference powered by [mlx-audio](https://github.com/Blaizzy/mlx-audio).

### Quick start

```bash
cd mlx-audio-server
./run.sh
```

Or:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install mlx-audio numpy misaki num2words spacy phonemizer

export PYO3_PYTHON="$(pwd)/.venv/bin/python"
cargo build --release -p mlx-audio-server
./target/release/mlx-audio-server
```

### Features

- **TTS** (`POST /v1/audio/speech`) — Kokoro, streaming chunked WAV or full file, voice cloning via `ref_audio`
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

# Source separation
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

## Building both

```bash
# Check both
PYO3_PYTHON=/path/to/.venv/bin/python cargo check --workspace

# Build both release binaries
PYO3_PYTHON=/path/to/.venv/bin/python cargo build --release --workspace
# → target/release/mlx-lm-server
# → target/release/mlx-audio-server
```

## Repo layout

```
mlx-local-server/
├── mlx-lm-server/        LLM inference server
│   ├── src/
│   ├── Cargo.toml
│   └── run.sh
├── mlx-audio-server/     Audio inference server
│   ├── src/
│   ├── Cargo.toml
│   └── run.sh
├── Cargo.toml            Workspace manifest + shared deps
└── Cargo.lock
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Rust 1.75+
- Python 3.12 or 3.13 (PyO3 0.22 max is 3.13)

## License

MIT
