# mlx-local-server

A monorepo of OpenAI-compatible inference servers for Apple Silicon, written in Rust. Both servers embed Python via [PyO3](https://pyo3.rs) — Metal acceleration stays in Python, everything else (HTTP, concurrency, streaming) runs natively in Rust.

| Server | What it does | Default port |
|---|---|---|
| [`mlx-lm-server`](./mlx-lm-server) | LLM chat completions, Anthropic compat, LoRA adapters, vision | `8080` |
| [`mlx-audio-server`](./mlx-audio-server) | TTS, STT, audio translation, source separation | `8001` |

**Single binaries. ~8 MB idle RSS each. Drop-in replacement for OpenAI API.**

> Built with [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) — Apple's open-source ML framework for Apple Silicon. Showcased at [WWDC 2025 — Build local AI agents on Mac with MLX](https://developer.apple.com/videos/play/wwdc2025/).

---

## Quick start

```bash
# LLM server  →  http://localhost:8080
./run.sh lm

# Audio server  →  http://localhost:8001
./run.sh audio

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
- **Benchmarking** (`POST /v1/benchmark`) — TTFT/tps percentiles
- **Model info** (`GET /v1/models/:id/info`) — scans HF cache, reads config.json
- **RAM guard** — rejects loads that would exceed available memory

### Sampler parameters

All generation endpoints accept the full set of mlx-lm sampling params:

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

# Chat with sampler params
curl http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama","messages":[{"role":"user","content":"Hello"}],
       "temperature":0.8,"top_k":50,"repetition_penalty":1.1}'

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
```

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
