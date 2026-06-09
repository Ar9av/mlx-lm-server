# mlx-audio-server

OpenAI-compatible audio inference server for Apple Silicon — TTS, STT, and source separation powered by [mlx-audio](https://github.com/Blaizzy/mlx-audio) and [MLX](https://github.com/ml-explore/mlx).

Runs entirely on-device. The Rust server embeds Python via PyO3 and calls into `mlx-audio` over the GIL — no subprocess, no network calls to external APIs.

## Features

- **TTS** (`POST /v1/audio/speech`) — OpenAI-compatible. Streams chunked WAV or returns a full audio file.
- **STT** (`POST /v1/audio/transcriptions`) — Whisper-family models. Returns JSON transcript with optional word/segment timestamps.
- **Translation** (`POST /v1/audio/translations`) — STT with forced English output.
- **Source separation** (`POST /v1/audio/separations`) — Separate a target sound from a mixture using a text description.
- **Model management** (`GET/POST/DELETE /v1/models`) — Hot-load/unload TTS, STT, or STS models at runtime.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Rust 1.75+
- Python 3.12 or 3.13 (not 3.14 — PyO3 0.22 maximum)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio): `pip install mlx-audio`

## Quick start

```bash
git clone https://github.com/Ar9av/mlx-audio-server
cd mlx-audio-server
./run.sh
```

Or manually:

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install mlx-audio numpy

export PYO3_PYTHON="$(pwd)/.venv/bin/python"
cargo build --release
./target/release/mlx-audio-server
```

## Configuration

All options via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MLX_AUDIO_HOST` | `0.0.0.0` | Bind address |
| `MLX_AUDIO_PORT` | `8001` | Port |
| `MLX_AUDIO_TTS_MODEL` | `mlx-community/Kokoro-82M-bf16` | Default TTS model |
| `MLX_AUDIO_STT_MODEL` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | Default STT model |
| `MLX_AUDIO_DEFAULT_VOICE` | `af_heart` | Default TTS voice |
| `MLX_AUDIO_DEFAULT_SPEED` | `1.0` | Default TTS speed |
| `MLX_AUDIO_MAX_CONCURRENT` | `1` | Max concurrent inference requests |
| `MLX_AUDIO_MAX_TTS_CHARS` | `10000` | Max input characters for TTS |
| `MLX_AUDIO_DEBUG` | `false` | Verbose logging |

## API

### TTS — `POST /v1/audio/speech`

OpenAI-compatible. Drop-in for `client.audio.speech.create(...)`.

```bash
curl http://localhost:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "kokoro", "input": "Hello from Apple Silicon!", "voice": "af_heart"}' \
  --output speech.wav
```

**Streaming:**
```bash
curl http://localhost:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "kokoro", "input": "Streaming audio...", "stream": true}' \
  --output stream.wav
```

**Custom voice clone:**
```json
{
  "model": "kokoro",
  "input": "Hello world",
  "ref_audio": "/path/to/reference.wav",
  "ref_text": "The text spoken in the reference audio"
}
```

### STT — `POST /v1/audio/transcriptions`

OpenAI-compatible multipart form.

```bash
curl http://localhost:8001/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3
```

With timestamps:
```bash
curl http://localhost:8001/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F timestamp_granularities[]=segment
```

### Translation — `POST /v1/audio/translations`

Same as transcriptions but forces English output.

### Source separation — `POST /v1/audio/separations`

```bash
curl http://localhost:8001/v1/audio/separations \
  -F file=@mixed_audio.wav \
  -F description="speech"
```

Response:
```json
{
  "target_url": "data:audio/wav;base64,...",
  "residual_url": "data:audio/wav;base64,...",
  "message": "Separation complete"
}
```

### Model management

```bash
# List loaded models
curl http://localhost:8001/v1/models

# Load a model
curl -X POST http://localhost:8001/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"model": "mlx-community/whisper-large-v3-turbo-asr-fp16", "type": "stt"}'

# Unload
curl -X DELETE http://localhost:8001/v1/models/stt
```

### Health

```bash
curl http://localhost:8001/health
```

## Python client example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8001/v1", api_key="local")

# TTS
with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="af_heart",
    input="Hello from mlx-audio-server!",
) as response:
    response.stream_to_file("output.wav")

# STT
with open("audio.wav", "rb") as f:
    transcript = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=f,
    )
print(transcript.text)
```

## Architecture

```
HTTP request
    │
    ▼
axum router (async, Tokio)
    │
    ▼
Semaphore permit (max_concurrent)
    │
    ▼
tokio::task::spawn_blocking
    │
    ▼
Python::with_gil → mlx_audio.{tts,stt,sts}
    │
    ▼
audio bytes → response body
```

The Python GIL is only held inside `spawn_blocking` threads, keeping the async runtime unblocked during inference.

## License

MIT
