# agents.md — codebase navigation for AI agents

## Repo layout

```
mlx-local-server/               ← Cargo workspace root
├── Cargo.toml                  ← workspace manifest + shared [workspace.dependencies]
├── run.sh                      ← unified entry point: ./run.sh lm|audio [flags...]
├── examples/
│   ├── sts_demo.py             ← live mic → STS demo (requires mlx-audio-server on :8001)
│   └── bench_lm.py             ← throughput / TTFT / concurrency benchmark for mlx-lm-server
├── mlx-lm-server/              ← Cargo crate: OpenAI-compatible chat/completion server
│   ├── Cargo.toml
│   ├── run.sh                  ← thin wrapper: exec ../../run.sh lm "$@"
│   └── src/
│       ├── main.rs             ← axum router, middleware wiring
│       ├── config.rs           ← Config::from_env() — all MLX_* env vars
│       ├── state.rs            ← AppState { mlx: MlxService, config, semaphore }
│       ├── mlx_service.rs      ← PyO3 bridge: load/generate/embed via mlx_lm Python
│       ├── models.rs           ← request/response types (serde)
│       ├── error.rs            ← AppError + IntoResponse impl
│       └── routes/
│           ├── mod.rs
│           ├── health.rs       ← GET /  /health  /status  /llms.txt
│           ├── chat.rs         ← POST /v1/chat/completions  /v1/tokenize
│           ├── completions.rs  ← POST /v1/completions
│           ├── embeddings.rs   ← POST /v1/embeddings
│           ├── models.rs       ← GET/POST/DELETE /v1/models  /api/models/local  /api/huggingface/models
│           ├── adapters.rs     ← LoRA adapter management
│           ├── anthropic.rs    ← POST /v1/messages (Anthropic-compatible)
│           └── benchmark.rs    ← POST /v1/benchmark
└── mlx-audio-server/           ← Cargo crate: TTS / STT / STS server
    ├── Cargo.toml              ← adds axum multipart feature
    ├── run.sh                  ← thin wrapper: exec ../../run.sh audio "$@"
    └── src/
        ├── main.rs             ← axum router
        ├── config.rs           ← Config::from_env() — all MLX_AUDIO_* env vars
        ├── state.rs            ← AppState { audio: AudioService, config, semaphore }
        ├── audio_service.rs    ← PyO3 bridge: TTS / STT / STS via mlx_audio Python
        ├── models.rs           ← request/response types
        ├── error.rs            ← AppError
        └── routes/
            ├── mod.rs
            ├── health.rs       ← GET /health
            ├── models.rs       ← GET/POST/DELETE /v1/models
            ├── speech.rs       ← POST /v1/audio/speech  (TTS)
            ├── transcriptions.rs ← POST /v1/audio/transcriptions  (STT)
            ├── translations.rs ← POST /v1/audio/translations  (STT → EN)
            └── separations.rs  ← POST /v1/audio/separations  (STS)
```

## Build & run

```bash
# first run: creates .venv, installs deps, builds, starts
./run.sh lm               # mlx-lm-server  on :8000
./run.sh audio            # mlx-audio-server on :8001

# force rebuild after Rust changes
BUILD=1 ./run.sh lm

# custom venv
VENV=/tmp/myvenv ./run.sh audio

# both can run simultaneously (different ports)
```

Build requires:
- Rust stable (1.75+) — `rustup toolchain install stable`
- Python 3.13 (or 3.12) — `brew install python@3.13`
- `uv` optional but preferred for faster venv creation

The `run.sh` sets `PYO3_PYTHON` (build-time) and `PYTHONPATH`/`VIRTUAL_ENV` (runtime) automatically.

## Environment variables

### mlx-lm-server

| Variable | Default | Purpose |
|---|---|---|
| `MLX_PORT` | `8000` | listen port |
| `MLX_HOST` | `0.0.0.0` | bind address |
| `MLX_DEFAULT_MODEL` | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | model loaded on startup |
| `MLX_DEFAULT_MAX_TOKENS` | `2048` | token cap per request |
| `MLX_DEFAULT_TEMPERATURE` | `0.7` | |
| `MLX_DEFAULT_TOP_P` | `0.9` | |
| `MLX_MAX_CONCURRENT` | `1` | semaphore permits (LLM is not concurrent-safe) |
| `MLX_MAX_MODEL_SIZE_GB` | unset | refuse models larger than N GB |
| `MLX_ALLOWED_MODELS` | unset | comma-separated allowlist |
| `MLX_CORS_ORIGINS` | `localhost:5173,localhost:3000` | comma-separated |
| `MLX_STREAM_TIMEOUT` | `120.0` | SSE idle timeout (s) |
| `MLX_DEBUG` | `false` | verbose tracing |

### mlx-audio-server

| Variable | Default | Purpose |
|---|---|---|
| `MLX_AUDIO_PORT` | `8001` | listen port |
| `MLX_AUDIO_HOST` | `0.0.0.0` | |
| `MLX_AUDIO_TTS_MODEL` | `mlx-community/Kokoro-82M-bf16` | |
| `MLX_AUDIO_STT_MODEL` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | |
| `MLX_AUDIO_DEFAULT_VOICE` | `af_heart` | Kokoro voice name |
| `MLX_AUDIO_DEFAULT_SPEED` | `1.0` | |
| `MLX_AUDIO_DEFAULT_LANGUAGE` | `en` | STT language hint |
| `MLX_AUDIO_MAX_TTS_CHARS` | `10000` | input length cap |
| `MLX_AUDIO_MAX_CONCURRENT` | `1` | |
| `MLX_AUDIO_DEBUG` | `false` | |

## API surface

### mlx-lm-server (:8000)

```
GET  /health
GET  /status
GET  /llms.txt
POST /v1/chat/completions        { model, messages, stream, max_tokens, temperature, top_p }
POST /v1/completions             { model, prompt, stream, max_tokens }
POST /v1/embeddings              { model, input }
POST /v1/tokenize                { model, messages }
GET  /v1/models
POST /v1/models/load             { model: "hf-repo-id" }
GET  /v1/models/:model_id/info
DEL  /v1/models/*model_id
GET  /v1/adapters
POST /v1/adapters/mount          { name, path }
DEL  /v1/adapters/:name
POST /v1/messages                (Anthropic messages API)
GET  /api/models/local
DEL  /api/models/local/:org/*model
GET  /api/huggingface/models     ?q=<query>
GET  /api/ps
POST /v1/benchmark               { model, prompt, n_tokens }
```

### mlx-audio-server (:8001)

```
GET  /health                     → { tts_model, stt_model, sts_model }
GET  /v1/models
POST /v1/models/load             { model_type: "tts"|"stt"|"sts", model: "hf-repo-id" }
DEL  /v1/models/:type
POST /v1/audio/speech            { model?, input, voice?, speed?, response_format? }
                                 → audio/wav (streaming)
POST /v1/audio/transcriptions    multipart: file=<audio>, model?, language?, temperature?
                                 → { text }
POST /v1/audio/translations      multipart: file=<audio>  (STT → English)
                                 → { text }
POST /v1/audio/separations       multipart: file=<audio>, description?
                                 → { target_url: "data:audio/wav;base64,…",
                                     residual_url: "data:audio/wav;base64,…" }
```

## Architecture: PyO3 bridge pattern

Both servers embed CPython via PyO3. The pattern used in every inference call:

```rust
let result = tokio::task::spawn_blocking(move || {
    Python::with_gil(|py| {
        // 1. Init Metal GPU stream for this thread (required by MLX)
        let mx = py.import("mlx.core")?;
        mx.call_method1("eval", (mx.call_method1("zeros", (1_i32,))?,))?;

        // 2. Call Python model
        let output = model.call_method1(py, "generate", (args,))?;
        Ok(output)
    })
}).await??;
```

**Critical invariants:**
- MLX Metal streams are thread-local. Every `spawn_blocking` thread **must** call `mx.eval(mx.zeros(1))` before touching model tensors — otherwise `RuntimeError: There is no Stream(gpu, N) in current thread`.
- `PYTHONPATH` must be set to the venv `site-packages` at process startup. PyO3's embedded Python does not inherit venv site-packages automatically even if `PYO3_PYTHON` is correct.
- `NamedTempFile` must be kept alive until after Python reads the file. `into_temp_path()` drops the file immediately on scope exit; keep the `NamedTempFile` binding alive instead.

## Key files to edit for common tasks

| Task | File |
|---|---|
| Add an LLM endpoint | `mlx-lm-server/src/routes/` + wire in `main.rs` |
| Change LLM generation params | `mlx-lm-server/src/mlx_service.rs` |
| Add an audio endpoint | `mlx-audio-server/src/routes/` + wire in `main.rs` |
| Change TTS/STT/STS Python calls | `mlx-audio-server/src/audio_service.rs` |
| Add a new env-var config field | `{crate}/src/config.rs` |
| Change default models | `{crate}/src/config.rs` (env var defaults) |
| Add a shared Rust dependency | root `Cargo.toml` `[workspace.dependencies]`, then `dep = { workspace = true }` in crate's `Cargo.toml` |
| Add a Python demo / test script | `examples/` |

## Python model details

- **TTS**: `mlx_audio.tts` — Kokoro model. `model.generate(text, voice=voice, lang_code=lang_code, verbose=False)` returns `(audio_array, sample_rate)`. `lang_code` must be inferred from voice prefix (`af_`/`am_` → `"a"`, `bf_`/`bm_` → `"b"`) — do not rely on system `espeak` binary.
- **STT**: `mlx_audio.stt` — Whisper family. `model.generate(path, language=lang, temperature=0.0)` returns transcript string.
- **STS**: `mlx_audio.sts` — SAM-Audio. `model.separate_long([str_path], descriptions=[desc], chunk_seconds=10.0, verbose=False)` returns `[(target_array, sr), (residual_array, sr)]`. Requires 48 kHz input. Needs mixed audio (voice + background) — pure silence has nothing to separate.
- **LM**: `mlx_lm` — `mlx_lm.load(model_id)` returns `(model, tokenizer)`. Generation via `mlx_lm.generate(model, tokenizer, prompt, max_tokens=N)`.

## Non-obvious gotchas

1. **WAV encoding in Rust**: The audio server builds RIFF WAV headers manually in Rust (not via Python's `soundfile`/`struct.pack`) because passing a Rust `Vec<i16>` to Python `struct.pack` doesn't work across the PyO3 boundary.
2. **Streaming WAV uses `u32::MAX` sentinels** for `chunk_size` and `data_len` fields (standard trick for unknown-length streams). Do not compute `u32::MAX + 36` — it overflows in release mode.
3. **`axum` multipart** is only needed by `mlx-audio-server`. It is declared as `axum = { workspace = true, features = ["multipart"] }` in `mlx-audio-server/Cargo.toml` on top of the workspace base.
4. **Concurrent model inference is serialised** via `tokio::sync::Semaphore` with `max_concurrent=1` default. MLX models are not thread-safe; do not change this without testing.
5. **HuggingFace model downloads** use `snapshot_download()` from Python. If a download is interrupted, `.incomplete` blob files may be left in `~/.cache/huggingface/hub/` and the next attempt will fail with `FileNotFoundError`. Delete the `.incomplete` file to recover.
