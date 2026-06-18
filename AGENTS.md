# AGENTS.md

This file is the operational playbook for any coding agent working in this repo.

Use it for:
- checking whether the machine can run the project
- starting the servers correctly
- verifying that the servers are healthy
- making changes safely in a dirty worktree
- publishing changes back to the repo

## Repository Summary

`mlx-local-server` is a Rust workspace with Python-backed MLX inference via PyO3.

Primary entrypoint:

```bash
./run.sh <lm|audio|image>
```

Main services:
- `lm` -> OpenAI-compatible LLM server
- `audio` -> TTS / STT / STS server
- `image` -> image generation server

Important repo docs:
- `AGENTS.md` -> agent setup, safety, startup, verification, and repo map
- `README.md` -> product-level usage docs

## Preflight Checks

Run these before changing or starting anything:

```bash
pwd
git status --short --branch
uname -m
sw_vers
rustc --version
cargo --version
python3 --version
python3.12 --version || true
python3.13 --version || true
command -v uv || true
```

Expected environment:
- Apple Silicon Mac: `uname -m` should be `arm64`
- macOS host
- Rust stable present
- Python 3.12 or 3.13 available
- `uv` is optional, but preferred

Notes:
- `run.sh` creates `.venv` automatically if it does not exist.
- `run.sh` installs the required Python packages for the selected mode automatically.
- First model load may download large artifacts from Hugging Face.

## Safe Working Rules

Before editing:
- inspect `git status --short`
- assume the worktree may already contain unrelated user files
- do not delete or revert unrelated changes
- do not remove cached models, benchmark files, or generated artifacts unless explicitly asked

Current branch model for this repo:
- the default branch is `master`
- there is no `main` branch at the time of writing

If a task says "push to main", interpret that as "publish to the default branch", unless the repo adds `main` later.

## Boot Flow

Use the unified launcher from the repo root.

### LLM server

```bash
./run.sh lm
```

Useful variants:

```bash
BUILD=1 ./run.sh lm
VENV=/tmp/mlx-local-server-venv ./run.sh lm
MLX_PORT=8000 ./run.sh lm
```

### Audio server

```bash
./run.sh audio
```

Useful variants:

```bash
BUILD=1 ./run.sh audio
VENV=/tmp/mlx-local-server-venv ./run.sh audio
```

### Image server

```bash
./run.sh image
```

Useful variants:

```bash
BUILD=1 ./run.sh image
VENV=/tmp/mlx-local-server-venv ./run.sh image
```

## Runtime Behavior

`run.sh` will:
1. create a Python venv if needed
2. install Python dependencies for the selected server
3. build the Rust binary if needed
4. set `PYO3_PYTHON`, `PYTHONPATH`, and `VIRTUAL_ENV`
5. execute the selected server binary

Do not bypass `run.sh` unless you specifically need a manual debug path.

## Default Ports

Trust the Rust config files over stale docs.

- LLM server default: `8000`
- Audio server default: `8001`
- Image server default: `8002`

## Health Checks

After startup, verify the relevant endpoint from another terminal.

### LLM

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/status
curl -s http://localhost:8000/v1/models
```

Minimal functional test:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Mistral-7B-Instruct-v0.3-4bit","messages":[{"role":"user","content":"Reply with exactly: ok"}],"max_tokens":8}'
```

### Audio

```bash
curl -s http://localhost:8001/health
curl -s http://localhost:8001/v1/models
```

Minimal TTS smoke test:

```bash
curl -s http://localhost:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"test","voice":"af_heart"}' \
  --output /tmp/mlx-agent-tts.wav
```

### Image

```bash
curl -s http://localhost:8002/health
curl -s http://localhost:8002/v1/models
```

Minimal generation smoke test:

```bash
curl -s http://localhost:8002/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a simple black square on a white background","size":"512x512"}'
```

## Build And Validation

Use these checks after code changes:

```bash
cargo fmt --all
cargo check --workspace
cargo test --workspace
```

If PyO3 binding resolution matters for the build, use:

```bash
PYO3_PYTHON=$(which python3.12) cargo check --workspace
```

If only Python-side runtime behavior changed, also rerun the relevant `./run.sh` smoke test.

## Common Failure Modes

### Python package import failures

Symptoms:
- PyO3 import errors
- missing `mlx_lm`, `mlx_audio`, or `mflux`

Actions:
- rerun with `./run.sh <mode>`
- if needed, remove only the repo venv and recreate it:

```bash
rm -rf .venv
./run.sh lm
```

### Wrong Python selected

Check:

```bash
echo "$PYO3_PYTHON"
```

Preferred fix:
- let `run.sh` manage Python selection
- for a forced rebuild, use `BUILD=1 ./run.sh <mode>`

### First-run model download latency

Symptoms:
- slow first request
- Hugging Face download activity

Interpretation:
- normal on first use
- do not treat as a failure unless the process exits or the HTTP endpoint never becomes healthy

### Hugging Face incomplete cache files

If a model download was interrupted, retry may fail because of `.incomplete` files in the Hugging Face cache. Clear the broken incomplete file only if you confirm that specific failure mode.

## File Map For Edits

Common locations:
- LLM routes: `mlx-lm-server/src/routes/`
- LLM service bridge: `mlx-lm-server/src/mlx_service.rs`
- Audio routes: `mlx-audio-server/src/routes/`
- Audio service bridge: `mlx-audio-server/src/audio_service.rs`
- Image routes: `mlx-image-server/src/routes/`
- shared workspace deps: `Cargo.toml`
- launcher: `run.sh`

Key structures:
- `mlx-lm-server/src/main.rs` -> LLM router and middleware wiring
- `mlx-lm-server/src/config.rs` -> `MLX_*` env var parsing
- `mlx-lm-server/src/state.rs` -> app state and concurrency semaphore
- `mlx-audio-server/src/main.rs` -> audio router
- `mlx-audio-server/src/config.rs` -> `MLX_AUDIO_*` env var parsing
- `mlx-audio-server/src/state.rs` -> audio app state
- `mlx-image-server/src/main.rs` -> image router
- `mlx-image-server/src/config.rs` -> `MLX_IMAGE_*` env var parsing

## Architecture Notes

All three servers are Rust HTTP frontends over Python MLX libraries via PyO3.

Critical invariants:
- MLX Metal streams are thread-local
- every `spawn_blocking` inference thread must initialize MLX before touching model tensors
- `PYTHONPATH` must point at the venv site-packages at process startup
- temporary audio files must remain alive until Python has read them

Operational implication:
- prefer editing server behavior in the Rust service bridge files, not in ad hoc wrapper code
- prefer launching through `run.sh`, because it sets the Python environment correctly

## Publish Flow

Use a narrow commit. Do not include unrelated untracked artifacts unless the user asked for them.

Inspect:

```bash
git status --short
git diff -- AGENTS.md
```

Stage and commit:

```bash
git add AGENTS.md
git commit -m "Add agent operating guide"
```

Publish to the repo default branch:

```bash
git push origin master
```

If the repository later adopts `main`, re-check with:

```bash
git branch -a
git remote show origin
```

## Minimal Agent Routine

For most tasks, follow this sequence:
1. read `AGENTS.md`
2. inspect `git status --short --branch`
3. run preflight checks
4. start the relevant service with `./run.sh`
5. verify with the matching health endpoint
6. make the requested change
7. run `cargo fmt --all` and `cargo check --workspace`
8. run the smallest relevant smoke test
9. commit only the intended files
10. push to `origin master`
