#!/usr/bin/env bash
# Usage: ./run.sh <lm|audio> [server-flags...]
#
# Environment overrides:
#   VENV=<path>   — Python venv to use/create  (default: .venv at repo root)
#   BUILD=1       — force a release rebuild even if the binary already exists
#
# Examples:
#   ./run.sh lm
#   ./run.sh audio --tts-model mlx-community/Kokoro-82M-v1.0-bf16
#   BUILD=1 ./run.sh lm
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"

MODE="${1:-}"
case "$MODE" in
  lm|audio) shift ;;
  *)
    echo "Usage: $0 <lm|audio> [server-flags...]"
    echo ""
    echo "  lm     — OpenAI-compatible chat / completions server  (port 8080)"
    echo "  audio  — TTS / STT / STS server                       (port 8001)"
    echo ""
    echo "Environment:"
    echo "  VENV=<path>   override venv location (default: \$REPO_ROOT/.venv)"
    echo "  BUILD=1       force a release rebuild"
    exit 1
    ;;
esac

# ── Python venv ───────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  echo "[setup] Creating Python venv at $VENV ..."
  if command -v uv &>/dev/null; then
    uv venv "$VENV" --python python3.13
  else
    python3.13 -m venv "$VENV" 2>/dev/null \
      || python3.12 -m venv "$VENV" 2>/dev/null \
      || python3 -m venv "$VENV"
  fi
fi

PYTHON="$VENV/bin/python"
"$PYTHON" -m ensurepip -q 2>/dev/null || true

# ── deps + crate selection ────────────────────────────────────────────────────
if [[ "$MODE" == "lm" ]]; then
  echo "[setup] Installing lm deps ..."
  "$PYTHON" -m pip install -q --upgrade pip
  "$PYTHON" -m pip install -q mlx-lm
  CRATE="mlx-lm-server"
  BINARY="mlx-lm-server"
else
  echo "[setup] Installing audio deps ..."
  "$PYTHON" -m pip install -q --upgrade pip
  "$PYTHON" -m pip install -q mlx-audio numpy misaki num2words spacy phonemizer
  CRATE="mlx-audio-server"
  BINARY="mlx-audio-server"
fi

# ── build ─────────────────────────────────────────────────────────────────────
SITE_PKGS="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
BINARY_PATH="$REPO_ROOT/target/release/$BINARY"

if [[ ! -f "$BINARY_PATH" || "${BUILD:-0}" == "1" ]]; then
  echo "[build] Building $CRATE (release) ..."
  PYO3_PYTHON="$PYTHON" \
    cargo build --release -p "$CRATE" 2>&1
fi

# ── run ───────────────────────────────────────────────────────────────────────
echo "[start] $BINARY"
exec env \
  PYTHONPATH="$SITE_PKGS" \
  VIRTUAL_ENV="$VENV" \
  PYO3_PYTHON="$PYTHON" \
  "$BINARY_PATH" "$@"
