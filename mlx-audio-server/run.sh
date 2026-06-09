#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"

if [[ ! -d "$VENV" ]]; then
  echo "Creating Python venv at $VENV ..."
  python3.13 -m venv "$VENV" || python3.12 -m venv "$VENV" || python3 -m venv "$VENV"
fi

PYTHON="$VENV/bin/python"

echo "Installing Python dependencies ..."
"$PYTHON" -m ensurepip --quiet 2>/dev/null || true
"$PYTHON" -m pip install -q --upgrade pip
"$PYTHON" -m pip install -q mlx-audio numpy misaki num2words spacy phonemizer

# Derive site-packages path
SITE_PKGS="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"

export PYO3_PYTHON="$PYTHON"
export PYTHONPATH="$SITE_PKGS"

echo "Building mlx-audio-server ..."
cd "$REPO_ROOT"
cargo build --release -p mlx-audio-server

echo "Starting server on http://0.0.0.0:${MLX_AUDIO_PORT:-8001} ..."
exec "$REPO_ROOT/target/release/mlx-audio-server" "$@"
