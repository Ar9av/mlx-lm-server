#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
CARGO="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo"

if [ ! -d "$VENV" ]; then
  echo "Creating Python venv..."
  uv venv "$VENV" --python python3.13
  uv pip install mlx-lm --python "$VENV/bin/python"
fi

if [ ! -f "$SCRIPT_DIR/target/release/mlx-lm-server" ] || [ "$1" = "--build" ]; then
  echo "Building (release)..."
  PYO3_PYTHON="$VENV/bin/python" PATH="$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin:$PATH" \
    "$CARGO" build --release 2>&1
fi

PYTHONPATH="$VENV/lib/python3.13/site-packages" \
  VIRTUAL_ENV="$VENV" \
  "$SCRIPT_DIR/target/release/mlx-lm-server"
