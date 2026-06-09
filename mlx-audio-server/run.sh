#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$SCRIPT_DIR/.venv}"

if [[ ! -d "$VENV" ]]; then
  echo "Creating Python venv at $VENV ..."
  python3.13 -m venv "$VENV" || python3.12 -m venv "$VENV" || python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

echo "Installing Python dependencies ..."
pip install -q --upgrade pip
pip install -q mlx-audio numpy

export PYO3_PYTHON="$VENV/bin/python"

echo "Building mlx-audio-server ..."
cargo build --release

echo "Starting server ..."
exec ./target/release/mlx-audio-server "$@"
