#!/usr/bin/env bash
# Thin wrapper — delegates to the repo-root run.sh with mode=lm
set -euo pipefail
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/run.sh" lm "$@"
