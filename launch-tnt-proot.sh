#!/usr/bin/env bash
set -euo pipefail

# Launch TNT inside proot with Termux API capture backend.
# You can still override values by exporting them before running this script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TERMUX_PREFIX_DEFAULT="/data/data/com.termux/files/usr"
TERMUX_BIN_DEFAULT="${TERMUX_PREFIX_DEFAULT}/bin"
TERMUX_TMP_DEFAULT="/data/data/com.termux/files/home/.cache/tnt"

if [[ -d "${TERMUX_BIN_DEFAULT}" ]]; then
  export PATH="${TERMUX_BIN_DEFAULT}:${PATH}"
fi

export TMPDIR="${TMPDIR:-${TERMUX_TMP_DEFAULT}}"
mkdir -p "${TMPDIR}"

export TNT_CAPTURE_BACKEND="${TNT_CAPTURE_BACKEND:-termux_api}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found in PATH."
  exit 1
fi

if ! command -v termux-microphone-record >/dev/null 2>&1; then
  echo "Error: termux-microphone-record not found."
  echo "Install Termux:API app and run: pkg install termux-api"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found."
  echo "Install ffmpeg in your proot distro (apt) or host environment."
  exit 1
fi

cd "${SCRIPT_DIR}"
exec uv run tnt "$@"
