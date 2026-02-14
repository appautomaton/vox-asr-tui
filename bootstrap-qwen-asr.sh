#!/usr/bin/env bash
set -euo pipefail

# Bootstrap TNT ASR runtime from vendored qwen-asr source.
# This script always:
# 1) builds bin/qwen_asr from bin/qwen-asr
# 2) downloads Qwen3-ASR-0.6B files to bin/qwen3-asr-0.6b

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
SRC_DIR="${ROOT_DIR}/bin/qwen-asr"
OUT_BIN="${ROOT_DIR}/bin/qwen_asr"
MODEL_DIR="${ROOT_DIR}/bin/qwen3-asr-0.6b"
MODEL_BASE_URL="https://huggingface.co/Qwen/Qwen3-ASR-0.6B/resolve/main"
MODEL_FILES=(
  "config.json"
  "generation_config.json"
  "model.safetensors"
  "vocab.json"
  "merges.txt"
)

if [[ $# -ne 0 ]]; then
  echo "Usage: ./bootstrap-qwen-asr.sh"
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: missing required command '$1'"
    exit 1
  fi
}

ensure_source_tree() {
  if [[ ! -d "${SRC_DIR}" ]]; then
    echo "Error: vendored source missing at ${SRC_DIR}"
    echo "Expected antirez/qwen-asr source to be committed under bin/qwen-asr."
    exit 1
  fi

  if [[ ! -f "${SRC_DIR}/Makefile" ]]; then
    echo "Error: ${SRC_DIR} is not a valid qwen-asr source tree (missing Makefile)."
    exit 1
  fi
}

build_binary() {
  require_cmd make
  require_cmd cc

  local uname_s
  uname_s="$(uname -s)"

  pushd "${SRC_DIR}" >/dev/null

  if [[ "${uname_s}" == "Darwin" ]]; then
    # On macOS, qwen-asr's `make blas` uses Apple Accelerate.
    # Do not gate this on cblas.h header checks (that is for OpenBLAS layouts).
    if make -s blas; then
      echo "Built qwen_asr with BLAS backend (Apple Accelerate)."
    else
      echo "BLAS build failed on macOS; falling back to non-BLAS build."
      echo "Hint: install Xcode Command Line Tools (`xcode-select --install`) and retry."
      make -s clean
      make -s qwen_asr CFLAGS="-Wall -Wextra -O3 -march=native -ffast-math" LDFLAGS="-lm -lpthread"
    fi
  elif printf '#include <cblas.h>\n' | cc -x c -E - >/dev/null 2>&1; then
    if make -s blas; then
      echo "Built qwen_asr with BLAS backend (OpenBLAS)."
    else
      echo "BLAS build failed, falling back to non-BLAS build."
      make -s clean
      make -s qwen_asr CFLAGS="-Wall -Wextra -O3 -march=native -ffast-math" LDFLAGS="-lm -lpthread"
    fi
  else
    echo "BLAS headers not found; building non-BLAS binary."
    echo "Hint (Debian/Ubuntu): install libopenblas-dev and rerun this script."
    make -s clean
    make -s qwen_asr CFLAGS="-Wall -Wextra -O3 -march=native -ffast-math" LDFLAGS="-lm -lpthread"
  fi

  if [[ ! -f qwen_asr ]]; then
    echo "Error: build completed but qwen_asr was not produced in ${SRC_DIR}."
    exit 1
  fi

  install -m 0755 qwen_asr "${OUT_BIN}"
  popd >/dev/null

  echo "Installed: ${OUT_BIN}"
}

download_models() {
  require_cmd curl
  mkdir -p "${MODEL_DIR}"

  for file in "${MODEL_FILES[@]}"; do
    dest="${MODEL_DIR}/${file}"
    url="${MODEL_BASE_URL}/${file}"

    if [[ "${file}" == "model.safetensors" ]]; then
      # Skip if likely complete; otherwise resume partial download.
      if [[ -s "${dest}" ]]; then
        size_bytes="$(wc -c <"${dest}")"
        if [[ "${size_bytes}" -ge 1500000000 ]]; then
          echo "Model file already present: ${dest}"
          continue
        fi
      fi
      echo "Downloading ${file} ..."
      curl --fail --location --continue-at - --retry 3 --silent --show-error --output "${dest}" "${url}"
      continue
    fi

    if [[ -s "${dest}" ]]; then
      echo "Model file already present: ${dest}"
      continue
    fi

    echo "Downloading ${file} ..."
    curl --fail --location --continue-at - --retry 3 --silent --show-error --output "${dest}" "${url}"
  done

  echo "Model files ready in: ${MODEL_DIR}"
}

ensure_source_tree
build_binary
download_models

echo "Run: uv run tnt"
