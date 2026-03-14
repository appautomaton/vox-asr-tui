#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Moonshine v2 medium-streaming runtime.
# This script:
# 1) downloads pinned Moonshine source
# 2) builds core libmoonshine for the host platform
# 3) builds bin/moonshine_asr helper against moonshine-c-api.h
# 4) downloads medium-streaming-en quantized model files

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"

MOONSHINE_TAG="v0.0.49"
MOONSHINE_SRC_ROOT="${ROOT_DIR}/bin/moonshine-src"
MOONSHINE_SRC_DIR="${MOONSHINE_SRC_ROOT}/moonshine-0.0.49"
MOONSHINE_ARCHIVE="${MOONSHINE_SRC_ROOT}/moonshine-${MOONSHINE_TAG}.tar.gz"
MOONSHINE_CORE_DIR="${MOONSHINE_SRC_DIR}/core"
MOONSHINE_BUILD_DIR="${MOONSHINE_CORE_DIR}/build-tnt"

HELPER_SRC="${ROOT_DIR}/csrc/moonshine_asr.cpp"
HELPER_BIN="${ROOT_DIR}/bin/moonshine_asr"
RUNTIME_ROOT="${ROOT_DIR}/bin/moonshine-runtime"
MODEL_DIR="${ROOT_DIR}/bin/moonshine-models/medium-streaming-en"
MODEL_BASE_URL="https://download.moonshine.ai/model/medium-streaming-en/quantized"

MODEL_FILES=(
  "adapter.ort"
  "cross_kv.ort"
  "decoder_kv.ort"
  "encoder.ort"
  "frontend.ort"
  "streaming_config.json"
  "tokenizer.bin"
)

if [[ $# -ne 0 ]]; then
  echo "Usage: ./bootstrap-moonshine.sh"
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: missing required command '$1'"
    exit 1
  fi
}

detect_platform() {
  local uname_s uname_m
  uname_s="$(uname -s)"
  uname_m="$(uname -m)"

  case "${uname_s}:${uname_m}" in
    Linux:x86_64|Linux:amd64)
      PLATFORM_ID="linux-x86_64"
      MOONSHINE_LIB_NAME="libmoonshine.so"
      ORT_LIB_REL="core/third-party/onnxruntime/lib/linux/x86_64/libonnxruntime.so.1"
      HELPER_RPATH="\$ORIGIN/moonshine-runtime/${PLATFORM_ID}"
      ;;
    Linux:aarch64|Linux:arm64)
      PLATFORM_ID="linux-aarch64"
      MOONSHINE_LIB_NAME="libmoonshine.so"
      ORT_LIB_REL="core/third-party/onnxruntime/lib/linux/aarch64/libonnxruntime.so.1"
      HELPER_RPATH="\$ORIGIN/moonshine-runtime/${PLATFORM_ID}"
      ;;
    Darwin:arm64)
      PLATFORM_ID="macos-arm64"
      MOONSHINE_LIB_NAME="libmoonshine.dylib"
      ORT_LIB_REL="core/third-party/onnxruntime/lib/macos/arm64/libonnxruntime.1.23.2.dylib"
      HELPER_RPATH="@executable_path/moonshine-runtime/${PLATFORM_ID}"
      ;;
    *)
      echo "Error: unsupported platform ${uname_s}/${uname_m}"
      echo "Supported: Linux x86_64/aarch64, macOS arm64"
      exit 1
      ;;
  esac

  RUNTIME_DIR="${RUNTIME_ROOT}/${PLATFORM_ID}"
  ORT_LIB_PATH="${MOONSHINE_SRC_DIR}/${ORT_LIB_REL}"
}

ensure_source_tree() {
  require_cmd curl
  require_cmd tar

  mkdir -p "${MOONSHINE_SRC_ROOT}"
  if [[ ! -d "${MOONSHINE_SRC_DIR}" ]]; then
    echo "Downloading Moonshine source ${MOONSHINE_TAG} ..."
    curl --fail --location --retry 3 --silent --show-error \
      --output "${MOONSHINE_ARCHIVE}" \
      "https://github.com/moonshine-ai/moonshine/archive/refs/tags/${MOONSHINE_TAG}.tar.gz"
    tar -xzf "${MOONSHINE_ARCHIVE}" -C "${MOONSHINE_SRC_ROOT}"
  fi

  if [[ ! -f "${MOONSHINE_CORE_DIR}/CMakeLists.txt" ]]; then
    echo "Error: invalid Moonshine source tree at ${MOONSHINE_SRC_DIR}"
    exit 1
  fi
}

build_moonshine_core() {
  require_cmd cmake

  echo "Building Moonshine core (${PLATFORM_ID}) ..."
  cmake -S "${MOONSHINE_CORE_DIR}" -B "${MOONSHINE_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
  cmake --build "${MOONSHINE_BUILD_DIR}" --target moonshine -j

  MOONSHINE_LIB_PATH="${MOONSHINE_BUILD_DIR}/${MOONSHINE_LIB_NAME}"
  if [[ ! -f "${MOONSHINE_LIB_PATH}" ]]; then
    echo "Error: Moonshine library not found at ${MOONSHINE_LIB_PATH}"
    exit 1
  fi
  if [[ ! -f "${ORT_LIB_PATH}" ]]; then
    echo "Error: ONNX Runtime library not found at ${ORT_LIB_PATH}"
    exit 1
  fi
}

build_helper() {
  require_cmd c++

  if [[ ! -f "${HELPER_SRC}" ]]; then
    echo "Error: helper source missing at ${HELPER_SRC}"
    exit 1
  fi

  echo "Building moonshine_asr helper ..."
  c++ \
    -std=c++20 \
    -O3 \
    -Wall -Wextra -pedantic \
    -I"${MOONSHINE_CORE_DIR}" \
    "${HELPER_SRC}" \
    -L"${MOONSHINE_BUILD_DIR}" \
    -lmoonshine \
    -Wl,-rpath,"${HELPER_RPATH}" \
    -o "${HELPER_BIN}"
  chmod +x "${HELPER_BIN}"
}

install_runtime_libs() {
  mkdir -p "${RUNTIME_DIR}"
  install -m 0755 "${MOONSHINE_LIB_PATH}" "${RUNTIME_DIR}/${MOONSHINE_LIB_NAME}"
  install -m 0755 "${ORT_LIB_PATH}" "${RUNTIME_DIR}/$(basename "${ORT_LIB_PATH}")"
  echo "Installed runtime libs to ${RUNTIME_DIR}"
}

download_models() {
  require_cmd curl
  mkdir -p "${MODEL_DIR}"

  for file in "${MODEL_FILES[@]}"; do
    local_dest="${MODEL_DIR}/${file}"
    local_url="${MODEL_BASE_URL}/${file}"
    if [[ -s "${local_dest}" ]]; then
      echo "Model file already present: ${local_dest}"
      continue
    fi
    echo "Downloading ${file} ..."
    curl --fail --location --retry 3 --silent --show-error \
      --output "${local_dest}" \
      "${local_url}"
  done

  echo "Model files ready in: ${MODEL_DIR}"
}

detect_platform
ensure_source_tree
build_moonshine_core
build_helper
install_runtime_libs
download_models

echo "Run: uv run tnt"
