#!/usr/bin/env bash
set -euo pipefail

# Push the Android build of whisper-rknn to a device and run it.
#
# Usage:
#   ./scripts/run-android-batch.sh <device-ip> [model-dir] [rknn-decoder] [rknn-encoder] [sample-path]
#
# Arguments:
#   device-ip    IP address (or host) of the target device, as used by `adb connect`.
#   model-dir    Directory containing the RKNN model files (default model/).
#   rknn-decoder Path to the RKNN decoder model (default <model-dir>/whisper_decoder_base_20s.rknn)
#   rknn-encoder Path to the RKNN encoder model (default <model-dir>/whisper_encoder_base_20s.rknn)
#   sample-path  Path to a sample audio file to transcribe (default jfk.wav in the samples directory).
#
# Environment variables:
#   REMOTE_DIR  Directory to push files to on the device (default /data/local/tmp).
#

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Parse environment
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-android}"
REMOTE_DIR="${REMOTE_DIR:-/data/local/tmp}"
REMOTE_MODEL_DIR="${REMOTE_DIR}/model"
REMOTE_LIB_DIR="${REMOTE_DIR}/lib"

# Parse argument
DEVICE_IP="${1:-}"
MODEL_DIR="${2:-model}"
RKNN_DECODER="${3:-${MODEL_DIR}/whisper_decoder_base_20s.rknn}"
RKNN_ENCODER="${4:-${MODEL_DIR}/whisper_encoder_base_20s.rknn}"
SAMPLE_PATH="${5:-${ROOT_DIR}/samples/jfk.wav}"

# Check device IP argument
if [[ -z "${DEVICE_IP}" ]]; then
  cat <<USAGE
Usage: $0 <device-ip> [model-dir] [rknn-decoder] [rknn-encoder] [sample-path]

Arguments:
  device-ip    IP address (or host) of the target device, as used by 'adb connect'.
  model-dir    Directory containing the RKNN model files (default model/).
  rknn-decoder Path to the RKNN decoder model (default <model-dir>/whisper_decoder_base_20s.rknn)
  rknn-encoder Path to the RKNN encoder model (default <model-dir>/whisper_encoder_base_20s.rknn)
  sample-path  Path to a sample audio file to transcribe (default jfk.wav in the samples directory).
USAGE
  exit 1
fi

# -----------------------------------------------------------------------------
# Preconditions
# -----------------------------------------------------------------------------

echo "=== Check preconditions ==="

WHISPER_RKNN_BIN="${BUILD_DIR}/whisper-rknn"
if [[ ! -x "${WHISPER_RKNN_BIN}" ]]; then
  echo "Error: ${WHISPER_RKNN_BIN} not found." >&2
  echo "Run the Android build first: ./scripts/build-android.sh docker" >&2
  exit 1
fi

echo "Whisper binary: ${WHISPER_RKNN_BIN}"

for tool in adb curl; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Error: required tool '${tool}' is not installed or not on PATH." >&2
    exit 1
  fi
done

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Error: model directory '${MODEL_DIR}' does not exist." >&2
  exit 1
fi
echo "Model directory: ${MODEL_DIR}"

if [[ ! -f "${MODEL_DIR}/mel_80_filters.txt" ]]; then
  echo "Error: model directory '${MODEL_DIR}' is missing mel_80_filters.txt." >&2
  exit 1
fi
echo "Found mel_80_filters.txt"

if [[ ! -f "${MODEL_DIR}/vocab_en.txt" ]]; then
  echo "Error: model directory '${MODEL_DIR}' is missing vocab_en.txt." >&2
  exit 1
fi
echo "Found vocab_en.txt"

if [[ ! -f "${RKNN_DECODER}" ]]; then
  echo "Error: RKNN decoder model '${RKNN_DECODER}' does not exist." >&2
  exit 1
fi
echo "Found RKNN Decoder: ${RKNN_DECODER}"

if [[ ! -f "${RKNN_ENCODER}" ]]; then
  echo "Error: RKNN encoder model '${RKNN_ENCODER}' does not exist." >&2
  exit 1
fi
echo "Found RKNN Encoder: ${RKNN_ENCODER}"

echo "All preconditions satisfied."

# -----------------------------------------------------------------------------
# ADB connect
# -----------------------------------------------------------------------------

echo
echo "=== Connect to device ==="

adb connect "${DEVICE_IP}" >/dev/null || true

# Resolve the serial: prefer an explicit ip:port match, fall back to the device.
SERIAL=""
if adb devices | awk '{print $1}' | grep -q "^${DEVICE_IP}:"; then
  SERIAL="$(adb devices | awk '{print $1}' | grep "^${DEVICE_IP}:" | head -n1)"
elif adb devices | awk 'NR>1 && $2=="device" {print $1}' | grep -qx "${DEVICE_IP}"; then
  SERIAL="${DEVICE_IP}"
fi

if [[ -z "${SERIAL}" ]]; then
  echo "Error: device ${DEVICE_IP} is not connected." >&2
  echo "Connected devices:" >&2
  adb devices >&2
  exit 1
fi

echo "Using device: ${SERIAL}"
ADB=(adb -s "${SERIAL}")

# -----------------------------------------------------------------------------
# Sync helpers
# -----------------------------------------------------------------------------

# Local file size in bytes (portable across macOS and Linux).
local_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

# Push a model file only if the device copy is missing or a different size.
# Large model files are expensive to transfer, so we compare byte size, which
# is instant on both ends and a strong signal for these immutable files.
sync_model() {
  local src="$1"
  local dst="$2"
  local name lsize rsize
  name="$(basename "${dst}")"

  lsize="$(local_size "${src}")"
  # A missing remote file makes stat exit non-zero; tolerate it (rsize stays
  # empty) so set -o pipefail / set -e don't abort the script on first push.
  rsize="$("${ADB[@]}" shell "stat -c%s '${dst}' 2>/dev/null" | tr -d '\r' || true)"

  if [[ "${rsize}" == "${lsize}" ]]; then
    echo "Up to date (${lsize} bytes): ${dst}"
    return
  fi

  echo "Pushing ${name} (local ${lsize} bytes, remote ${rsize:-missing}) ..."
  "${ADB[@]}" push "${src}" "${dst}"
}

# -----------------------------------------------------------------------------
# Push files
# -----------------------------------------------------------------------------

echo
echo "=== Push files to device ==="

"${ADB[@]}" shell mkdir -p "${REMOTE_MODEL_DIR}" "${REMOTE_LIB_DIR}"

# Push binaries
"${ADB[@]}" push "${WHISPER_RKNN_BIN}" "${REMOTE_DIR}/whisper-rknn"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rknpu2/Android/arm64-v8a/librknnrt.so" "${REMOTE_LIB_DIR}/"

# Push model text files
"${ADB[@]}" push "${MODEL_DIR}/mel_80_filters.txt" "${REMOTE_MODEL_DIR}/"
"${ADB[@]}" push "${MODEL_DIR}/vocab_en.txt" "${REMOTE_MODEL_DIR}/"

# Model files are large; only push when missing or changed on the device.
sync_model "${RKNN_DECODER}" "${REMOTE_MODEL_DIR}/decoder.rknn"
sync_model "${RKNN_ENCODER}" "${REMOTE_MODEL_DIR}/encoder.rknn"

# sample
sync_model "${SAMPLE_PATH}" "${REMOTE_DIR}/sample.wav"

"${ADB[@]}" shell chmod 755 "${REMOTE_DIR}/whisper-rknn"

# -----------------------------------------------------------------------------
# Transcribe sample
# -----------------------------------------------------------------------------

echo
echo "=== Start whisper-rknn on ${SERIAL} ==="

# Force a pseudo-terminal (-t -t) so the server's stdout is line-buffered and
# streams live; without a tty it stays block-buffered and nothing appears until
# the process exits. Merge stderr into stdout so both are shown.
"${ADB[@]}" shell -t -t "cd ${REMOTE_DIR} && LD_LIBRARY_PATH=${REMOTE_LIB_DIR} ./whisper-rknn --enable-timestamps ${REMOTE_MODEL_DIR}/encoder.rknn ${REMOTE_MODEL_DIR}/decoder.rknn en ${REMOTE_DIR}/sample.wav 2>&1"
