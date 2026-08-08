#!/usr/bin/env bash
set -euo pipefail

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${CURRENT_DIR}/../model"
mkdir -p "${MODEL_DIR}"

echo "=== Fetching Whisper encoder model ==="
wget -O "${MODEL_DIR}/whisper_encoder_base_20s.onnx" \
  https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/whisper/whisper_encoder_base_20s.onnx

echo "=== Fetching Whisper decoder model ==="
wget -O "${MODEL_DIR}/whisper_decoder_base_20s.onnx" \
  https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/whisper/whisper_decoder_base_20s.onnx
