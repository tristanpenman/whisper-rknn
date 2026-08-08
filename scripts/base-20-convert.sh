#!/usr/bin/env bash
set -euo pipefail

echo "=== Convert Whisper encoder model to RKNN format ==="
docker compose run --rm --remove-orphans --build \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=python python \
    python -m whisper_rknn.convert \
      model/whisper_encoder_base_20s.onnx \
      rk3588 \
      fp \
      model/whisper_encoder_base_20s.rknn

echo "=== Convert Whisper decoder model to RKNN format ==="
docker compose run --rm --remove-orphans --build \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=python python \
    python -m whisper_rknn.convert \
      model/whisper_decoder_base_20s.onnx \
      rk3588 \
      fp \
      model/whisper_decoder_base_20s.rknn
