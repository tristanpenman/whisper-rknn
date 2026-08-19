#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER="python"
TARGET="rk3588"
DTYPE="fp"
MAX_TOKENS=""
OUTPUT_DIR="model"
ONNX_ONLY=false
FORCE=false

usage() {
  cat <<'USAGE'
Usage: ./scripts/convert-whisper.sh [options] <model> <seconds>

Arguments:
  model                   Whisper model name or checkpoint path
  seconds                 Audio window: whole seconds from 5 to 30

Options:
  --target <platform>     RKNN target (default: rk3588)
  --dtype <fp|i8|u8>      RKNN output type (default: fp)
  --max-tokens <count>    Fixed decoder input length (default: model-derived)
  --output-dir <path>     Output directory (default: model)
  --onnx-only             Stop after ONNX export
  --force                 Replace existing output files
  -h, --help              Show this help
USAGE
}

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:?missing value for --target}"
      shift 2
      ;;
    --dtype)
      DTYPE="${2:?missing value for --dtype}"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="${2:?missing value for --max-tokens}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?missing value for --output-dir}"
      shift 2
      ;;
    --onnx-only)
      ONNX_ONLY=true
      shift
      ;;
    --force)
      FORCE=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
  usage >&2
  exit 2
fi

MODEL_TYPE="${POSITIONAL[0]}"

CHUNK_LENGTH="${POSITIONAL[1]}"
if [[ ! "$CHUNK_LENGTH" =~ ^([5-9]|1[0-9]|2[0-9]|30)$ ]]; then
  echo "seconds must be a whole integer from 5 to 30" >&2
  exit 2
fi

if [[ "$DTYPE" != "fp" && "$DTYPE" != "i8" && "$DTYPE" != "u8" ]]; then
  echo "--dtype must be fp, i8, or u8" >&2
  exit 2
fi

if [[ -n "$MAX_TOKENS" && ! "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-tokens must be a positive integer" >&2
  exit 2
fi

if [[ "$OUTPUT_DIR" = /* ]]; then
  case "$OUTPUT_DIR" in
    "$ROOT_DIR"|"$ROOT_DIR"/*)
      ;;
    *)
      echo "--output-dir must be inside the repository" >&2
      exit 2
      ;;
  esac
  if [[ "$OUTPUT_DIR" == "$ROOT_DIR" ]]; then
    OUTPUT_ARG="."
  else
    OUTPUT_ARG="${OUTPUT_DIR#"$ROOT_DIR"/}"
  fi
else
  OUTPUT_ARG="$OUTPUT_DIR"
fi

if [[ "/$OUTPUT_ARG/" == *"/../"* ]]; then
  echo "--output-dir must not contain '..'" >&2
  exit 2
fi

if [[ -f "$MODEL_TYPE" ]]; then
  MODEL_PATH="$(cd "$(dirname "$MODEL_TYPE")" && pwd)/$(basename "$MODEL_TYPE")"
  case "$MODEL_PATH" in
    "$ROOT_DIR"/*) MODEL_ARG="${MODEL_PATH#"$ROOT_DIR"/}" ;;
    *) echo "checkpoint path must be inside the repository" >&2; exit 2 ;;
  esac
  MODEL_LABEL="$(basename "${MODEL_TYPE%.*}")"
else
  MODEL_ARG="$MODEL_TYPE"
  MODEL_LABEL="${MODEL_TYPE//./-}"
fi

MODEL_LABEL="$(printf '%s' "$MODEL_LABEL" | tr -cs 'A-Za-z0-9_-' '-')"
MODEL_LABEL="${MODEL_LABEL#-}"
MODEL_LABEL="${MODEL_LABEL%-}"
PREFIX="${MODEL_LABEL}_${CHUNK_LENGTH}s"
ENCODER_ONNX="${OUTPUT_ARG%/}/whisper_encoder_${PREFIX}.onnx"
DECODER_ONNX="${OUTPUT_ARG%/}/whisper_decoder_${PREFIX}.onnx"
ENCODER_RKNN="${OUTPUT_ARG%/}/whisper_encoder_${PREFIX}.rknn"
DECODER_RKNN="${OUTPUT_ARG%/}/whisper_decoder_${PREFIX}.rknn"

cd "$ROOT_DIR"
EXPORT_ARGS=(
  python -m whisper_rknn.export_onnx
  --model-type "$MODEL_ARG"
  --chunk-length "$CHUNK_LENGTH"
  --output-dir "$OUTPUT_ARG"
)
if [[ -n "$MAX_TOKENS" ]]; then
  EXPORT_ARGS+=(--max-tokens "$MAX_TOKENS")
fi
if [[ "$FORCE" == true ]]; then
  EXPORT_ARGS+=(--force)
elif [[ "$ONNX_ONLY" == false && ( -e "$ENCODER_RKNN" || -e "$DECODER_RKNN" ) ]]; then
  echo "RKNN output already exists (pass --force to replace it)" >&2
  exit 2
fi

echo "=== Export Whisper ${MODEL_TYPE} with a ${CHUNK_LENGTH}-second window ==="
docker compose run --rm --remove-orphans --build \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.cache/home \
  $CONTAINER \
  "${EXPORT_ARGS[@]}"

if [[ "$ONNX_ONLY" == true ]]; then
  exit 0
fi

echo "=== Convert encoder to RKNN (${TARGET}, ${DTYPE}) ==="
docker compose run --rm --remove-orphans \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.cache/home \
  $CONTAINER \
  python -m whisper_rknn.convert \
    "$ENCODER_ONNX" "$TARGET" "$DTYPE" "$ENCODER_RKNN"

echo "=== Convert decoder to RKNN (${TARGET}, ${DTYPE}) ==="
docker compose run --rm --remove-orphans \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.cache/home \
  $CONTAINER \
  python -m whisper_rknn.convert \
    "$DECODER_ONNX" "$TARGET" "$DTYPE" "$DECODER_RKNN"

echo "Created:"
echo "--> ${ENCODER_ONNX}"
echo "--> ${DECODER_ONNX}"
echo "--> ${ENCODER_RKNN}"
echo "--> ${DECODER_RKNN}"
