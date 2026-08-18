import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
import whisper
from onnxsim import simplify

warnings.filterwarnings("ignore", category=UserWarning)

SAMPLE_RATE = 16000
MEL_FRAMES_PER_SECOND = 100
MIN_CHUNK_LENGTH = 5
MAX_CHUNK_LENGTH = 30


class ShortAudioEncoder(torch.nn.Module):
    """Whisper encoder with positional embeddings sliced to the input length."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.gelu = torch.nn.GELU()

    def forward(self, x):
        encoder = self.encoder
        x = self.gelu(encoder.conv1(x))
        x = self.gelu(encoder.conv2(x))
        x = x.permute(0, 2, 1)

        positions = encoder.positional_embedding[-x.shape[1] :]
        x = (x + positions).to(x.dtype)
        for block in encoder.blocks:
            x = block(x)
        return encoder.ln_post(x)


def chunk_length_seconds(value):
    """Parse --chunk-length as a whole number of seconds within the supported range."""
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"must be a whole number of seconds, got {value!r}"
        ) from None
    if not MIN_CHUNK_LENGTH <= seconds <= MAX_CHUNK_LENGTH:
        raise argparse.ArgumentTypeError(
            f"must be from {MIN_CHUNK_LENGTH} to {MAX_CHUNK_LENGTH} seconds, "
            f"got {seconds}"
        )
    return seconds


def setup_model(model_type):
    model = whisper.load_model(model_type).to("cpu")
    model.requires_grad_(False)
    model.eval()
    return model


def setup_data(model, n_mels, max_tokens, chunk_length):
    audio = np.random.randn(chunk_length * SAMPLE_RATE).astype(np.float32)
    x_mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).unsqueeze(0)
    encoder = ShortAudioEncoder(model.encoder)
    encoder_output = encoder(x_mel)
    x_tokens = torch.randint(0, 100, (1, max_tokens), dtype=torch.long)
    return encoder, x_mel, encoder_output, x_tokens


def simplify_onnx_model(model_path):
    original_model = onnx.load(model_path)
    onnx.checker.check_model(original_model)
    simplified_model, check = simplify(original_model)
    if not check:
        raise RuntimeError(f"Failed to validate simplified model: {model_path}")
    onnx.checker.check_model(simplified_model)
    onnx.save(simplified_model, model_path)


def model_label(model_type):
    if model_type in whisper.available_models():
        return model_type.replace(".", "-")
    label = Path(model_type).stem
    return re.sub(r"[^A-Za-z0-9_-]+", "-", label).strip("-") or "custom"


def parse_args():
    parser = argparse.ArgumentParser(description="Export fixed-shape Whisper ONNX models")
    parser.add_argument(
        "--model-type",
        dest="model_type",
        required=True,
        help="Whisper model name or checkpoint path",
    )
    parser.add_argument(
        "--chunk-length",
        type=chunk_length_seconds,
        required=True,
        metavar="SECONDS",
        help=(
            f"fixed audio window in whole seconds, from {MIN_CHUNK_LENGTH} to "
            f"{MAX_CHUNK_LENGTH}"
        ),
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model"),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace existing ONNX models",
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be greater than zero")
    return parser, args


def main():
    parser, args = parse_args()
    model = setup_model(args.model_type)
    if args.max_tokens > model.dims.n_text_ctx:
        parser.error(
            f"--max-tokens cannot exceed the model text context of "
            f"{model.dims.n_text_ctx}"
        )
    if args.n_mels != model.dims.n_mels:
        parser.error(
            f"--n-mels must match the model Mel count of {model.dims.n_mels}"
        )
    encoder_frames = args.chunk_length * MEL_FRAMES_PER_SECOND // 2
    if encoder_frames > model.dims.n_audio_ctx:
        parser.error(
            f"--chunk-length cannot exceed the model audio context of "
            f"{model.dims.n_audio_ctx * 2 // MEL_FRAMES_PER_SECOND} seconds"
        )

    label = model_label(args.model_type)
    suffix = f"{label}_{args.chunk_length}s"
    encoder_path = args.output_dir / f"whisper_encoder_{suffix}.onnx"
    decoder_path = args.output_dir / f"whisper_decoder_{suffix}.onnx"
    existing = [path for path in (encoder_path, decoder_path) if path.exists()]
    if existing and not args.force:
        parser.error(
            "output already exists (pass --force to replace it): "
            + ", ".join(str(path) for path in existing)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    encoder, x_mel, encoder_output, x_tokens = setup_data(
        model, args.n_mels, args.max_tokens, args.chunk_length
    )
    expected_mel_frames = args.chunk_length * MEL_FRAMES_PER_SECOND
    if x_mel.shape != (1, args.n_mels, expected_mel_frames):
        raise RuntimeError(f"Unexpected encoder input shape: {tuple(x_mel.shape)}")
    if encoder_output.shape[1] != expected_mel_frames // 2:
        raise RuntimeError(
            f"Unexpected encoder context length: {encoder_output.shape[1]}"
        )

    torch.onnx.export(
        encoder,
        x_mel,
        encoder_path,
        input_names=["x"],
        output_names=["out"],
        opset_version=12,
    )
    torch.onnx.export(
        model.decoder,
        (x_tokens, encoder_output),
        decoder_path,
        input_names=["tokens", "audio"],
        output_names=["out"],
        opset_version=12,
    )

    simplify_onnx_model(encoder_path)
    simplify_onnx_model(decoder_path)
    print(f"Encoder input:  {tuple(x_mel.shape)}")
    print(f"Encoder output: {tuple(encoder_output.shape)}")
    print(f"Encoder model:  {encoder_path}")
    print(f"Decoder model:  {decoder_path}")


if __name__ == "__main__":
    main()
