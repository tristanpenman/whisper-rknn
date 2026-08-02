import argparse
import importlib
import sys
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxsim import simplify


SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path = [path for path in sys.path if path != SCRIPT_DIR]
whisper = importlib.import_module("whisper")

warnings.filterwarnings("ignore", category=UserWarning)


def setup_model(model_type):
    model = whisper.load_model(model_type).to("cpu")
    model.requires_grad_(False)
    model.eval()
    return model


def setup_data(model, n_mels):
    sample_rate = 16000
    audio_array = np.random.randn(1, 40 * sample_rate).astype(np.float32)
    audio = whisper.pad_or_trim(audio_array.flatten())
    x_mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels).unsqueeze(0)
    encoder_output = model.encoder(x_mel)
    max_tokens = 12
    x_tokens = torch.randint(0, 100, (1, max_tokens), dtype=torch.long)
    return x_mel, encoder_output, x_tokens


def simplify_onnx_model(model_path):
    original_model = onnx.load(model_path)
    simplified_model, check = simplify(original_model)
    onnx.save(simplified_model, model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export whisper onnx model", add_help=True
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        default="base",
        help="model type, could be tiny, base, small, medium, ...",
    )
    parser.add_argument(
        "--n_mels",
        type=int,
        required=False,
        default=80,
        help="number of mels",
    )
    args = parser.parse_args()

    print("whisper available_models: ", whisper.available_models())
    model = setup_model(args.model_type)
    x_mel, encoder_output, x_tokens = setup_data(model, args.n_mels)

    save_encoder_model_path = (
        f"../model/whisper_encoder_{args.model_type}.onnx"
    )
    save_decoder_model_path = (
        f"../model/whisper_decoder_{args.model_type}.onnx"
    )
    torch.onnx.export(
        model.encoder,
        x_mel,
        save_encoder_model_path,
        input_names=["x"],
        output_names=["out"],
        opset_version=12,
    )

    torch.onnx.export(
        model.decoder,
        (x_tokens, encoder_output),
        save_decoder_model_path,
        input_names=["tokens", "audio"],
        output_names=["out"],
        opset_version=12,
    )

    simplify_onnx_model(save_encoder_model_path)
    print("\nThe encoder model is saved in:", save_encoder_model_path)
    simplify_onnx_model(save_decoder_model_path)
    print("The decoder model is saved in:", save_decoder_model_path)
