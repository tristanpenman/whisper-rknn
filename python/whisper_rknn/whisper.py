import argparse

import numpy as np
import scipy
import soundfile as sf
import torch
import torch.nn.functional as F

from whisper_rknn.rknn_utils import (
    load_model,
    model_input_shapes,
    release_model,
)

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
MAX_INITIAL_TIMESTAMP_INDEX = 50


def ensure_sample_rate(
    waveform,
    original_sample_rate,
    desired_sample_rate=16000
):
    if original_sample_rate != desired_sample_rate:
        print(
            f"resample_audio: {original_sample_rate} HZ -> "
            f"{desired_sample_rate} HZ"
        )
        desired_length = int(
            round(
                float(len(waveform))
                / original_sample_rate
                * desired_sample_rate
            )
        )
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform, desired_sample_rate


def ensure_channels(
    waveform,
    original_channels,
    desired_channels=1
):
    if original_channels != desired_channels:
        print(f"convert_channels: {original_channels} -> {desired_channels}")
        waveform = np.mean(waveform, axis=1)
    return waveform, desired_channels


def get_char_index(c):
    if "A" <= c <= "Z":
        return ord(c) - ord("A")
    if "a" <= c <= "z":
        return ord(c) - ord("a") + (ord("Z") - ord("A") + 1)
    if "0" <= c <= "9":
        return (
            ord(c)
            - ord("0")
            + (ord("Z") - ord("A"))
            + (ord("z") - ord("a"))
            + 2
        )
    if c == "+":
        return 62
    if c == "/":
        return 63
    print(f"Unknown character {ord(c)}, {c}")
    exit(-1)


def base64_decode(encoded_string):
    if not encoded_string:
        print("Empty string!")
        exit(-1)

    output_length = len(encoded_string) // 4 * 3
    decoded_string = bytearray(output_length)

    index = 0
    output_index = 0
    while index < len(encoded_string):
        if encoded_string[index] == "=":
            return " "

        first_byte = (get_char_index(encoded_string[index]) << 2) + (
            (get_char_index(encoded_string[index + 1]) & 0x30) >> 4
        )
        decoded_string[output_index] = first_byte

        if (
            index + 2 < len(encoded_string)
            and encoded_string[index + 2] != "="
        ):
            second_byte = (
                (get_char_index(encoded_string[index + 1]) & 0x0F) << 4
            ) + ((get_char_index(encoded_string[index + 2]) & 0x3C) >> 2)
            decoded_string[output_index + 1] = second_byte

            if (
                index + 3 < len(encoded_string)
                and encoded_string[index + 3] != "="
            ):
                third_byte = (
                    (get_char_index(encoded_string[index + 2]) & 0x03) << 6
                ) + get_char_index(encoded_string[index + 3])
                decoded_string[output_index + 2] = third_byte
                output_index += 3
            else:
                output_index += 2
        else:
            output_index += 1

        index += 4

    return decoded_string.decode("utf-8", errors="replace")


def read_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as vocab_file:
        vocab = {}
        for line in vocab_file:
            if len(line.strip().split(" ")) < 2:
                key = line.strip().split(" ")[0]
                value = ""
            else:
                key, value = line.strip().split(" ")
            vocab[key] = value
    return vocab


def pad_or_trim(audio_array, mel_frames):
    x_mel = np.zeros((N_MELS, mel_frames), dtype=np.float32)
    real_length = min(audio_array.shape[1], mel_frames)
    x_mel[:, :real_length] = audio_array[:, :real_length]
    return x_mel


def mel_filters(n_mels):
    assert n_mels in {80}, f"Unsupported n_mels: {n_mels}"
    filters_path = "../model/mel_80_filters.txt"
    mels_data = np.loadtxt(filters_path, dtype=np.float32).reshape((80, 201))
    return torch.from_numpy(mels_data)


def log_mel_spectrogram(audio, n_mels, padding=0):
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)

    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT)

    stft = torch.stft(
        audio, N_FFT, HOP_LENGTH, window=window, return_complex=True
    )
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


def run_encoder(encoder_model, in_encoder):
    if "rknn" in str(type(encoder_model)):
        return encoder_model.inference(inputs=in_encoder)[0]
    if "onnx" in str(type(encoder_model)):
        return encoder_model.run(None, {"x": in_encoder})[0]

    raise TypeError(f"Unsupported encoder model: {type(encoder_model)}")


def _decode(decoder_model, tokens, out_encoder):
    if "rknn" in str(type(decoder_model)):
        return decoder_model.inference(
            [np.asarray([tokens], dtype="int64"), out_encoder]
        )[0]
    if "onnx" in str(type(decoder_model)):
        return decoder_model.run(
            None,
            {
                "tokens": np.asarray([tokens], dtype="int64"),
                "audio": out_encoder,
            },
        )[0]

    raise TypeError(f"Unsupported decoder model: {type(decoder_model)}")


def timestamp_argmax(logits, generated_tokens, timestamp_begin=50364):
    end_token = 50257
    no_timestamps_token = 50363
    filtered_logits = logits.copy()
    filtered_logits[no_timestamps_token] = -np.inf

    if not generated_tokens:
        filtered_logits[:timestamp_begin] = -np.inf
        first_disallowed_timestamp = (
            timestamp_begin + MAX_INITIAL_TIMESTAMP_INDEX + 1
        )
        filtered_logits[first_disallowed_timestamp:] = -np.inf
        return int(filtered_logits.argmax())

    last_was_timestamp = generated_tokens[-1] >= timestamp_begin
    penultimate_was_timestamp = len(generated_tokens) < 2 or generated_tokens[-2] >= timestamp_begin

    if last_was_timestamp:
        if penultimate_was_timestamp:
            filtered_logits[timestamp_begin:] = -np.inf
        else:
            filtered_logits[:end_token] = -np.inf

    timestamp_tokens = [token for token in generated_tokens if token >= timestamp_begin]
    if timestamp_tokens:
        minimum_timestamp = timestamp_tokens[-1]
        if not last_was_timestamp or penultimate_was_timestamp:
            minimum_timestamp += 1
        filtered_logits[timestamp_begin:minimum_timestamp] = -np.inf

    timestamp_logits = filtered_logits[timestamp_begin:]
    max_timestamp_logit = np.max(timestamp_logits)
    if np.isfinite(max_timestamp_logit):
        timestamp_log_probability = max_timestamp_logit + np.log(
            np.exp(timestamp_logits - max_timestamp_logit).sum()
        )
        if timestamp_log_probability > np.max(filtered_logits[:timestamp_begin]):
            filtered_logits[:timestamp_begin] = -np.inf

    return int(filtered_logits.argmax())


def run_decoder(
    decoder_model,
    out_encoder,
    vocab,
    task_code,
    max_tokens,
    enable_timestamps=False,
):
    end_token = 50257  # tokenizer.eot
    initial_prompt = [50258, task_code, 50359]
    if not enable_timestamps:
        initial_prompt.append(50363)  # tokenizer.no_timestamps
    timestamp_begin = 50364  # tokenizer.timestamp_begin

    tokens = [0] * max_tokens
    tokens[:len(initial_prompt)] = initial_prompt
    token_count = len(initial_prompt)
    generated_tokens = []
    next_token = 50258  # tokenizer.sot

    while next_token != end_token and token_count < max_tokens:
        out_decoder = _decode(decoder_model, tokens, out_encoder)
        logits = out_decoder[0, token_count - 1]
        next_token = (
            timestamp_argmax(logits, generated_tokens, timestamp_begin)
            if enable_timestamps
            else int(logits.argmax())
        )
        generated_tokens.append(next_token)

        if next_token != end_token:
            tokens[token_count] = next_token
            token_count += 1

    tokens_str = "".join(vocab[str(token)] for token in generated_tokens)

    result = (
        tokens_str.replace("\u0120", " ")
        .replace("<|endoftext|>", "")
        .replace("\n", "")
    )
    if task_code == 50260:  # TASK_FOR_ZH
        result = base64_decode(result)
    return result


def model_dimensions(encoder_model, decoder_model):
    encoder_inputs = model_input_shapes(encoder_model)
    decoder_inputs = model_input_shapes(decoder_model)

    encoder_shape = encoder_inputs.get("x")
    token_shape = decoder_inputs.get("tokens")
    decoder_audio_shape = decoder_inputs.get("audio")
    if (
        encoder_shape is None
        or token_shape is None
        or decoder_audio_shape is None
    ):
        raise ValueError("Models do not have the expected Whisper inputs")
    if len(encoder_shape) != 3 or encoder_shape[:2] != [1, N_MELS]:
        raise ValueError(f"Unexpected encoder input shape: {encoder_shape}")
    if len(token_shape) != 2 or token_shape[0] != 1:
        raise ValueError(f"Unexpected decoder token shape: {token_shape}")
    if len(decoder_audio_shape) != 3 or decoder_audio_shape[0] != 1:
        raise ValueError(
            f"Unexpected decoder audio shape: {decoder_audio_shape}"
        )

    mel_frames = encoder_shape[2]
    max_tokens = token_shape[1]
    encoder_frames = mel_frames // 2
    if mel_frames <= 0 or mel_frames % 2 != 0:
        raise ValueError(f"Invalid encoder frame count: {mel_frames}")
    if mel_frames * HOP_LENGTH % SAMPLE_RATE != 0:
        raise ValueError(
            f"Encoder input does not represent a whole number of seconds: "
            f"{mel_frames} frames"
        )
    if max_tokens < 4:
        raise ValueError(f"Decoder token input is too short: {max_tokens}")
    if decoder_audio_shape[1] != encoder_frames:
        raise ValueError(
            "Encoder and decoder context lengths do not match: "
            f"{encoder_frames} and {decoder_audio_shape[1]}"
        )

    return mel_frames * HOP_LENGTH // SAMPLE_RATE, mel_frames, max_tokens


def load_array_from_file(filename, mel_frames):
    with open(filename, "r", encoding="utf-8") as file:
        data = file.readlines()

    array = []
    for line in data:
        row = [float(num) for num in line.split()]
        array.extend(row)

    return np.array(array).reshape((N_MELS, mel_frames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Whisper Python Demo", add_help=True
    )
    # basic params
    parser.add_argument(
        "--encoder_model_path",
        type=str,
        required=True,
        help="model path, could be .rknn or .onnx file",
    )
    parser.add_argument(
        "--decoder_model_path",
        type=str,
        required=True,
        help="model path, could be .rknn or .onnx file",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="recognition task, could be en or zh",
    )
    parser.add_argument(
        "--audio_path", type=str, required=True, help="audio path"
    )
    parser.add_argument(
        "--target", type=str, default="rk3588", help="target RKNPU platform"
    )
    parser.add_argument(
        "--device_id", type=str, default=None, help="device id"
    )
    parser.add_argument(
        "--enable-timestamps",
        action="store_true",
        help="include Whisper timestamp markers in the output",
    )
    args = parser.parse_args()

    # Set inputs
    if args.task == "en":
        vocab_path = "../model/vocab_en.txt"
        task_code = 50259
    elif args.task == "zh":
        vocab_path = "../model/vocab_zh.txt"
        task_code = 50260
    else:
        print(
            "\n\033[1;33mCurrently only English or Chinese recognition tasks "
            "are supported. Please specify --task as en or zh\033[0m"
        )
        exit(1)
    # Init models before preprocessing so their fixed input shapes determine
    # the audio window and decoder token count.
    encoder_model = load_model(
        args.encoder_model_path, args.target, args.device_id
    )
    decoder_model = load_model(
        args.decoder_model_path, args.target, args.device_id
    )
    chunk_length, mel_frames, max_tokens = model_dimensions(
        encoder_model, decoder_model
    )
    print(
        f"Model dimensions: {chunk_length}s audio, "
        f"{max_tokens} decoder tokens"
    )

    vocab = read_vocab(vocab_path)
    audio_data, sample_rate = sf.read(args.audio_path)
    channels = audio_data.ndim
    audio_data, channels = ensure_channels(audio_data, channels)
    audio_data, sample_rate = ensure_sample_rate(audio_data, sample_rate)
    audio_array = np.array(audio_data, dtype=np.float32)
    audio_array = log_mel_spectrogram(audio_array, N_MELS).numpy()
    x_mel = pad_or_trim(audio_array, mel_frames)
    x_mel = np.expand_dims(x_mel, 0)

    # Encode/Decode
    out_encoder = run_encoder(encoder_model, x_mel)
    result = run_decoder(
        decoder_model,
        out_encoder,
        vocab,
        task_code,
        max_tokens,
        enable_timestamps=args.enable_timestamps,
    )
    print("\nWhisper output:", result)

    # Release
    release_model(encoder_model)
    release_model(decoder_model)
