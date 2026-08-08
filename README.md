# Whisper RKNN

This repo contains a starter CMake project for running Whisper speech recognition models on Rockchip devices via RKNN. This code has been adapted from Rockchip's [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo).

The current implementation provides a C++ command-line demo for batch transcription on RK3588 devices running Linux and Android. It loads separate Whisper encoder and decoder RKNN models, prepares audio on the CPU, runs inference through the RKNN runtime, and prints the transcription and real-time factor.

Current capabilities:

* English and Chinese transcription
* WAV-file input through libsndfile
* Mono conversion and resampling to 16 kHz
* Log-Mel spectrogram preprocessing with FFTW and Arm NEON
* Linux/aarch64 and Android/arm64-v8a builds

Streaming transcription and microphone input are planned but are not yet implemented.

### Contents

* [Background](#background)
  * [Whisper](#whisper)
  * [RKNN](#rknn)
* [Project Structure](#project-structure)
  * [Layout](#layout)
  * [Dependencies](#dependencies)
* [Models and Data](#models-and-data)
  * [Model Conversion](#model-conversion)
* [Python Implementation](#python-implementation)
  * [Run the Python Demo](#run-the-python-demo)
  * [Export ONNX Models](#export-onnx-models)
* [Linux CLI](#linux-cli)
  * [Build](#build)
  * [Run](#run)
* [Android](#android)
* [CMake Configuration](#cmake-configuration)
* [Development Status](#development-status)
* [License](#license)

## Background

Whisper is an automatic speech recognition model that converts audio into text. This project adapts a Whisper encoder-decoder inference pipeline to Rockchip's RKNN runtime so that its neural-network stages can run on a supported Rockchip NPU.

### Whisper

Whisper processes audio in fixed-size windows. By default, the current demo prepares a log-Mel spectrogram for up to 20 seconds of audio, passes it through an RKNN encoder, and then repeatedly invokes an RKNN decoder to produce text tokens.

This is presently a batch pipeline: the complete audio file is read before inference begins. See [Development Status](#development-status) for planned streaming work.

### RKNN

RKNN is Rockchip's model format and NPU inference runtime. The CLI expects a compatible encoder and decoder that have been converted to `.rknn` files. The repository provides scripts for downloading the supported ONNX models and converting them for the RK3588.

The bundled runtime libraries target the following platforms:

| Platform | Architecture  |
|----------|---------------|
| Linux    | `aarch64`     |
| Linux    | `armhf`       |
| Android  | `arm64-v8a`   |
| Android  | `armeabi-v7a` |

The provided build scripts currently select `aarch64` for Linux and `arm64-v8a` for Android.

## Project Structure

This project targets Rockchip Linux and Android devices based on the RK3588. CMake is used for both platforms, with Docker-based helper scripts providing the supported build workflow.

### Layout

* `cmake/` - CMake helper modules for fetched dependencies.
* `cpp/src/` - C and C++ audio preprocessing, RKNN inference, and CLI sources.
* `cpp/tests/` - Native unit tests.
* `notes/` - Design notes and implementation research.
* `scripts/` - Linux and Android build helper scripts.
* `thirdparty/` - Bundled RKNN, libsndfile, and FFTW headers and prebuilt libraries.
* `CMakeLists.txt` - Shared CMake build configuration.
* `Dockerfile.android` - Android NDK build environment.
* `Dockerfile.native` - Linux/aarch64 cross-build environment.
* `docker-compose.yml` - Docker Compose services used by the build scripts.

### Dependencies

| Dependency                  | Purpose                                             |
|-----------------------------|-----------------------------------------------------|
| RKNN Runtime                | Loads and runs the encoder and decoder on the NPU.  |
| libsndfile                  | Reads audio files into floating-point sample data.  |
| FFTW                        | Computes the short-time Fourier transform.          |
| JSON for Modern C++         | Shared JSON dependency for ongoing development.     |
| GoogleTest                  | Builds the native unit-test executable.             |

RKNN, libsndfile, and FFTW are provided under `thirdparty/`. JSON for Modern C++ and GoogleTest are fetched by CMake while configuring the build.

## Models and Data

The executable requires two compatible RKNN model files:

| File            | Purpose                                       |
|-----------------|-----------------------------------------------|
| Encoder `.rknn` | Converts the log-Mel spectrogram to features. |
| Decoder `.rknn` | Generates transcription tokens.               |

The current tensor sizes are configured for the Whisper base model with a 20-second audio window. The encoder and decoder must be exported as a compatible pair with the tensor shapes expected by the C++ implementation.

The CLI also reads preprocessing and vocabulary data from paths relative to its working directory:

| Path                         | Purpose                         |
|------------------------------|---------------------------------|
| `model/mel_80_filters.txt`   | 80-channel Mel filter bank.     |
| `model/vocab_en.txt`         | English token vocabulary.       |
| `model/vocab_zh.txt`         | Chinese token vocabulary.       |

These data assets must be available alongside the converted RKNN models. Run the executable from the repository root, or reproduce the same `model/` layout beside the working directory used on the target device.

### Model Conversion

The repository provides a wrapper that exports an original OpenAI Whisper model with a shorter, fixed audio window and converts the resulting encoder-decoder pair to RKNN. Run it from the repository root. Conversion requires Docker with the Docker Compose plugin, and downloading an OpenAI checkpoint requires network access the first time that model is used.

To produce a 15-second or 20-second base model, run:

```bash
./scripts/convert-whisper.sh base 15
./scripts/convert-whisper.sh base 20
```

The wrapper accepts an OpenAI Whisper model name or the path to a checkpoint inside the repository. It exports and validates fixed-shape ONNX graphs, then converts both graphs for the RK3588 without integer quantization. For example, the 20-second command writes:

```text
model/whisper_encoder_base_20s.onnx
model/whisper_decoder_base_20s.onnx
model/whisper_encoder_base_20s.rknn
model/whisper_decoder_base_20s.rknn
```

Use `--target`, `--dtype`, `--max-tokens`, and `--output-dir` to change the conversion settings. `--onnx-only` stops after ONNX export, `--force` replaces existing outputs, and `--help` prints the complete interface. The checkpoint download is cached under `.cache/`.

The older download workflow remains available for the pre-exported 20-second base pair. First, download the ONNX models:

```bash
./scripts/base-20-fetch.sh
```

The script downloads both models into `model/`:

```text
model/whisper_encoder_base_20s.onnx
model/whisper_decoder_base_20s.onnx
```

Existing files at those paths are replaced when the script is run again.

Convert that downloaded pair with:

```bash
./scripts/base-20-convert.sh
```

`base-20-convert.sh` runs RKNN Toolkit inside the Python container. It converts both models for `rk3588` devices without integer quantization and writes:

```text
model/whisper_encoder_base_20s.rknn
model/whisper_decoder_base_20s.rknn
```

These output paths match the model names used by the CLI examples below. The conversion scripts do not create the Mel filter bank or vocabulary files, so `model/mel_80_filters.txt`, `model/vocab_en.txt`, and
`model/vocab_zh.txt` must also be present before running the example app.

## Python Implementation

The `python/whisper_rknn/` package contains the Python reference implementation used to prepare and validate models before running the C++ CLI. It implements the same audio loading, mono conversion, 16 kHz resampling, log-Mel preprocessing, encoder inference, and iterative decoder inference as the C++ implementation.

The Python implementation can run an ONNX encoder-decoder pair through ONNX Runtime on the host CPU. It can also load an RKNN pair on-device, through RKNN Toolkit. Its preprocessing defaults to an 80-Mel, 20-second Whisper model with a 12-token decoder input.

The Python dependencies are installed in the `python` Docker Compose service. Start a shell in that environment from the repository root:

```bash
docker compose run --rm --remove-orphans --build python
cd python
```

The commands below assume the current directory is `python/`. This is required because the implementation reads the Mel filter bank and vocabularies from `../model/`.

### Run the Python Demo

Run the downloaded ONNX models on the CPU with:

```bash
python -m whisper_rknn.whisper \
  --encoder_model_path ../model/whisper_encoder_base_20s.onnx \
  --decoder_model_path ../model/whisper_decoder_base_20s.onnx \
  --task en \
  --audio_path ../model/test_en.wav
```

To run converted RKNN models, provide the target platform:

```bash
python -m whisper_rknn.whisper \
  --encoder_model_path ../model/whisper_encoder_base_20s.rknn \
  --decoder_model_path ../model/whisper_decoder_base_20s.rknn \
  --task en \
  --audio_path ../model/test_en.wav \
  --target rk3588
```

Use `--task zh` with `../model/test_zh.wav` for the bundled Chinese example. The RKNN workflow requires a compatible target accessible to RKNN Toolkit. Pass `--device_id <device-id>` when a specific connected device must be selected.

Pass `--enable-timestamps` to include Whisper segment timestamp markers in the Python demo output. The Python demo also accepts these model-shape options:

| Option                         | Default | Description                                      |
|--------------------------------|---------|--------------------------------------------------|
| `--chunk_length <seconds>`     | `20`    | Audio window to preprocess, in whole seconds.    |
| `--max_tokens <count>`         | `12`    | Fixed decoder token input length; minimum is 4.  |

Both values must match the tensor shapes of the encoder-decoder model pair. For example, the bundled 20-second models should use `--chunk_length 20`, and a decoder exported with `--max_tokens 12` should be run with `--max_tokens 12`.

### Export ONNX Models

The `whisper_rknn.export_onnx` module exports separate, fixed-shape Whisper encoder and decoder graphs and simplifies and validates both graphs. The public wrapper invokes it automatically. It can also be run directly inside the Python container:

```bash
python -m whisper_rknn.export_onnx \
  --model-type base \
  --chunk-length 20 \
  --n-mels 80 \
  --max-tokens 12
```

The exporter downloads the requested OpenAI Whisper checkpoint when it is not already cached, so the container must have network access on the first run. It writes the resulting pair to:

```text
model/whisper_encoder_base_20s.onnx
model/whisper_decoder_base_20s.onnx
```

The paths are relative to the repository root. The exporter accepts whole-second windows from 10 to 30 seconds and slices the encoder's positional embeddings to the selected fixed input length; it does not modify the installed `openai-whisper` package. The `--n-mels` option defaults to `80`. `--max-tokens` sets the decoder's fixed token input length and defaults to `12`; it cannot exceed the selected model's text context size. Existing files are preserved unless `--force` is passed.

The current application is validated only with the Whisper base model, 80 Mel channels, and a 20-second window. Other model types, Mel counts, or window lengths require compatible model tensor shapes and matching runtime options. Pass the model's window length through `--chunk_length` in Python or `--chunk-length` in C++. Pass the decoder's fixed token input length through `--max_tokens` in Python or `--max-tokens` in C++. The encoder width is `384` for tiny, `512` for base, and `1024` for medium. Treat other configurations as unsupported until their conversion and inference results have been validated.

## Linux CLI

### Build

Build the CLI for a 64-bit Rockchip Linux target using the Docker wrapper:

```bash
./scripts/build-native.sh docker
```

The resulting executable is written to `build-native/whisper_rknn`. Because this is an aarch64 cross-build intended for a Rockchip target, it cannot be run directly on a typical x86 development machine.

Copy the executable, model files, data assets, and RKNN runtime library to the device. For example:

```bash
scp build-native/whisper_rknn user@device:/opt/whisper-rknn/
scp encoder.rknn decoder.rknn user@device:/opt/whisper-rknn/model/
scp model/mel_80_filters.txt model/vocab_en.txt model/vocab_zh.txt \
  user@device:/opt/whisper-rknn/model/
scp thirdparty/rknpu2/Linux/aarch64/librknnrt.so \
  user@device:/opt/whisper-rknn/lib/
```

Ensure that the dynamic linker can locate `librknnrt.so` on the target. One option is to set `LD_LIBRARY_PATH` before running the executable:

```bash
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

### Run

The CLI accepts the encoder, decoder, transcription task, and input audio path as positional arguments:

```text
./whisper-rknn [--disable-neon] [--enable-timestamps] \
  [--chunk-length <seconds>] [--max-tokens <count>] \
  <encoder_path> <decoder_path> <task> <audio_path>
```

Pass `--disable-neon` to use the scalar Mel-spectrogram matrix multiplication implementation. This is useful for comparing its output with the default Arm NEON implementation. Pass `--enable-timestamps` to include Whisper segment timestamp markers. The `--chunk-length` option defaults to 20 seconds, and `--max-tokens` defaults to 12 with a minimum of 4. Both values must match the model tensor shapes.

Supported tasks are:

| Task | Language |
|------|----------|
| `en` | English  |
| `zh` | Chinese  |

For English transcription:

```bash
./whisper_rknn \
  model/whisper_encoder_base_20s.rknn \
  model/whisper_decoder_base_20s.rknn \
  en \
  audio/example.wav
```

For Chinese transcription, use `zh` and ensure that `model/vocab_zh.txt` is available.

The audio is converted from stereo to mono when needed and resampled to 16 kHz. Audio longer than the configured chunk length is truncated to that window; the default is the first 20 seconds. On success, the CLI prints the recognized text, inference time, audio duration, and real-time factor.

## Android

Build the Android CLI with:

```bash
./scripts/build-android.sh docker
```

The script targets Android API 34 and the `arm64-v8a` ABI. Its Docker image installs the expected Android SDK, NDK, CMake, and Ninja versions, and the resulting executable is written to `build-android/whisper_rknn`.

This repository currently builds a command-line executable rather than an APK. Deploy the executable and its model/data assets to a compatible Android device with `adb`, then invoke it from a shell where the RKNN runtime library is available.

## CMake Configuration

The build scripts are the supported entry points for configuring and building the project. They accept an optional build type, such as `Debug`:

```bash
./scripts/build-native.sh docker Debug
./scripts/build-android.sh docker Debug
```

The following cache variables can be used when extending or adapting the build:

| Variable                     | Default                        | Description                                      |
|------------------------------|--------------------------------|--------------------------------------------------|
| `RKNN_INCLUDE_DIR`           | Bundled RKNN include directory | Directory containing `rknn_api.h`.               |
| `RKNN_RUNTIME_LIB`           | Bundled target runtime         | Path to the target's `librknnrt.so`.             |
| `ENABLE_ASAN`                | unset                          | Enable AddressSanitizer flags for debug builds.  |

## Development Status

This repository is under active development. The current batch CLI establishes the basic inference and cross-build path; planned work includes:

* Microphone input and continuous transcription.
* A streaming pipeline with incremental decoding.
* Broader model-conversion and validation tooling.
* Broader test audio and decoder performance reporting.
* Android application integration.

More detailed implementation notes are tracked in [`PLAN.md`](./PLAN.md) and under [`notes/`](./notes/).

## License

Except where otherwise noted, the source code in this repository is licensed under the [Apache License, Version 2.0](./LICENSE).

Third-party components under `thirdparty/`, model files, datasets, and other externally sourced assets are not covered by this repository's Apache-2.0 license. They remain subject to their respective copyright notices and license terms. In particular, distributions that include or link the bundled RKNN Runtime, FFTW, or libsndfile components must comply with the applicable third-party terms.
