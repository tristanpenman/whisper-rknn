# Whisper RKNN

This repo contains a starter CMake project for running Whisper speech recognition models on Rockchip devices via RKNN. This code has been adapted from Rockchip's [rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo).

The current implementation provides a C++ command-line demo for batch transcription on RK3588 devices running Linux and Android. It loads separate Whisper encoder and decoder RKNN models, prepares audio on the CPU, runs inference through the RKNN runtime, and prints the transcription and real-time factor.

Current capabilities:

* English and Chinese transcription
* WAV-file input through libsndfile
* Mono conversion and resampling to 16 kHz
* Log-Mel spectrogram preprocessing with FFTW and Arm NEON
* Linux/aarch64 and Android/arm64-v8a builds

Streaming transcription, microphone input, Python tooling, and model conversion are planned but are not yet implemented.

### Contents

* [Background](#background)
  * [Whisper](#whisper)
  * [RKNN](#rknn)
* [Project Structure](#project-structure)
  * [Layout](#layout)
  * [Dependencies](#dependencies)
* [Models and Data](#models-and-data)
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

Whisper processes audio in fixed-size windows. The current demo prepares a log-Mel spectrogram for up to 20 seconds of audio, passes it through an RKNN encoder, and then repeatedly invokes an RKNN decoder to produce text tokens.

This is presently a batch pipeline: the complete audio file is read before inference begins. See [Development Status](#development-status) for planned streaming work.

### RKNN

RKNN is Rockchip's model format and NPU inference runtime. The CLI expects a compatible encoder and decoder that have already been converted to `.rknn` files. Model conversion is not currently part of this repository.

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

These assets and the converted RKNN models must be prepared separately. Run the executable from the repository root, or reproduce the same `model/` layout beside the working directory used on the target device.

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
./whisper-rknn [--disable-neon] <encoder_path> <decoder_path> <task> <audio_path>
```

Pass `--disable-neon` to use the scalar Mel-spectrogram matrix multiplication implementation. This is useful for comparing its output with the default Arm NEON implementation.

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

The audio is converted from stereo to mono when needed and resampled to 16 kHz. Audio longer than 20 seconds is currently truncated to the first 20 seconds. On success, the CLI prints the recognized text, inference time, audio duration, and real-time factor.

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
* Python model conversion and validation tools.
* Broader test audio and decoder performance reporting.
* Android application integration.

More detailed implementation notes are tracked in [`PLAN.md`](./PLAN.md) and under [`notes/`](./notes/).

## License

Except where otherwise noted, the source code in this repository is licensed under the [Apache License, Version 2.0](./LICENSE).

Third-party components under `thirdparty/`, model files, datasets, and other externally sourced assets are not covered by this repository's Apache-2.0 license. They remain subject to their respective copyright notices and license terms. In particular, distributions that include or link the bundled RKNN Runtime, FFTW, or libsndfile components must comply with the applicable third-party terms.
