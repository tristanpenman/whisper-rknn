# Explainer

_How Whisper works in this project_

Whisper is an encoder-decoder speech recognition model. The encoder turns a compact representation of an audio signal into learned audio features, and the decoder consumes those features while generating text one token at a time. In this project, ordinary CPU code reads and prepares the audio, while Rockchip's RKNN runtime executes the neural-network encoder and decoder on a supported Rockchip NPU.

The implementation is a batch transcription pipeline rather than a streaming one. It reads an entire file, uses at most the configured audio chunk, constructs one log-Mel spectrogram with the corresponding fixed model shape, runs the encoder once, and repeatedly runs the decoder until it produces an end-of-text token or reaches a safety limit. The chunk length defaults to 20 seconds, and the decoder input defaults to 12 tokens.

At a high level, data moves through the program like this:

```text
audio file
    |
    | libsndfile: decode to floating-point PCM samples
    v
mono, 16 kHz waveform
    |
    | Hann windows + FFTW short-time Fourier transforms
    | power spectrum + Mel filter bank + log scaling
    | NEON accelerates CPU vector operations
    v
80 x (100 * chunk length) log-Mel spectrogram
    |
    | RKNN Whisper encoder
    v
(50 * chunk length) x 512 learned audio features
    |
    | RKNN Whisper decoder, invoked repeatedly
    v
token IDs
    |
    | vocabulary lookup and text cleanup
    v
transcription
```

## Program entry point and configuration

[`cpp/src/main.cc`](../cpp/src/main.cc) coordinates the pipeline. The command line supplies an encoder model, a decoder model, a task (`en` or `zh`), and an audio file. The task selects both a vocabulary file and a language token. English uses `model/vocab_en.txt` and token 50259; Chinese uses `model/vocab_zh.txt` and token 50260. The optional `--chunk-length` and `--max-tokens` arguments select the audio window and decoder input length. Their Python equivalents are `--chunk_length` and `--max_tokens`.

The signal-processing constants live in [`cpp/src/process.h`](../cpp/src/process.h). The important values are a 16 kHz sample rate, a 400-sample FFT window, a 160-sample hop, 201 real-FFT frequency bins, and 80 Mel bands. At 16 kHz, the window covers 25 ms and the hop advances by 10 ms. For a chunk length of `C` seconds, preprocessing produces `100C` time steps, giving the encoder an `80 x 100C` input. The encoder output contains `50C x 512` floating-point features for the supported base model. With the default `C = 20`, those shapes are `80 x 2000` and `1000 x 512`.

`main()` allocates the spectrogram from the selected chunk length, along with the Mel filter, vocabulary, and result buffers, before doing any inference. It then performs four broad stages: load and normalize the waveform, load the filter bank and vocabulary, initialize both RKNN models, and preprocess plus transcribe the audio. The final timer output reports a real-time factor: the combined preprocessing and inference time divided by the processed audio duration. A value below 1 means that timed stage was faster than real time.

## 1. Reading and normalizing audio

The audio utilities are declared in [`cpp/src/audio_utils.h`](../cpp/src/audio_utils.h) and implemented in [`cpp/src/audio_utils.c`](../cpp/src/audio_utils.c). They represent decoded audio with an `AudioBuffer`: an owned array of `float` samples plus frame count, channel count, and sample rate.

### libsndfile

libsndfile is the file-format boundary. `readAudio()` calls `sf_open()` to inspect and open the input, copies the reported frame count, channel count, and sample rate into the `AudioBuffer`, and uses `sf_readf_float()` to decode the file directly into interleaved floating-point PCM samples. This keeps container parsing and sample-format conversion out of the Whisper-specific code. The demo is documented around WAV input, although libsndfile itself supplies the decoding API rather than the program parsing WAV bytes.

libsndfile does not perform the Whisper preprocessing. After loading, `main()` calls project-owned helpers to put the waveform into the form expected by the model. If the input has exactly two channels, `convertChannels()` averages each left/right pair to produce mono audio. If the sample rate is not 16 kHz, `resampleAudio()` performs linear interpolation at the desired sample positions. The conversion to mono happens first, so the resampler operates on one sample per frame.

The `saveAudio()` helper also uses libsndfile and can write floating-point WAV data, but it is not part of the current transcription path.

## 2. Constructing the log-Mel spectrogram

Whisper does not feed raw waveform samples directly to its encoder. It first converts the waveform into a log-Mel spectrogram: a two-dimensional array that describes how much energy exists in perceptually spaced frequency bands over time. This work is implemented by `preprocessAudio()` and its private helpers in [`cpp/src/process.cc`](../cpp/src/process.cc).

`main()` first loads `model/mel_80_filters.txt` with `readMelFilters()`. This file contains the precomputed `80 x 201` Mel filter bank expected by the model. Each of its 80 rows combines energy from nearby FFT frequency bins into one Mel-frequency band.

`preprocessAudio()` limits the input to `chunkLength * kSampleRate` samples. Longer files are truncated rather than split into multiple chunks. Shorter inputs produce fewer spectrogram columns at first; `padMelSpectrogram()` copies those columns into a zero-initialized encoder buffer with `80 x (100 * chunkLength)` values. The default 20-second window therefore accepts 320,000 samples and produces an `80 x 2000` buffer.

The actual feature extraction in `computeLogMelSpectrogram()` follows these steps:

1. `makeHannWindow()` creates a 400-value Hann window. Multiplying each frame by this smooth curve reduces sharp discontinuities at frame boundaries and therefore reduces spectral leakage.
2. `reflectPad()` adds 200 reflected samples at both ends of the waveform. This centers analysis windows near the boundaries without introducing an abrupt block of artificial silence.
3. The code takes overlapping 400-sample frames, 160 samples apart, multiplies each frame by the Hann window, and computes a real-to-complex FFT.
4. `transposeComplex()` reorganizes the FFT output so subsequent loops see frequency-major data.
5. `computeMagnitudes()` converts every complex FFT value `(real, imaginary)` into power with `real^2 + imaginary^2`.
6. The `80 x 201` Mel filter bank is multiplied by the `201 x time` power spectrum. This produces energy values for 80 Mel bands at each time step.
7. `clampAndNormalize()` clamps energy away from zero, applies base-10 logarithms, limits the dynamic range to 8 below the maximum log energy, and applies Whisper's `(value + 4) / 4` scaling.

The result is not a picture even though it is often visualized as one. It is a row-major floating-point tensor whose axes are Mel frequency and time, shaped exactly as the exported encoder expects.

### FFTW

FFTW performs the frequency analysis at the heart of the short-time Fourier transform. `computeStftsNeon()` allocates FFTW-aligned input and output buffers with `fftwf_malloc()`, creates a single-precision real-to-complex plan with `fftwf_plan_dft_r2c_1d()`, and reuses that plan for every overlapping frame. Each call to `fftwf_execute()` transforms 400 real waveform values into 201 complex frequency bins. The `fftwf_` prefix is significant: this project links the single-precision `libfftw3f.a`, matching the `float` tensors used elsewhere.

FFTW is responsible only for the transforms. The project code handles framing, windowing, padding, power calculation, Mel filtering, and log normalization around it.

### Arm NEON extensions

NEON is Arm's SIMD instruction set: one instruction can operate on several numeric lanes at once. Because the target devices are Arm-based, [`cpp/src/process.cc`](../cpp/src/process.cc) currently defines `ENABLE_NEON` as 1 and includes `<arm_neon.h>`.

The active STFT path uses NEON intrinsics such as `vld1q_f32`, `vmulq_f32`, and `vst1q_f32` to load, multiply, and store four waveform/window values at a time before FFTW executes the transform. The Mel-filter multiplication uses four-lane multiply-accumulate operations through `vmlaq_f32`, followed by a horizontal sum. NEON therefore accelerates CPU-side arithmetic surrounding inference; it neither decodes the audio file nor replaces FFTW or the Rockchip NPU.

The source also contains scalar STFT code behind the disabled `#else` branch. `ENABLE_NEON` is currently a source-level macro, not a CMake option, so the checked-in executable is explicitly tied to a compiler and target that provide Arm NEON intrinsics.

## 3. Loading the encoder and decoder

Whisper's learned neural network is split into two `.rknn` files. [`cpp/src/whisper.cc`](../cpp/src/whisper.cc) wraps both with the same `RknnAppContext`, while `RknnWhisperContext` in [`cpp/src/whisper.h`](../cpp/src/whisper.h) owns one context for the encoder and one for the decoder.

`initializeWhisperModel()` calls `rknn_init()` to load a model and create an RKNN context. It then queries the number and attributes of all input and output tensors with `rknn_query()`, prints their shapes, formats, types, and quantization information, and retains copies of that metadata. This is useful for diagnosing whether the supplied model files agree with the selected chunk length and token count, although the runtime does not derive those values from the model metadata automatically.

The RKNN runtime, provided by `librknnrt.so`, is the bridge to Rockchip's execution stack. The C++ program supplies host buffers through `rknn_inputs_set()`, starts an inference with `rknn_run()`, and retrieves outputs with `rknn_outputs_get()`. The converted RKNN graphs contain the actual Whisper transformer layers and weights; this repository's C++ code orchestrates those graphs rather than implementing transformer attention itself.

## 4. Encoding the audio

`runWhisperInference()` first calls the private `runEncoder()` function. For a chunk length of `C` seconds, `runEncoder()` copies the entire `80 x 100C` floating-point log-Mel spectrogram into one RKNN input, runs the encoder once, requests a floating-point output, and copies `50C x 512` values into an intermediate buffer. The default 20-second configuration uses `80 x 2000` input values and `1000 x 512` output features.

Conceptually, Whisper's encoder first uses learned convolutional layers to project the spectrogram into its internal width and reduce the time dimension. Transformer encoder blocks then use self-attention, allowing each position to incorporate information from the rest of the audio window. The resulting features preserve information the decoder needs to predict text while reducing `100C` spectrogram time steps to `50C` feature positions. In the default configuration, this reduces 2,000 steps to 1,000 positions. All of those learned encoder operations are inside the supplied RKNN model rather than expressed in the C++ source.

## 5. Generating tokens with the decoder

`runDecoder()` receives the encoder features as its second RKNN input. Its first input is a fixed window of `maxTokens` integer token IDs, defaulting to 12. The initial prompt contains the start-of-transcript token, the selected language token, token 50359, and, unless timestamps are enabled, token 50363. These special tokens configure the transcription mode expected by the exported decoder. The minimum accepted value is 4 so the full non-timestamp prompt fits.

The decoder is autoregressive: each invocation predicts what should come next from the audio features and the current token window. Inside the RKNN decoder graph, masked self-attention relates each token to the earlier token context, cross-attention draws relevant information from the encoder's audio features, and a final projection produces vocabulary scores, or logits. `argmax()` examines the `kVocabSize` logits for the final position and selects the highest-scoring token. This is greedy decoding: the program does not use a beam search, sampling, temperature fallback, or confidence-based retry.

The selected token's text is appended through the loaded vocabulary and placed after the populated prompt and output tokens. The next decoder invocation reads logits from the last populated position. Decoding stops when it selects token 50257 (`<|endoftext|>`) or fills the configured token input. The encoder output remains unchanged and is passed to every decoder invocation. A larger `maxTokens` value provides more decoding context, but it can only be used with a decoder model exported for that same input length.

Finally, `runDecoder()` replaces the vocabulary's `Ġ` word-boundary marker with a normal space, removes the textual end marker and newlines, and Base64-decodes the accumulated Chinese representation when the Chinese task was selected. The resulting string is returned to `main()`, which prints it.

## 6. Cleanup and lifetime

RKNN output buffers are returned to the runtime with `rknn_outputs_release()` after use. `releaseWhisperModel()` frees the copied tensor metadata and destroys each RKNN context with `rknn_destroy()`. `main()` also frees the libsndfile-backed waveform allocation and every duplicated vocabulary token. This explicit cleanup reflects the C-style ownership boundaries used by libsndfile, FFTW, and RKNN.

## What runs where

The pipeline deliberately divides work between general-purpose Arm CPU code and the Rockchip inference runtime:

| Work                                                | Implementation                                   | Typical processor              |
|-----------------------------------------------------|--------------------------------------------------|--------------------------------|
| Decode the audio container into samples             | libsndfile in `audio_utils.c`                    | CPU                            |
| Stereo-to-mono conversion and linear resampling     | Project loops in `audio_utils.c`                 | CPU                            |
| Windowing and Mel matrix arithmetic                 | NEON path in `process.cc`                        | Arm CPU SIMD                   |
| Short-time Fourier transforms                       | FFTW single-precision API in `process.cc`        | CPU                            |
| Whisper encoder and decoder transformer graphs      | RKNN models through `whisper.cc`                 | Rockchip execution runtime/NPU |
| Greedy token selection and text assembly            | Project code in `process.cc` and `whisper.cc`    | CPU                            |

## Current boundaries

This code is a deliberately small demonstration, so its boundaries are worth keeping in mind:

- It is batch-only: the whole audio file is loaded before preprocessing or inference begins.
- It processes no more than the configured chunk length, which defaults to the first 20 seconds, and does not segment longer recordings.
- It explicitly converts stereo input, but it does not define a conversion for channel counts other than one or two.
- Resampling is simple linear interpolation rather than a band-limited production-quality resampler.
- Decoding uses the configured fixed token input, defaulting to 12 tokens, so it cannot generate beyond the remaining context after the initial prompt.
- English and Chinese are the only task choices wired into the CLI.
- The exported encoder and decoder must match the selected chunk length and maximum token count, as well as the special-token conventions in `main.cc` and `whisper.cc`.

Within those constraints, the implementation contains the complete path from an audio file to printed text: libsndfile supplies samples, FFTW and CPU vector code turn them into the features Whisper expects, RKNN runs the learned encoder and decoder, and the vocabulary layer turns the decoder's token choices into a transcription.
