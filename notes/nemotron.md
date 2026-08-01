# Nemotron

This note considers adapting this repository to run NVIDIA's `nemotron-speech-streaming-en-0.6b` model on a Rockchip NPU through RKNN. The multilingual Nemotron 3.5 ASR model uses a related architecture, but adds language prompting and automatic language detection that would require additional runtime work.

The adaptation is feasible in principle, but it is closer to replacing the Whisper inference engine than swapping models. The CMake configuration, audio-file handling, and generic RKNN model-loading code can be reused. Preprocessing, model export, streaming state, decoding, and likely memory management must be redesigned.

As of August 2026, no published end-to-end implementation of Nemotron ASR on a Rockchip NPU was found. The closest reference is `sherpa-onnx`: it separately implements Nemotron in C++ using ONNX Runtime and streaming transducers using RKNN, but does not appear to combine the two into a Nemotron `.rknn` deployment.

## Architectural differences

| Area          | Current repository                        | Nemotron Speech Streaming                         |
|---------------|-------------------------------------------|---------------------------------------------------|
| Topology      | Whisper encoder and Transformer decoder   | FastConformer encoder, RNN-T predictor and joiner |
| Operation     | Fixed 20-second batch                     | Stateful 80, 160, 560 or 1120 ms chunks           |
| Features      | 80 Mel bins and FFT size 400              | 128 Mel bins and FFT size 512                     |
| Decoder state | Repeated fixed token window               | Recurrent state and blank-token transitions       |
| Encoder state | No state between calls                    | Attention and convolution caches across 24 layers |
| Model size    | Whisper base                              | Approximately 600 million parameters              |
| Model graphs  | Encoder and decoder                       | Encoder, predictor and joiner                     |

The current implementation hard-codes Whisper tensor dimensions, vocabulary size, special tokens, and preprocessing constants. Its encoder takes one fixed-size Mel spectrogram and produces one fixed-size output. Its decoder repeatedly processes a fixed token window and selects one token with `argmax`. These interfaces do not match an RNN-T recognizer.

NVIDIA describes Nemotron Speech Streaming as a 24-layer cache-aware FastConformer with an RNN-T decoder. Each audio chunk carries encoder attention and convolution caches into the next invocation. See the [NVIDIA model card](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b) for the model architecture and supported streaming configurations.

## Model export

The native model is distributed as a NeMo/PyTorch checkpoint rather than an ONNX or RKNN model. The most useful existing exporter is the [sherpa-onnx Nemotron export script](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/nemotron-speech-streaming-en-0.6b/export_onnx.py). It exports:

- `encoder.onnx`
- `decoder.onnx`
- `joiner.onnx`
- `tokens.txt`

The script exposes the encoder cache dimensions and produces separate models for the supported chunk durations. This is a good starting point because it turns the NeMo checkpoint into three explicit inference graphs with external state.

The exported ONNX files are not guaranteed to convert directly with RKNN Toolkit2. In particular, the dynamically quantized `*.int8.onnx` models use ONNX Runtime-oriented quantization. A safer initial route is to export the unquantized ONNX graphs and let RKNN Toolkit2 produce FP16 or INT8 RKNN models.

RKNN generally works most reliably with static tensor shapes. The first prototype should select one chunk size, probably 560 or 1120 ms, and export fixed feature and cache dimensions. Every operator and tensor layout must then be checked against the support restrictions of the exact RKNN Toolkit2 version being used.

## Streaming state

Nemotron's encoder consumes new feature frames and its previous attention and convolution caches. It returns encoded frames and updated caches for the next chunk. The runtime therefore needs persistent storage for at least:

- Attention cache, commonly exported as `cache_last_channel`.
- Convolution cache, commonly exported as `cache_last_time`.
- Cache-length and processed-frame bookkeeping.
- Predictor recurrent hidden and cell state.
- The previous non-blank output token.

The first chunk, subsequent chunks, final partial chunk, stream reset, and endpoint reset all require distinct handling. This differs substantially from the current stateless encoder call.

Nemotron offers a choice between latency and recognition accuracy. NVIDIA's published configurations use 80, 160, 560, and 1120 ms chunks. Smaller chunks submit work to the NPU more frequently and place greater pressure on state transfer and invocation overhead. A 560 ms configuration is a reasonable starting point for a feasibility prototype.

## RNN-T decoding

The current Whisper decoder loop cannot be reused. For each encoded time step, an RNN-T greedy decoder must:

1. Run the predictor using the previous emitted token and recurrent state.
2. Combine the encoder and predictor outputs in the joiner.
3. Select the most likely output symbol.
4. Advance to the next encoder frame when the symbol is blank.
5. Otherwise emit the symbol, update predictor state, and allow another symbol to be emitted from the same encoder frame.

The decoder needs a maximum-symbols-per-frame guard to prevent an infinite non-blank loop. A complete streaming recognizer also needs endpoint detection, token timestamp handling, stream reset, partial-result management, and optionally beam search or hotword support.

The [sherpa-onnx streaming C API documentation](https://k2-fsa.github.io/sherpa/onnx/c-api/html/online_asr.html) shows how the three Nemotron graphs and token file are configured. Its C++ implementation is the best available reference for the required state machine and decoding behavior.

## Feature extraction

Nemotron's published export configuration uses:

- 16 kHz mono audio.
- 128 Mel filters.
- FFT size 512.
- 25 ms Hann windows.
- 10 ms frame stride.
- A small amount of dither.
- NeMo-compatible logarithm, padding, and normalization behavior.

Matching the reference feature extractor numerically is important. Small differences can affect both recognition quality and quantization calibration. Rather than incrementally modifying the Whisper-specific Mel implementation, it would be safer to use `kaldi-native-fbank`, as sherpa-onnx does, or port its exact feature configuration and validate the resulting tensors against NeMo.

## Memory and scheduling

Nemotron is much larger than Whisper base. Sherpa's multilingual Nemotron package has an approximately 627 MB INT8 encoder, plus its predictor and joiner. The English model is in the same general size class. See the [sherpa-onnx Nemotron model documentation](https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html) for published package sizes.

The current runtime allocates input buffers and copies tensors for every model invocation. Copying every attention and convolution cache through host memory for each short audio chunk could erase much of the benefit of cache-aware streaming. A performant port will likely need persistent buffers, RKNN tensor memory APIs where practical, and profiling of both copies and NPU submission overhead.

The predictor and joiner are relatively small and are invoked frequently. They may be faster on the CPU than on the NPU once submission and transfer overhead is included. A hybrid implementation with the FastConformer encoder on RKNN and the predictor and joiner on the CPU should be benchmarked rather than assuming that all three graphs belong on the NPU.

## Quantization

Initial conversion should use FP16. Intermediate features, encoder outputs, updated caches, predictor state, joiner logits, and final transcripts should be compared with NeMo or ONNX Runtime before introducing INT8 quantization.

INT8 calibration must use representative streaming state. A calibration set containing only first chunks with zero-filled caches would not represent steady-state encoder execution. It should include:

- First, middle, and final chunks.
- Non-zero attention and convolution caches.
- Speech, silence, background noise, and speaker variation.
- Short utterances and sustained speech.
- Predictor inputs and recurrent states representative of real decoding.

Encoder, predictor, and joiner quantization should be evaluated independently. It may be necessary to leave numerically sensitive outputs or entire small graphs in FP16 or on the CPU.

## Reference implementations

The following projects provide the most relevant pieces of an implementation:

- [Sherpa Nemotron ONNX exporter](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/nemo/nemotron-speech-streaming-en-0.6b/export_onnx.py) provides graph splitting, tensor metadata, feature parameters, and fixed streaming configurations.
- [Sherpa Nemotron documentation](https://k2-fsa.github.io/sherpa/onnx/nemo/nemotron-streaming.html) provides pre-exported models and runnable CPU and Android examples.
- [Sherpa streaming C API](https://k2-fsa.github.io/sherpa/onnx/c-api/html/online_asr.html) demonstrates configuring the encoder, predictor, joiner, tokens, and greedy decoder.
- [Sherpa RKNN transducer backend](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/rknn/online-zipformer-transducer-model-rknn.cc) demonstrates streaming transducer execution through RKNN. It is Zipformer-specific and cannot simply load Nemotron, but it is the closest reference for RKNN model state and decoder scheduling.
- [Sherpa RKNN model documentation](https://k2-fsa.github.io/sherpa/onnx/rknn/models.html) documents working RK3588 streaming-transducer deployments and RKNN runtime compatibility issues.
- [Rockchip RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) contains Whisper, Wav2Vec2, and Zipformer examples, but currently lists no Nemotron model.
- [RKNN Toolkit2 operator support](https://github.com/rockchip-linux/rknn-toolkit2/blob/master/doc/RKNN-Toolkit2_OP_Support-1.6.0.md) is a useful initial compatibility reference. The detailed operator-support document shipped with the selected Toolkit2 release should be treated as authoritative.

## Feasibility

The largest uncertainty is whether the cache-aware FastConformer encoder can be converted accurately and run efficiently on RKNN. That should be tested before building a complete replacement CLI.

1. Export the English Nemotron model with sherpa-onnx for one fixed chunk size, initially 560 ms.
2. Run the exported ONNX model through sherpa-onnx on the RK3588 CPU and record reference transcripts, intermediate outputs, and real-time factor.
3. Convert only the encoder to an FP16 RKNN model.
4. Compare every encoded output and updated cache against ONNX Runtime for the first chunk and several subsequent chunks.
5. Keep the RNN-T predictor and joiner on the CPU for the first complete prototype.
6. Once transcription matches, benchmark converting the predictor and joiner to RKNN against leaving them on the CPU.
7. Replace repeated allocations and copies with persistent tensor buffers, then profile streaming latency.
8. Attempt INT8 conversion only after the FP16 pipeline is correct, using representative cached-state calibration data.

The lowest-risk implementation strategy is to reuse sherpa-onnx's feature extraction and Nemotron decoding logic and add a Nemotron-specific RKNN encoder backend. Evolving the current Whisper-specific decoder directly would require reimplementing functionality that sherpa-onnx already provides and validates.
