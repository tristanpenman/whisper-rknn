# Plan

## Inference

- ~~Get RKNN model zoo example working~~
- ~~Import relevant C++ source from model zoo example~~
- ~~Transcribe from WAV file~~
- ~~Import Android wrappers from Marian/VLM RKNN projects~~
- Transcribe from microphone

## Streaming

- ~~Investigate other streaming implementations~~
  - ~~[Whisper-Streaming](https://github.com/ufal/whisper_streaming)~~
  - ~~[SimulStreaming](https://github.com/ufal/SimulStreaming)~~
- ~~RKNN implementation of streaming~~
- Vanilla C++ implementation of streaming

## Fallbacks

- Reimplement a local-agreement fallback for stalled streaming
- Force-commit the latest hypothesis after ten seconds of buffered audio (capped by the model window)
  - Advance and reset agreement
  - Retain five-second updates and normal agreement by default
- Handle empty hypotheses, boundary context loss, and end-of-input flushing without duplication
- Fix alignment
  - Preserve repeated phrases
  - Avoid double-removing timestamped context
  - Prevent unclosed segments from advancing audio trimming
- Test fallback and window progression
  - Document accuracy tradeoffs and validate both Docker builds and target-device transcription

## Testing

- ~~Print real-time factor stats~~
- Sample wave files in different languages
- Decoder loop stats
- Evaluation scripts

## Conversion

- ~~Basic model conversion script~~
- Document conversion process in README
- Alternative conversion path for streaming (maybe)
