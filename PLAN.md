# Plan

## Inference

- Get RKNN model zoo example working
- Import relevant C++ source from model zoo example
- Transcribe from microphone
- Transcribe from WAV file
- Import Android wrappers from Marian/VLM RKNN projects

## Streaming

- Investigate other streaming implementations
  - [Whisper-Streaming](https://github.com/ufal/whisper_streaming)
  - [SimulStreaming](https://github.com/ufal/SimulStreaming)
- C++ implementation of streaming
- RKNN native implementation of streaming

## Testing

- Sample wave files in different languages
- Print real-time factor stats
- Decoder loop stats

## Conversion

- Basic model conversion script
- Alternative conversion path for streaming (maybe)
- Document conversion process in README
