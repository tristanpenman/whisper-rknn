# Streaming

Whisper is not natively a streaming ASR model, but several techniques can make it behave like one with reasonably low latency.

Whisper was designed around audio windows of up to 30 seconds, with the audio encoded before autoregressive decoding. A practical streaming system therefore does not simply feed individual audio frames into Whisper. Instead, it repeatedly transcribes a moving audio window and determines which part of the resulting transcript is stable enough to commit.

## Basic Architecture

A practical pipeline looks roughly like this:

```text
microphone
    │
    ▼
16 kHz PCM
    │
    ▼
VAD / speech detection
    │
    ▼
rolling audio buffer
    │
    ├──── new audio every ~0.5–1 s
    ▼
Whisper
    │
    ▼
candidate transcript
    │
    ▼
stability / alignment logic
    │
    ├── stable prefix ──► committed transcript
    │
    └── unstable suffix ─► reconsider next iteration
```

The difficult part is generally the stability and alignment logic rather than simply making Whisper inference fast.

For example, successive passes might produce:

```text
t=1.0  "I think we should"
t=1.5  "I think we should probably"
t=2.0  "I think we should probably go"
t=2.5  "I think we should probably go home"
```

The system might commit:

```text
I think we should
```

at `t=1.5`, while continuing to revise the remainder.

## Local Agreement

This is approximately the approach taken by [Whisper-Streaming](https://github.com/ufal/whisper_streaming), which introduces a *Local Agreement* policy. A prefix is committed only after consecutive decoding iterations agree on it.

The project reported about 3.3 seconds average latency while maintaining high transcription quality on long-form speech. Whisper-Streaming has since been superseded by [SimulStreaming](https://github.com/ufal/SimulStreaming), which combines LocalAgreement with newer attention-based streaming techniques.

Although SimulStreaming potentially offers better performance, the goal of this project is to implement Local Agreement on the Rockchip NPU.

### RK3588 Architecture

For an RK3588 implementation, a viable Local Agreement architecture could look like this:

```text
Audio capture
16 kHz mono PCM
      │
      ▼
lightweight VAD
      │
      ▼
rolling buffer
~5–10 seconds
      │
      ▼
Whisper encoder
      │
      ▼
decoder
      │
      ▼
token timestamps / alignment
      │
      ▼
LocalAgreement
      │
      ├── committed tokens
      │
      └── provisional tokens
```

Inference could initially run approximately every **500–1000 ms**.

A UI could immediately display provisional output:

```text
I think we should prob...
```

and replace it as later decoding arrives:

```text
I think we should probably go home.
```

The perceived latency can therefore be around **0.5–1.5 seconds**, while the guaranteed stable transcript trails somewhat further behind.

## First Milestone

The first milestone is to replay a WAV file incrementally in one-second steps, produce committed and provisional token sequences, and never duplicate committed output.

### State

The streaming layer will maintain three token sequences:

- `committed`: tokens permanently emitted to the consumer.
- `previous`: the uncommitted hypothesis from the preceding inference pass.
- `current`: the uncommitted hypothesis from the latest inference pass.

Committed output is immutable. Conflicting output remains provisional until two consecutive inference passes agree on it.

Agreement must be calculated using token IDs rather than rendered strings. String comparison is fragile around BPE boundaries and the Chinese vocabulary's Base64 decoding. Therefore, it will be necessary for the decoder to output token IDs in conjunction with the current text output.

### Agreement Policy

For every inference update:

1. Align the new hypothesis with the already committed output.
2. Remove the committed portion from the previous and current hypotheses.
3. Find the longest common token prefix of the remaining hypotheses.
4. Append that prefix to `committed` and emit it once.
5. Discard the conflicting suffix of `previous`.
6. Retain the unmatched suffix of `current` as the next provisional hypothesis.

For example:

```text
previous:     "we need to leave on fri"
current:      "we need to leave on friday morning"
newly stable: "we need to leave on"
provisional:  "friday morning"
```

When the hypotheses conflict, the streaming layer will not choose or combine the alternatives:

```text
previous:     "send it to john tomorrow"
current:      "send it to joan tomorrow"
newly stable: "send it to"
provisional:  "joan tomorrow"
```

If the following pass produces `send it to joan tomorrow morning`, the shared `joan tomorrow` prefix can then be committed. If there is no common prefix, the update emits no committed text and retains the latest hypothesis as provisional.

A basic token-prefix helper is sufficient for the initial policy:

```cpp
std::size_t commonPrefixLength(
    const std::vector<int>& previous,
    const std::vector<int>& current)
{
    std::size_t length = 0;
    while (length < previous.size()
        && length < current.size()
        && previous[length] == current[length]) {
        ++length;
    }
    return length;
}
```

At end-of-stream, perform one final decode and flush its remaining hypothesis as final output. This explicitly bypasses the normal requirement for agreement with another future update.

### Inference Interface

`runDecoder()` currently converts generated tokens directly into cleaned text. It will instead return the generated token IDs as well as display text, for example:

```cpp
struct TranscriptionHypothesis
{
    std::vector<int> tokenIds;
    std::string text;
};
```

The streaming component will be independent of RKNN inference so that its state transitions can be unit-tested on the build host.

Updates exposed to the CLI or another consumer should distinguish permanent and replaceable output:

```cpp
struct StreamingUpdate
{
    std::vector<int> committedTokens;
    std::vector<int> provisionalTokens;
    bool isFinal = false;
};
```

The CLI may redraw provisional text, but downstream consumers should receive committed text only unless they explicitly opt into provisional updates.

### Rolling Audio Buffer

The initial implementation will use a conservative configuration:

- Update interval: one second.
- Minimum audio before the first decode: two seconds.
- Maximum active window: 20 seconds, matching the current encoder model.
- Agreement depth: two consecutive hypotheses.
- Input source: an existing audio file delivered incrementally.

Each update appends samples to the active buffer, preprocesses the window, runs the encoder and decoder, and submits the returned hypothesis to the agreement component.

Independent, non-overlapping chunks must not be concatenated because words may be split at their boundaries. The active buffer must retain overlapping acoustic context between updates.

## Second Milestone

Leverage timestamps to trim the buffer.

### Alignment and Buffer Trimming

Local agreement can compare tokens without timestamps, but timestamps are needed to discard old audio safely. The current decoder prompt includes token `50363`, which in the Whisper model represents the special control token `<|notimestamps|>`. It tells the decoder to output text without adding time markers.

An early experiment must establish whether the current ONNX and RKNN decoder can generate timestamp tokens after changing the prompt, or whether the decoder needs to be re-exported and reconverted.

Once timestamps are available, every committed segment will retain absolute start and end times. A new decode will be compared only after the last committed time:

```text
committed through: 12.4 s
new decode covers: 11.5-20.0 s
comparison range:  tokens after 12.4 s
```

After committing a segment, old audio can be removed up to its confirmed time. Approximately 0.5 to one second of preceding audio should remain as acoustic context. Text generated for time already committed must be ignored even if a later inference pass revises it.

Until timestamp alignment is implemented, the prototype may let its buffer grow to the 20-second limit, but it must not guess an audio trimming point from text length.

After trimming, a bounded tail of committed tokens should be supplied to the decoder as linguistic context. The present decoder input is fixed at `kMaxTokens = 12`, so useful prompt context may require another model export.

## Optimisation

This section looks at potential optimisations, specifically targeting Whisper's streaming limitations.

### Simul-Whisper

[Simul-Whisper](https://arxiv.org/abs/2406.10052) uses Whisper's cross-attention alignment to determine when generated tokens have enough audio evidence. It also handles words truncated by chunk boundaries.

The authors report only 1.46 percentage points absolute WER degradation with 1-second chunks.

### Whisper-T

[Whisper-T](https://arxiv.org/abs/2412.11272) targets efficient streaming on constrained hardware. It combines decoder optimizations, buffer alignment, and CPU/GPU pipelining.

Reported results include approximately 0.5-second per-word delay in some configurations and about 1 second latency per word at approximately 7 W on a MacBook Air.

These approaches show that sub-second-ish Whisper interaction is possible, although it requires more than simply invoking ordinary Whisper on 500 ms chunks.

### Compute Requirements

For edge deployment, repeated processing of overlapping context can become a performance drain.

Suppose decoding occurs every 500 ms using a 10-second rolling window:

```text
0.5 s new audio
       ↓
process ~10 s
       ↓
0.5 s new audio
       ↓
process ~10 s again
```

This effectively processes approximately 20 seconds of audio for every second of new audio.

If the model processes audio at `5× realtime`, then:

```text
20 / 5 = 4 seconds compute per second
```

The system cannot keep up.

### Voice Activity Detection

A Voice Activity Detector (VAD) is useful independently of Whisper. Instead of continuously processing:

```text
speech
speech
speech
silence
silence
silence
speech
...
```

the system can identify speech regions first.

VAD also provides useful boundaries for committing transcription. For example, a 300–600 ms silence can indicate that Whisper has enough future context to commit the preceding words.

A production implementation can therefore combine two commit mechanisms:

```text
LocalAgreement
      │
      ├── successive decodes agree → commit
      │
      └── uncertain
             │
             ▼
        speech continues

VAD detects pause
      │
      ▼
aggressively commit preceding phrase
```

This can improve perceived latency compared with relying entirely on a fixed delay.
