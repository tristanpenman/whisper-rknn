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

## LocalAgreement and Whisper-Streaming

This is approximately the approach taken by [Whisper-Streaming](https://github.com/ufal/whisper_streaming), which introduces a *LocalAgreement* policy. A prefix is committed only after consecutive decoding iterations agree on it.

The project reported about 3.3 seconds average latency while maintaining high transcription quality on long-form speech.

Whisper-Streaming has since been superseded by [SimulStreaming](https://github.com/ufal/SimulStreaming), which combines LocalAgreement with newer attention-based streaming techniques.

The goal of this project is to apply these techniques to the Rockchip NPU.

## Overlapping Context

A naive implementation might process independent chunks:

```text
0–1s    → Whisper
1–2s    → Whisper
2–3s    → Whisper
```

This tends to work poorly because Whisper loses linguistic context and words can be split across chunk boundaries.

Instead, overlapping or rolling context can be used:

```text
0–2s       → decode
0–3s       → decode
1–4s       → decode
2–5s       → decode
...
```

Previously committed text can also be retained as context.

In practice, the active buffer might contain **5–15 seconds** of audio even though an update is requested every **500 ms–1 s**. This gives the user incremental results without forcing Whisper to understand isolated 500 ms fragments.

The window should not grow indefinitely:

```text
0–2
0–3
0–4
0–5
...
0–120
```

Otherwise the system continually recomputes an increasingly expensive input. Once text is safely committed, old audio can be discarded.

## Voice Activity Detection

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

## Lower-Latency Approaches

More recent work specifically targets Whisper's streaming limitations.

### Simul-Whisper

[Simul-Whisper](https://arxiv.org/abs/2406.10052) uses Whisper's cross-attention alignment to determine when generated tokens have enough audio evidence. It also handles words truncated by chunk boundaries.

The authors report only 1.46 percentage points absolute WER degradation with 1-second chunks.

### Whisper-T

[Whisper-T](https://arxiv.org/abs/2412.11272) targets efficient streaming on constrained hardware. It combines decoder optimizations, buffer alignment, and CPU/GPU pipelining.

Reported results include approximately 0.5-second per-word delay in some configurations and about 1 second latency per word at approximately 7 W on a MacBook Air.

These approaches show that sub-second-ish Whisper interaction is possible, although it requires more than simply invoking ordinary Whisper on 500 ms chunks.

## Compute Requirements

For edge deployment, repeated processing of overlapping context can become the dominant problem.

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

The optimization target therefore is not simply:

> Whisper must run faster than realtime.

It is closer to:

> **Incremental Whisper inference must process each update faster than the update interval.**

For example:

```text
chunk interval:       500 ms
Whisper inference:    180 ms
```

leaves substantial headroom.

By contrast:

```text
chunk interval:       500 ms
Whisper inference:    650 ms
```

means transcription will progressively fall behind unless work is skipped, parallelized, or otherwise reduced.

## RK3588 Architecture

For an RK3588 implementation, a reasonable starting architecture is:

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

## Encoder Reuse

A major architectural question on the RK3588 is whether encoder computation can be reused rather than repeatedly encoding the overlapping audio window.

Vanilla Whisper does not naturally provide efficient streaming encoder-state reuse. Repeatedly processing overlapping windows can therefore waste substantial compute.

This is one reason genuinely streaming ASR architectures, such as **Conformer**, **RNN-T**, and **TDT** based systems, can be better suited to continuous low-latency transcription.

If the requirement is specifically to run a **Whisper-compatible model on RK3588**, SimulStreaming and LocalAgreement are useful starting points.

If the actual requirement is high-quality multilingual continuous ASR on RK3588 with less than approximately one second of latency, Whisper should be compared against models designed for streaming rather than assumed to be the best architecture.
