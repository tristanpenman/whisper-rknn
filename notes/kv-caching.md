# KV caching

This project can potentially use KV caching to improve decoder performance, but not with the decoder model as currently exported.

## Current decoder behavior

The current implementation performs full-prefix decoding for every generated token:

- The decoder has fixed inputs consisting of `tokens[max_tokens]` and the encoder output.
- Every decoding iteration invokes the entire decoder again.
- Self-attention keys and values for earlier tokens are recomputed each time.
- The decoder produces logits for every token position, although the runtime only uses the logits at `tokenCount - 1`.

Consequently, decoder work grows roughly quadratically with the generated sequence length.

## Cached decoder interface

Using KV caching would require a new stateful decoder export whose inputs and outputs include the attention cache. A practical RKNN interface would look approximately like this:

```text
Inputs:
  current token(s)
  encoded audio
  self-attention K/V cache for every decoder layer
  decoding position

Outputs:
  logits for the newest token
  updated self-attention K/V cache
```

Whisper's cross-attention keys and values could also be computed once from the encoder output and reused. This would avoid repeatedly projecting unchanged audio features.

## Possible designs

### Prefill and token decoders

- A prefill graph processes the initial prompt.
- A fixed-shape, one-token graph processes each subsequent token and updates explicit K/V tensors.
- This design offers the largest reduction in decoder computation.

### Chunked fixed-shape decoder

- Small token blocks are processed while fixed-capacity K/V tensors are carried between invocations.
- This design may convert more reliably to RKNN, but padding causes some wasted computation.

## RKNN considerations

- RKNN models generally need static tensor shapes, so caches should be exported with fixed maximum sizes.
- In-place cache mutation should not be assumed. Explicit cache inputs and outputs are safer.
- Dynamic indexing, causal-mask construction, and cache-update operations must be tested for RKNN conversion support.
- Cache tensors can be sizable. Copying every layer's complete cache on each invocation could erode the performance benefit. RKNN tensor-memory APIs or a design that outputs only new K/V slices may be important.
- The C++ and Python inference paths, model-shape discovery, conversion scripts, and decoder tests would all need corresponding changes.

The streaming implementation currently reruns recognition over changing audio windows. KV state is reusable within one autoregressive decode, but generally cannot be carried unchanged between windows because the encoder output, and therefore the cross-attention context, changes.

## Recommendation

KV caching is technically applicable and is likely to be valuable, especially for longer outputs. However, it requires a redesign of the model export and runtime interface rather than a small C++ optimization.

Before implementing it fully, first export a cached-decoder prototype using a small Whisper model, or even a single decoder layer. Use that prototype to verify that RKNN Toolkit can convert the required cache-update operations and that transferring cache tensors does not dominate the resulting performance gain.
