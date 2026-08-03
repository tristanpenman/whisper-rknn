## Matmuls

Before Whisper can transcribe speech, it turns the audio waveform into a spectrogram: a grid that describes the sound energy at different frequencies over time. One step in that process combines the spectrogram's frequency bins into the 80 Mel bands expected by the Whisper encoder. The project performs this step with matrix multiplication.

## The NEON Mel matrix multiplication bug

The project had a hand-optimized implementation of this multiplication for Arm processors using NEON instructions. It ran successfully and produced plausible spectrograms, but it grouped values along the wrong dimension. A recent change corrected that indexing mistake. This note explains where the bug occurred, what the old and new implementations calculate, and why fixing it may not produce an obvious change in every transcription.

## Where the multiplication occurs

The multiplication is part of the audio frontend, before either RKNN model is run:

```text
audio
  -> STFT power spectrum
  -> Mel matrix multiplication
  -> logarithm and normalization
  -> RKNN Whisper encoder
  -> RKNN Whisper decoder
  -> transcription
```

In `computeLogMelSpectrogram()`, `magnitudes` is a row-major matrix with shape `201 x time`. Each row represents one FFT frequency bin and each column represents one time frame. The Mel filters are a row-major `80 x 201` matrix. Multiplying them produces the `80 x time` Mel spectrogram expected by the encoder.

For Mel band `m` and time frame `t`, the required calculation is:

```text
mel[m, t] = sum(filter[m, k] * magnitude[k, t])
```

The summation combines different frequency bins at one fixed point in time.

## What the old implementation calculates

Consider four adjacent frequency bins whose Mel-filter weights are `a`, `b`, `c`, and `d`. The correct contribution to one output value is:

```text
a * magnitude[k,     t]
b * magnitude[k + 1, t]
c * magnitude[k + 2, t]
d * magnitude[k + 3, t]
```

The old implementation loads its right-hand NEON vector from:

```cpp
&right[shared * rightColumns + column]
```

Because `right` is row-major, the four contiguous values at that address are four adjacent time frames from the same frequency row. The old implementation therefore calculates:

```text
a * magnitude[k, t]
b * magnitude[k, t + 1]
c * magnitude[k, t + 2]
d * magnitude[k, t + 3]
```

Only the first SIMD lane uses the intended time-frequency coordinate. The remaining lanes replace an adjacent frequency-bin lookup with an adjacent time-frame lookup.

The old code's search for the first and last nonzero filter weights does not correct this indexing problem. It only limits the range processed for each sparse Mel filter and adjusts that range to a multiple of four.

## What the new implementation calculates

The new implementation vectorizes in the other direction. It loads four adjacent time values for a single frequency bin, multiplies all four by that bin's one Mel-filter weight, and accumulates them into four separate output columns:

```text
output[m, t    ] += filter[m, k] * magnitude[k, t    ]
output[m, t + 1] += filter[m, k] * magnitude[k, t + 1]
output[m, t + 2] += filter[m, k] * magnitude[k, t + 2]
output[m, t + 3] += filter[m, k] * magnitude[k, t + 3]
```

This is mathematically equivalent to the scalar implementation, apart from ordinary floating-point accumulation differences. The scalar tail handles any remaining output columns when the number of time frames is not divisible by four.

## Why the transcription can remain unchanged

Although the old calculation is wrong, it substitutes nearby points rather than unrelated parts of the input. A displaced value is at most three FFT bins away from the intended frequency and three frames, or approximately 30 ms, away in time.

Several properties of the pipeline reduce the visible impact:

- Speech energy is usually correlated across nearby frequencies and nearby time frames.
- Mel filters intentionally combine adjacent FFT frequency bins.
- The logarithm compresses differences in linear spectral energy.
- Dynamic-range clipping and normalization further limit some differences.
- The Whisper encoder is tolerant of modest frontend perturbations.
- Greedy decoding changes the transcript only when a perturbation makes a different token logit become the maximum.

Consequently, the internal Mel values and encoder activations can differ while the selected token sequence remains identical. Once a token does change, the autoregressive decoder can amplify the effect because later predictions depend on earlier selected tokens.

The checked-in Mel filters are also quite narrow. Each filter has between 1 and 14 nonzero coefficients. Across all filters, only 67 of 391 nonzero coefficients occupy the old routine's lane that reads the intended frequency row. The other coefficients nevertheless read locally related spectrogram values, which makes the old result incorrect but potentially usable on easy audio.

Differences are more likely to become visible with low-SNR audio, weak consonants, rapid speech, ambiguous words, unfamiliar accents, or cases where the two highest decoder logits are already close. A few clean samples are unlikely to demonstrate a reliable word-error-rate difference.

## Testing caveats

The following distinctions matter when testing:

- Passing `--disable-neon` disables the NEON code path and instead returns the scalar result
- The current default ARM path uses the fixed NEON implementation. The original bug can be restored by reverting the relevant commit:

  ```
  git revert c0ea1d15cc9b0a613f884af8bdf92d52bc18e9e1
  ```

For a useful accuracy comparison, run both frontend implementations on the same representative audio corpus and retain both their normalized Mel tensors and final transcripts. Direct Mel error measurements reveal whether the old path was actually selected, while corpus-level word error rate measures whether those numerical changes cross the model's token-decision boundaries.
