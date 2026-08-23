# FFT

Whisper uses [FFTW](https://www.fftw.org/), which is unfortunately GPL licensed.

## Analysis

Specifically, Rockchip's Whisper implementation relied on these functions, all localised to `process.cc`:

* `fftwf_malloc`
* `fftwf_plan_dft_r2c_1d`
* `fftwf_execute`
* `fftwf_free`
* `fftwf_destroy_plan`

These are called from `computeStftsNeon` and `computeStfts`. Which version is called is determined by whether NEON extensions are enabled.

`fftwf_plan_dft_r2c_1d` is called once per invocation. It plans a 1-D real-to-complex FFT operation on a single, contiguous data sequence:

```cpp
auto* input = static_cast<float*>(fftwf_malloc(sizeof(float) * windowLength));
auto* output = static_cast<fftwf_complex*>(fftwf_malloc(sizeof(fftwf_complex) * (windowLength / 2 + 1)));
const fftwf_plan plan = fftwf_plan_dft_r2c_1d(windowLength, input, output, FFTW_ESTIMATE);

for (int frame = 0; frame < numFrames; ++frame) {
    /* Execute */
}
```

`fftwf_execute` is called once for each frame in both functions. What differs is the setup prior to `fftwf_execute`. When NEON extensions are enabled, setup looks like this:

```cpp
    const int start = frame * hopLength;
    for (int i = 0; i < windowLength - 3; i += 4) {
        if (start + i < audioLength) {
            const float32x4_t samples = vld1q_f32(audio.data() + start + i);
            const float32x4_t windowValues = vld1q_f32(window.data() + i);
            vst1q_f32(input + i, vmulq_f32(samples, windowValues));
        } else {
            vst1q_f32(input + i, vdupq_n_f32(0.0f));
        }
    }

    for (int i = windowLength - windowLength % 4; i < windowLength; ++i) {
        if (start + i < audioLength) {
            input[i] = audio[start + i] * window[i];
        } else {
            input[i] = 0.0f;
        }
    }
```

With NEON disabled:

```cpp
    const int start = frame * hopLength;
    for (int i = 0; i < windowLength; ++i) {
        if (start + i < audioLength) {
            input[i] = audio[start + i] * window[i];
        } else {
            input[i] = 0.0f;
        }
    }
```

In both cases, execution looks like this:

```cpp
    fftwf_execute(plan);
    std::memcpy(
        stftResult + frame * (windowLength / 2 + 1),
        output,
        sizeof(fftwf_complex) * (windowLength / 2 + 1));
```

The NEON loops perform the same windowing operation four samples at a time, while the regular loop can be understood as follows:

> This prepares one overlapping chunk of audio for the FFT—one frame of the short-time Fourier transform (STFT).
>
> * `start` is where the current frame begins in the audio.
> * `windowLength` is 400 samples (25 ms at 16 kHz), so this loop processes 400 samples. Set via `kFftSize = 400`.
> * `hopLength` is 160 samples (10 ms), therefore consecutive frames overlap by 400 - 160 = 240 samples. Set via `kHopLength = 160`.
> * `window` is a Hann window that fades the beginning and end of each frame toward zero. Constructed by `makeHannWindow`.

The plan generated with `fftwf_plan_dft_r2c_1d` assumes a window length of 400 samples, in the variable `input`. Output will be written to `output`.

The flag `FFTW_ESTIMATE` specifies that, instead of actual measurements of different algorithms, a simple heuristic is used to pick a (probably sub-optimal) plan quickly.

## Alternatives

Gemini suggests:

> For Rockchip ARM processors, the best alternatives to FFTW are ARM CMSIS-DSP (for heavily optimized core-specific assembly/NEON routines), PFFFT (extremely fast for small-to-medium sizes using SIMD), or KissFFT (simple, lightweight, and highly portable).

Key constraints are that Whisper uses a 400-point real FFT. That transform length is likely to determine which CMSIS-DSP API is viable.

### ARM CMSIS-DSP

[CMSIS-DSP](https://github.com/ARM-software/CMSIS-DSP) contains real and complex FFT implementations optimized for Cortex-M and Cortex-A processors. It is available under the Apache-2.0 license and has explicit NEON implementations, making it the alternative most closely aligned with the RK3588 target.

The conventional `arm_rfft_fast_init_f32` API only supports power-of-two real FFT lengths from 32 to 4096, so it cannot initialise Whisper's 400-point transform. The NEON implementation also provides `arm_rfft_fast_init_dynamic_f32`, which generates the required tables at runtime and supports mixed-radix lengths containing factors of three and five. This makes a 400-point real FFT viable on the AArch64 Linux and Android targets, provided CMSIS-DSP is built with its NEON implementation enabled.

The dynamic transform instance should be created once and reused for every STFT frame. Initialisation calculates trigonometric tables and allocates memory, so it should not happen inside the frame loop.

The NEON RFFT API requires separate input, output and temporary buffers. For a transform of length `N`, each of these buffers contains `N` floats. Its output is a packed representation in which the DC and Nyquist values share the first complex slot. The existing code instead expects `N / 2 + 1` complex values, so integration would need to expand the CMSIS-DSP output into the current representation:

```text
stft[0]     = {output[0], 0}
stft[k]     = {output[2 * k], output[2 * k + 1]} for 1 <= k < N / 2
stft[N / 2] = {output[1], 0}
```

CMSIS-DSP would be the most complicated alternative to integrate. Its source and license would need to be added under `thirdparty`, its CMake project configured for AArch64 and NEON, and the appropriate transform and common-table components linked into all three project executables. The FFT-specific code should also stop exposing `fftwf_complex` to magnitude calculation and transposition, using a library-independent complex type instead.

### PFFFT

[PFFFT](https://github.com/marton78/pffft) is a small, single-precision FFT library with explicit ARM NEON, x86 SIMD, PowerPC Altivec and WebAssembly SIMD implementations. It uses a permissive BSD-like license, provides CMake integration, and performs no allocation while executing a transform when the caller supplies a work buffer. A setup can therefore be created once and reused for every STFT frame.

The principal problem is the 400-point real FFT. With four-lane SIMD, PFFFT real transforms must be based on a minimum length of 32 and may then be multiplied by factors of two, three and five. A real transform of length 400 is not valid under those rules, so `pffft_new_setup(400, PFFFT_REAL)` would fail.

A 400-point complex transform is valid because `400 = 16 * 5 * 5`, where 16 is the minimum complex SIMD length. The project could set every imaginary input value to zero, perform a complex FFT, and retain bins 0 through 200. This would require roughly twice the transform work and input storage of a native real FFT, however, so PFFFT's apparent SIMD advantage would need to be demonstrated with a benchmark.

Another mathematically valid workaround would be an 800-point real transform with samples 400 through 799 set to zero, retaining only the even-numbered output bins. This also doubles the transform length and complicates output handling, so it offers no clear advantage over the complex approach.

PFFFT returns packed real-transform output and would require DC/Nyquist conversion similar to CMSIS-DSP. Its CMake configuration would also need to be checked on both targets to ensure it actually selects NEON; the runtime `pffft_simd_arch()` result can be used to verify this.

### PocketFFT

[PocketFFT](https://github.com/mreineck/pocketfft/tree/cpp) is a compact C++11 header-only library under the BSD-3-Clause license. It supports arbitrary transform lengths, with efficient real-transform codelets for factors of two, three, four and five. Since `400 = 2^4 * 5^2`, the Whisper transform uses its efficient mixed-radix path rather than a generic or Bluestein transform.

Its `r2c` interface directly produces `N / 2 + 1` complex values. For this project that is 201 complex bins, matching the current FFTW result without CMSIS-DSP- or PFFFT-style packed-output conversion. Integration would mainly involve vendoring the header, replacing `fftwf_complex` with `std::complex<float>` or another library-independent representation, and removing the FFTW include, archive and non-PIE linker workaround.

PocketFFT is the most portable and simplest option, but its documentation only claims CPU vectorisation when processing two-dimensional or higher-dimensional transforms. An individual 400-point one-dimensional transform is therefore unlikely to use a dedicated NEON kernel. Plan caching is also disabled by default, and repeatedly invoking the high-level interface for approximately 3,000 frames of 30-second audio may introduce avoidable setup and temporary-allocation overhead.

This could be mitigated by enabling its small plan cache or by preparing and transforming several STFT frames as a batch. Batching all frames would increase peak memory use, so moderate batches may be a better compromise.

## Recommendations

PocketFFT should be implemented first as a simple, portable baseline. It supports the required 400-point real FFT directly, produces the same 201-bin shape as FFTW, has the smallest build-system impact, and removes the GPL dependency without tying preprocessing to ARM.

CMSIS-DSP should then be evaluated as the optimized ARM64 backend. It has the best chance of taking advantage of RK3588 NEON hardware for the exact transform length, but requires dynamic mixed-radix initialisation, an extra work buffer, packed-output conversion and more involved CMake integration.

Both backends should initially remain selectable alongside FFTW. Differential tests should compare individual complex bins, squared magnitudes and the final normalized log-mel spectrogram for impulses, constant values, single-frequency signals, random input, short audio and final padded frames. This will detect differences in DC/Nyquist packing, imaginary sign, scaling and floating-point error. The complete log-mel calculation should then be benchmarked on an RK3588 rather than drawing conclusions from isolated FFT timings, because setup, allocation, windowing and output conversion are significant for such a small transform.

PFFFT should only be pursued if a benchmark shows that its 400-point complex workaround materially outperforms PocketFFT and approaches CMSIS-DSP. Its inability to perform the required 400-point real FFT directly makes it a less natural fit despite its good allocation behaviour and explicit NEON support.

Changing `kFftSize` from 400 to a library-friendly power of two is not recommended. It would alter the frequency-bin spacing and make the current 201-column Whisper mel filters incompatible with the FFT output.
