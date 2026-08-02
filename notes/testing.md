# Testing

This document proposes tests that focus on `cpp/src/audio_utils.cc` and `cpp/src/process.cc`. Some tests already exist for channel conversion. These tests will build upon that, using the same patterns where appropriate.

Temporary fixtures should be created in a per-test temporary directory where practical. Audio comparisons should use a small floating-point tolerance, and float WAV files should be used to avoid integer quantization differences.

## Audio file I/O

### Save and read a mono WAV

Generate a short, deterministic float signal, save it with `saveAudio()`, and load it with `readAudio()`. Verify that both functions return zero, that the frame count, sample rate, and channel count are preserved, and that every decoded sample matches the input within a small tolerance.

### Save and read a stereo WAV

Save interleaved left and right samples and read them back. Verify the metadata, sample count, sample values, and interleaved channel layout.

### Read a nonexistent file

Pass a unique path that does not exist to `readAudio()` and expect `-1`. This exercises the `sf_open()` failure path.

### Save to an invalid location

Pass a path beneath a nonexistent directory to `saveAudio()` and expect `-1`.

## Resampling

### Upsample with linear interpolation

Resample `[0, 10]` from 2 Hz to 4 Hz. Verify that the result has four frames and values approximately equal to `[0, 5, 10, 10]`.

### Downsample

Resample a known ramp from 4 Hz to 2 Hz. Verify the resulting frame count and the expected selected or interpolated values.

### Use a non-integral output ratio

Choose sample rates whose ratio produces a fractional output length and verify that the frame count follows the implementation's `std::round()` behavior.

### Keep data unchanged at the same rate

Resample with equal original and desired sample rates and verify that the values and frame count remain unchanged.

### Update sample-rate metadata

Verify that a successful resample sets `audio.sampleRate` to the desired sample rate. The current implementation only updates `numFrames`, so this test would expose a likely metadata defect.

### Reject empty input and invalid rates

Once validation is added, verify that empty input, a zero or negative original rate, and a zero or negative desired rate return `-1` without modifying the buffer. The current implementation can index an empty vector or divide by zero.

## String helpers and Base64 decoding

### Replace substrings

Test a single occurrence, repeated non-overlapping occurrences, replacements that are shorter and longer than the search string, deletion using an empty replacement, a missing search string, and an empty search string. The empty search string should leave the value unchanged.

### Decode standard Base64 vectors

Verify `Zg==` decodes to `f`, `Zm8=` to `fo`, and `Zm9v` to `foo`. Add a multi-block string and encoded binary data containing a null byte to ensure the result is treated as a length-aware `std::string`.

### Reject malformed Base64

Test an empty value, invalid characters, incomplete groups, misplaced padding, and a value beginning with `=`. The current implementation exits the process for some errors and returns a single space for leading padding. Death tests could document the current behavior, but changing the API to report a nonfatal error would make it safer and easier to test.

## Mel-filter and vocabulary parsing

### Read mel filters

Write a temporary text fixture containing known values, including negative values and scientific notation, and verify that `readMelFilters()` parses them correctly.

### Respect the mel-filter limit

Provide more values than `maxLines`, place sentinel values beyond the permitted destination range, and verify that only the requested number of elements are written.

### Handle missing and malformed mel-filter files

Verify that a missing file returns `-1`. A short file or a malformed value currently results in success without reporting how many values were parsed; define the desired behavior as failure unless all required values are read, then test it after hardening the parser.

### Read vocabulary entries

Create a small fixture such as `0 hello` and `7 world`, each on its own line, and verify the indices and tokens. Add a test that defines whether trailing newlines are retained or stripped; stripping them is likely the more useful contract.

### Handle malformed vocabulary entries

Test a missing file and a line without the separating space. The missing file should return `-1`; the malformed line should produce a clean error after validation is added, rather than performing pointer arithmetic on a null result from `strchr()`.

### Enforce vocabulary capacity

`readVocab()` currently receives no destination capacity and can overrun the supplied array. Revise the API to accept a maximum entry count, then verify that extra lines are rejected or ignored according to the chosen contract.

## Argmax

### Find maxima at different positions

Allocate the full expected logits buffer and place a unique maximum at the beginning, middle, and end of the final token row. Verify that the corresponding vocabulary index is returned.

### Ignore earlier token rows

Place a value in an earlier token row that is larger than every value in the final row. Verify that only the final row is searched.

### Handle negative values and ties

Verify that the least-negative value wins when all values are negative. For equal maxima, verify that the first matching index is returned, consistent with the strict `>` comparison.

## Audio preprocessing

### Produce the expected shape for short audio

Supply a short deterministic signal, correctly sized mel filters, and an output buffer sized to `kNumMels * kEncoderInputSize`. Verify that preprocessing completes and that every computed value is finite.

### Pad short spectrograms by row

Prefill the output with a sentinel, preprocess audio shorter than 20 seconds, and verify that each mel row's computed prefix is replaced while the remaining columns retain the selected padding value. If zero is the intended padding, initialize the output to zero and assert that explicitly.

### Process maximum-length audio

Use exactly `kMaxAudioLength` samples and verify that the complete `80 * 2000` output is finite.

### Trim overlong audio

Compare the output for a maximum-length signal with the output for the same signal followed by extra samples. The results should match within an FFT-appropriate tolerance.

### Exercise known signals

Preprocess silence, an impulse, and a sine wave. Verify that none produces NaN or infinity, that values remain in the expected normalized range, that silence is uniform or nearly uniform, and that the sine-wave output differs measurably from silence.

### Compare against a reference mel spectrogram

Check in a deterministic waveform and reference output generated by a trusted Whisper implementation. Compare either the full tensor or a representative set of bins with documented tolerances. This provides end-to-end coverage of reflection padding, the Hann window, STFT layout, power calculation, mel-filter multiplication, log clamping, normalization, and output padding.

### Compare scalar and NEON results

Compare preprocessing output from the native NEON path with a scalar reference for the same fixture. This is particularly valuable because the NEON matrix multiplication has specialized sparse-range and vector memory-access logic that basic shape and finiteness assertions will not validate.
