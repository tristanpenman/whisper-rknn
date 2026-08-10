# Bugs

Obvious Bugs / Code Errors

## High severity

### Audio shorter than one hop can cause undefined behavior

Audio shorter than one hop, or 160 samples, produces zero Mel columns. `clampAndNormalize()` then reads `melSpectrogram[0]` from an empty vector.

The preprocessing API should reject empty input and either support short input by padding it safely or require a documented minimum length.

### Mel normalization mixes linear and logarithmic values

`clampAndNormalize()` in `cpp/src/process.cc` initializes `maxValue` from the first raw Mel value. It then transforms the spectrogram with `log10()` and compares the transformed values with that untransformed initial value.

This can select the wrong dynamic-range threshold, especially for silence and quiet recordings. `maxValue` should be initialized after the logarithmic transformation or to negative infinity.

## Medium severity

### The Hann window differs from the Whisper reference

`makeHannWindow()` in `cpp/src/process.cc` uses `length - 1` as the cosine denominator, producing a symmetric Hann window. The Python implementation calls `torch.hann_window()` with its default `periodic=True`, which uses `length` as the denominator.

This makes every C++ STFT frame differ slightly from the preprocessing used to train and export Whisper. An end-to-end C++-versus-Python Mel fixture should cover this issue.

### The command-line program always reports success

The cleanup block in `cpp/src/main.cc` overwrites the result of audio loading, model initialization, or inference with the model-release results. `main()` then unconditionally returns zero.

Missing audio, invalid models, and inference failures therefore appear successful to calling scripts. The original failure status should be preserved across cleanup and returned to the caller.

### Vocabulary parsing can crash or overflow its destination

`readVocab()` in `cpp/src/process.cc` assumes that every line contains a space and performs pointer arithmetic on the result of `strchr()` without checking for null. A malformed line can therefore crash.

The function also receives no destination capacity and writes one entry for every input line. An oversized vocabulary can write beyond the supplied array. The bundled vocabulary files are currently well-formed and contain the expected number of entries, but the parser itself is unsafe.

### C++ decoder truncation is reported as success

The C++ decoder stops when it reaches the configured maximum token count, but still returns success and may return partial text. It should expose failure or truncation explicitly.

## Low severity

### Mel-filter parsing accepts incomplete input

`readMelFilters()` in `cpp/src/process.cc` returns success after any number of successfully parsed values, including zero. A truncated or malformed file can therefore leave part of the filter array unchanged without reporting an error.

The bundled filter file currently contains the expected 16,080 values. The function should nevertheless require the requested number of values and reject unexpected trailing data if exact input is required.

### Unsupported Python model paths cause an indirect exception

`init_model()` in `python/whisper_rknn/whisper.py` only assigns `model` for `.rknn` and `.onnx` paths. Any other extension reaches `return model` with no local value assigned, raising `UnboundLocalError` instead of a useful validation error.
