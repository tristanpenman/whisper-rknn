# Python

Follow the project's supported Python versions and existing formatter, linter, type checker, packaging, and CLI conventions.

## Projects and Packages

- Use a virtual environment or the project's container rather than the system Python.
- Run Python versions supported by the project and CI. Update that contract before using newer language features.
- Use absolute imports between top-level packages. Within a package, use explicit relative imports when they make the local relationship clear.
- Group imports as standard library, third-party packages, then local modules. Avoid wildcard imports.
- Keep imports at module scope unless an optional, expensive, circular, or platform-specific dependency must be delayed.
- Run package tools with `python -m package.module` where practical.

## Style and Names

- Let the project's formatter control layout. Without one, follow PEP 8 and nearby code with four-space indentation.
- Use `snake_case` for modules, functions, methods, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants.
- Include units in names such as `elapsed_ms` when they are not obvious.
- Use f-strings for interpolation.
- Do not use mutable default arguments. Use `None` and create the value inside the function.
- Distinguish `None` from other falsey values when zero, an empty string, or an empty collection is valid.

## Types and Documentation

- Add type annotations to public interfaces and non-obvious data structures. Avoid annotations that merely repeat an obvious local type.
- Use dataclasses when related values need named fields but not custom behaviour.
- Give modules and public APIs concise docstrings. Document shapes, axis order, units, side effects, and raised exceptions when the signature does not make them clear.

## Files and Resources

- Prefer `pathlib.Path` for path manipulation. Do not build paths with string concatenation.
- Open text files with an explicit encoding, normally UTF-8.
- Use context managers for files and other supported resources.
- Use `try` and `finally` when a native, model, graphics, or device resource requires explicit release.
- Create output directories explicitly with `mkdir(parents=True, exist_ok=True)` when the command is expected to do so.

## Errors and Diagnostics

- Raise specific exceptions from reusable code. Do not use `assert` for user input or runtime errors.
- When wrapping an exception, preserve its cause with `raise ... from exc`.
- Use `sys.exit()` or return an exit status only at an application boundary. Do not call the interactive `exit()` helper.
- Use logging for diagnostics and `print()` for the intended output of a small command-line tool.
- Use `time.perf_counter()` for elapsed-time measurements.

## Command-Line Applications

- Use the CLI framework already present in the project, such as `argparse` or Click.
- Put argument declaration and parsing in a separate function when the interface is more than trivial.
- Express types, defaults, required values, and fixed choices through the parser.
- Reject invalid argument combinations before loading models or large datasets.
- Put the executable path in a `main()` function and call it behind an `if __name__ == "__main__":` guard.
- Add a shebang only to modules intended to run directly. Python 3 source does not need a UTF-8 encoding declaration.

## Numerical and Scientific Code

- Make array shapes, axis order, dtype, coordinate system, and units explicit at interfaces.
- Check dimensions and element counts before reshaping or calling native and device APIs.
- Choose dtypes deliberately at model, image, file-format, and native-library boundaries. Do not rely on platform-default widths.
- Use NumPy operations when they are clearer, but avoid unnecessary copies and conversions in performance-sensitive paths.
- Put thresholds, padding rules, sentinel values, and transpose operations in named functions or constants.
- Convert NumPy scalar values to built-in Python types when required by serializers or foreign APIs.
- Separate tests that require accelerators, displays, native runtimes, large datasets, or network access from ordinary unit tests.

## Example

```python
from pathlib import Path

import numpy as np


def load_volume(path: Path, expected_shape: tuple[int, int, int]) -> np.ndarray:
    """Load a float32 volume and validate its voxel dimensions."""
    try:
        volume = np.load(path).astype(np.float32, copy=False)
    except OSError as exc:
        raise RuntimeError(f"Unable to load volume: {path}") from exc

    if volume.shape != expected_shape:
        raise ValueError(
            f"Volume {path} has shape {volume.shape}; expected {expected_shape}"
        )

    return volume
```
