# ofxStableDiffusion Tests

The test suite is intentionally lightweight and focused on wrapper-level logic
that should remain stable even when the bundled native `stable-diffusion.cpp`
 snapshot changes.

## Coverage

- Video timing helpers
- Video mode sequencing (`Standard`, `Loop`, `PingPong`, `Boomerang`)
- Label mapping for the public video mode enum
- Image mode labels, task routing, and preset defaults
- Ranking helper labels and best-score ordering

## Run

```bash
cmake -S tests -B tests/build
cmake --build tests/build --config Release
ctest --test-dir tests/build -C Release --output-on-failure
```

On Windows with Visual Studio generators, `ctest -C Release` is recommended.
The current tests are pure C++ and do not require the native diffusion library
or an openFrameworks runtime.
