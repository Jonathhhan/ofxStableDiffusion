# ofxStableDiffusion Tests

The test suite is intentionally lightweight and focused on wrapper-level logic
that should remain stable even when the bundled native `stable-diffusion.cpp`
 snapshot changes.

## Coverage

### Core Functionality
- **Video helpers**: Timing calculations, frame sequencing for all video modes (`Standard`, `Loop`, `PingPong`, `Boomerang`)
- **Image helpers**: Mode names, input image requirements, task routing, default strength and CFG scale parameters
- **Ranking helpers**: Image selection mode names, score ranking algorithm, metadata handling

### Test Organization

#### Basic Tests (Original)
- `test_image_helpers.cpp` - Basic image mode helper functions
- `test_video_helpers.cpp` - Basic video sequencing and timing
- `test_ranking_helpers.cpp` - Basic ranking functionality

#### Extended Tests (New)
- `test_image_helpers_extended.cpp` - Comprehensive coverage of all image modes and their default parameters
- `test_video_helpers_extended.cpp` - Edge cases for video timing, sequencing, and frame lookups
- `test_ranking_helpers_extended.cpp` - Advanced ranking scenarios including edge cases

### Edge Cases Covered
- **Empty collections**: Zero frames, zero scores
- **Single-element collections**: Single frame sequences, single scores
- **Boundary values**: Negative frame counts, zero/negative FPS, negative time values
- **Extreme values**: Very large time values, maximum frame indices
- **Identical values**: Same scores (verifies stable sort behavior)
- **Invalid states**: Invalid scores, mixed valid/invalid scores
- **All enum values**: Complete coverage of all mode enumerations

### Test Statistics
- **6 test executables** with comprehensive coverage
- **100+ individual test assertions**
- All tests are pure C++ with no openFrameworks runtime dependency

## Run

```bash
cmake -S tests -B tests/build
cmake --build tests/build --config Release
ctest --test-dir tests/build -C Release --output-on-failure
```

On Windows with Visual Studio generators, `ctest -C Release` is recommended.
The current tests are pure C++ and do not require the native diffusion library
or an openFrameworks runtime.
