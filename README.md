# ofxStableDiffusion

`ofxStableDiffusion` is an openFrameworks addon that wraps
[`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) for
text-to-image, image-to-image, image-to-video, and upscaling workflows.

Current addon version: `1.0.0`

## Requirements

- **openFrameworks**: 0.11.0 or later (tested with 0.12.0)
- **C++ Standard**: C++17 or later
- **Platform Support**: Windows (x64), Linux (x64), macOS (experimental Metal support)

The addon is now structured more like a production addon:

- typed request/config/result objects
- background-thread generation
- owned image/video outputs
- explicit native-library staging
- wrapper-level integration seams for `ofxGgml` interaction
- lightweight unit tests for the new video helper layer

## Highlights

- Text-to-image and image-to-image generation
- Native image modes: `TextToImage`, `ImageToImage`, and `Inpainting`
- Best-of-N image reranking through a callback seam that can be driven by `ofxGgml` CLIP scoring
- Image-to-video generation with `Standard`, `Loop`, `PingPong`, and `Boomerang` presentation modes
- ESRGAN upscaling support
- Progress callbacks for diffusion steps
- `getCapabilities()` plus capability/model-family helpers for UI gating and backend-aware workflows
- parameter-tuning helpers for image/video defaults, ranges, and clamping by model family
- Optional `ofxStableDiffusionHoloscanBridge` scaffold for live `frame -> conditioning -> diffusion -> preview` pipelines, with a native Holoscan runtime path on Linux and a clean fallback path when Holoscan is not installed or the platform is not supported yet
- `ofxStableDiffusionVideoWorkflowHelpers.h` for reusable video-generation presets, request validation, and richer render-manifest export on top of the existing request/result layer
- Legacy entry points still available for wrapper-level migration
- Standalone native runtime management instead of sharing `ggml` binaries across addons

## Repo Layout

- `src/`
  Addon wrapper, typed request/result model, and background-thread integration
- `src/core/`
  Enums and owned result types
- `src/video/`
  Video clip behavior, pure helper utilities, and reusable workflow helpers for:
  - `FastPreview`, `LowVram`, `Balanced`, `Quality`, and `BatchStoryboard` presets
  - preflight request validation before a longer render starts
  - richer JSON render manifests that capture prompt, dimensions, fps, seed, LoRAs, and clip summary data
- `libs/stable-diffusion/`
  Bundled header/libs and vendoring location for upstream native source
- `scripts/`
  Native rebuild scripts
- `tests/`
  Lightweight CMake-based unit tests
- `docs/`
  Architecture and native-build notes

## API Shape

The addon now supports two layers of use:

### Modern typed wrapper API

```cpp
#include "ofxStableDiffusion.h"

ofxStableDiffusion sd;

ofxStableDiffusionContextSettings context;
context.modelPath = "data/models/sd/sd_turbo.safetensors";
context.nThreads = 8;
context.weightType = SD_TYPE_F16;
sd.configureContext(context);

ofxStableDiffusionImageRequest request;
request.prompt = "cinematic portrait, rim lighting";
request.width = 512;
request.height = 512;
request.sampleSteps = 20;
sd.generate(request);

// Optional: apply a stack of LoRA/LoCon adapters per request
ofxStableDiffusionLora loraA;
loraA.path = "data/loras/edge.safetensors";
loraA.strength = 0.7f;
request.loras = {loraA};
sd.generate(request);

// Update the active LoRA stack globally
sd.setLoras({loraA});

// Hot-reload textual-inversion embeddings (reloads the context)
sd.reloadEmbeddings("data/embeddings");

// List currently discoverable embeddings (name, absolute path)
const auto embeddings = sd.listEmbeddings();

// Inspect what the currently configured model/runtime can actually do
const auto capabilities = sd.getCapabilities();

// Save generated video frames plus a JSON sidecar with per-frame prompts/seeds
sd.saveVideoFramesWithMetadata("output/storyboard", "shot");
```

The parameter-tuning helpers can also be used outside the wrapper instance when
you want model-family-aware defaults before building a request:

```cpp
const auto imageProfile =
    ofxStableDiffusionParameterTuningHelpers::resolveImageProfile(
        context,
        ofxStableDiffusionImageMode::ImageToImage);

request.cfgScale = imageProfile.defaultCfgScale;
request.sampleSteps = imageProfile.defaultSampleSteps;
request.strength = imageProfile.defaultStrength;
```

### Error Handling

The addon provides advanced error handling with error codes, messages, suggestions, and history tracking:

```cpp
ofxStableDiffusionImageRequest request;
request.width = 513;  // Invalid: not a multiple of 64
request.height = 512;
request.batchCount = 20;  // Invalid: exceeds maximum of 16

sd.generate(request);

// Check for errors programmatically
if (sd.getLastErrorCode() != ofxStableDiffusionErrorCode::None) {
    // Get detailed error information
    const auto& errorInfo = sd.getLastErrorInfo();
    ofLogError() << "Error: " << errorInfo.message;
    ofLogError() << "Suggestion: " << errorInfo.suggestion;
    ofLogError() << "Error code: " << ofxStableDiffusionErrorCodeLabel(errorInfo.code);
}

// Or use the simple string-based error (backward compatible)
if (!sd.getLastError().empty()) {
    ofLogError() << "Error: " << sd.getLastError();
}

// Access error history for debugging
const auto& history = sd.getErrorHistory();
for (const auto& error : history) {
    ofLogNotice() << "[" << error.timestampMicros << "] "
                  << error.message << " -> " << error.suggestion;
}

// Clear error history
sd.clearErrorHistory();
```

**Available Error Codes:**
- `None` - No error
- `ModelNotFound` - Model file not found
- `ModelCorrupted` - Model file corrupted
- `ModelLoadFailed` - Model loading failed
- `OutOfMemory` - Insufficient memory
- `InvalidDimensions` - Invalid width/height
- `InvalidBatchCount` - Invalid batch count
- `InvalidFrameCount` - Invalid video frame count
- `MissingInputImage` - Input image required but not provided
- `GenerationFailed` - Generation process failed
- `ThreadBusy` - Another task is running
- `UpscaleFailed` - Upscaling failed
- `Unknown` - Unknown error

Each error code automatically provides an actionable suggestion to help resolve the issue.

### Legacy compatibility API

The older `newSdCtx`, `txt2img`, `img2img`, and `img2vid` entry points are still
available so existing call sites, including addon-to-addon bridges, do not have
to migrate immediately.

## Video Behavior

Video generation returns owned frames through `ofxStableDiffusionVideoClip`.

Animated requests can also drive per-frame prompt interpolation, parameter
animation, and seed sequencing through `ofxStableDiffusionVideoRequest::animationSettings`.
Each returned frame now carries the prompt / negative prompt / CFG / strength /
seed values that were actually used, and clips can export both PNG sequences and
JSON metadata.

Useful video-export helpers:

- `ofxStableDiffusionVideoClip::saveMetadataJson(...)`
- `ofxStableDiffusionVideoClip::saveFrameSequenceWithMetadata(...)`
- `ofxStableDiffusion::saveVideoMetadata(...)`
- `ofxStableDiffusion::saveVideoFramesWithMetadata(...)`

Supported playback/presentation modes:

- `Standard`
  Return generated frames as-is
- `Loop`
  Repeat the first frame at the end for easier closed-loop playback
- `PingPong`
  Play forward, then back through the interior frames
- `Boomerang`
  Play forward, then fully reverse including the endpoints

This is especially useful when a UI layer or another addon wants a more natural
preview clip without re-asking the native runtime for more frames.

### Video Performance Recommendations

- **Use fast models for previews**: Prefer LCM/Turbo-style checkpoints with 4–8 steps for quick iteration; only re-run hero frames at higher quality.
- **Quantize when possible**: F16 or Q8/Q5 levels often cut VRAM use by 50–75% and speed up inference, enabling higher resolutions or longer clips without swapping.
- **Keep the model warm and resident**: Reuse the same loaded context via the model manager; avoid swapping checkpoints mid-run and consider a throwaway warmup frame to eliminate first-run latency.
- **Minimize unique frames**: Generate the smallest necessary source frame count, then stretch duration with `PingPong`, `Boomerang`, or `Loop` playback instead of regenerating.
- **Right-size resolution, fps, and steps**: Lower resolution and fps where acceptable; clamp `sampleSteps` to ~15–25 for finals (lower for previews). Per-frame time scales directly with unique frames.
- **Trim preview I/O**: Skip metadata/JSON exports on preview passes; save sidecar files only on final renders to avoid extra disk churn.

## Coding Conventions (openFrameworks-aligned)

- Use tabs (as in existing headers) with K&R braces; lowerCamelCase for functions/members and PascalCase for types, keeping the `ofxStableDiffusion*` prefix for public types.
- Include order: `ofMain.h`, then addon headers, then STL/system headers.
- Document public API with `///` Doxygen-style comments; avoid mixing `//` for header docs.
- Avoid exceptions; return `bool` or error codes and populate `ofxStableDiffusionError` for failures.
- Keep small helpers inline in headers; move heavier logic to `.cpp` files to limit inline bloat.
- Prefer RAII and STL containers over raw `new`/`delete`.
- Keep generation calls on the main thread; callbacks should stay lightweight for the OF event loop.
- Favor openFrameworks core types at the API edge: `ofPixels`/`ofImage` for images, `ofJson` for metadata, `ofVec*`/`ofFloatColor`/`ofRectangle` where geometry or color is needed. Convert to STL or native structs internally only when necessary for performance or binding.

## Image Modes

The typed image request layer exposes the native image modes:

- `TextToImage`
- `ImageToImage`
- `Inpainting`

## CLIP-Rerank Integration

`ofxStableDiffusion` now exposes a wrapper-level image ranking callback for
Best-of-N workflows.

That is the intended integration point for `ofxGgml` CLIP scoring:

- generate a batch in `ofxStableDiffusion`
- score the outputs with `ofxGgmlClipInference`
- rerank or collapse to the best image without sharing native runtimes

This keeps diffusion and CLIP loosely coupled while still enabling a strong
cross-addon workflow.

## `ofxGgml` Integration Guidance

This addon can be integrated with `ofxGgml` in two ways:

### 1. Addon-Level Integration (Recommended)

- keep `ofxStableDiffusion` standalone at the native-runtime layer
- integrate `ofxGgml` with it through the addon API
- do not share the low-level `ggml` binary directly across addons

Why:

- upstream `stable-diffusion.cpp` may require a different `ggml` revision
- backend flags and ABI expectations can diverge
- wrapper-level integration is more stable than native binary coupling

There is already an addon-level bridge seam on the `ofxGgml` side through
`ofxGgmlStableDiffusionAdapters.h`. This addon also exposes a small bridge helper header:

- `src/bridges/ofxStableDiffusionGgmlBridge.h`

### 2. System GGML Integration (Optional)

For advanced use cases where you want a single GGML binary shared across addons, you can build `ofxStableDiffusion` to consume GGML from `ofxGgml`:

```bash
# Linux/macOS
./scripts/build-stable-diffusion.sh --use-system-ggml --ofxggml-path ../ofxGgml

# Windows
.\scripts\build-stable-diffusion.ps1 -UseSystemGgml -OfxGgmlPath ..\ofxGgml
```

This enables `stable-diffusion.cpp`'s built-in system GGML support via `-DSD_USE_SYSTEM_GGML=ON`. The build scripts will:
- Detect `ofxGgml` at the specified path (defaults to `../../ofxGgml`)
- Use ofxGgml's GGML headers and libraries instead of bundling GGML
- Validate that ofxGgml is built before proceeding

**Important considerations:**
- ofxGgml must be built first with matching backend flags (CPU/CUDA/Vulkan)
- GGML version compatibility between stable-diffusion.cpp and ofxGgml must be maintained
- This mode is opt-in; the default remains standalone for stability

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/OFXGGML_BRIDGE.md](docs/OFXGGML_BRIDGE.md)

## Native Runtime

The addon stages native artifacts into addon-local paths:

- `libs/stable-diffusion/include`
- `libs/stable-diffusion/lib/vs`
- `libs/stable-diffusion/lib/Linux64`
- `libs/ggml/include`
- `libs/ggml/lib/vs`
- `libs/variants/<backend>/...`

Rebuild helpers:

- `scripts/build-stable-diffusion.ps1`
- `scripts/build-stable-diffusion.bat`
- `scripts/build-stable-diffusion.sh`
- `scripts/download-stable-diffusion-release.ps1`
- `scripts/setup_addon.ps1`
- `scripts/setup_windows.bat`

More detail: [docs/NATIVE_BUILD.md](docs/NATIVE_BUILD.md)

Backend flags now follow the same style as `ofxGgml`, but the build selects one
backend per build:

- `--cpu`, `--cpu-only` / `-CpuOnly`
  Force CPU-only native builds (default)
- `--gpu`, `--cuda` / `-Cuda`
  Enable CUDA explicitly
- `--vulkan` / `-Vulkan`
  Enable Vulkan explicitly
- `--metal` / `-Metal`
  Enable Metal explicitly where supported
- `--all` / `-All`
  Build every available backend variant and leave the canonical runtime on the
  best one in this priority order: `cuda`, then `vulkan`, then `cpu-only`

Each backend build is also snapshotted under `libs/variants/<backend>`, so you
can switch the canonical addon runtime later without rebuilding everything.

Variant selector helpers:

- `scripts/select-stable-diffusion-backend.ps1 -Backend cuda`
- `scripts/setup_windows.bat --skip-native --select-backend cuda`

For Windows, `scripts/setup_windows.bat` and `scripts/setup_addon.ps1` now
always refresh the vendored source from the latest upstream release-tag source
snapshot, then build the native runtime locally.

- `--source-release-tag TAG`
  Override the upstream release tag used for the vendored source snapshot

When an upstream Windows zip does not include `stable-diffusion.lib`, the addon
setup script synthesizes the Visual Studio import library from the downloaded
DLL exports automatically.

## Current Native Source Status

The repo now includes a vendored upstream `stable-diffusion.cpp` source snapshot
under `libs/stable-diffusion/source`, pinned to:

- upstream repo: `https://github.com/leejet/stable-diffusion.cpp`
- upstream release tag: `master-585-44cca3d`
- upstream commit: `44cca3d`
- vendored on: `2026-04-21`

The optional Windows prebuilt-runtime flow is currently pinned to the upstream
GitHub release tag `master-585-44cca3d`, which was the latest upstream release
published on `2026-04-19`. Override it with `--release-tag` if you want a
different upstream runtime. Source: [stable-diffusion.cpp releases](https://github.com/leejet/stable-diffusion.cpp/releases)

The addon now includes the upstream header directly through
`libs/stable-diffusion/include/stable-diffusion.h`, without re-exporting the
older enum aliases. Addon code should use the current upstream names such as
`EULER_A_SAMPLE_METHOD`, `DPMPP2Mv2_SAMPLE_METHOD`, and `SCHEDULER_COUNT`.

## Testing

The test suite focuses on wrapper-level logic that should stay stable even when
the native runtime changes.

Run:

```bash
cmake -S tests -B tests/build
cmake --build tests/build --config Release
ctest --test-dir tests/build -C Release --output-on-failure
```

You can also use:

- `scripts/run-tests.ps1`
- `scripts/run-tests.sh`

Test notes: [tests/README.md](tests/README.md)

## Example

The example project lives in `ofxStableDiffusionExample/` and now exposes:

- progress/error status
- busy-state gating
- video-mode selection
- a small `Holoscan Bridge` section for the new bridge MVP, including prompt handoff, loaded-image submission, and inline bridge preview
  - the native Holoscan runtime path is Linux-only for now; Windows and other platforms stay on the addon fallback lane until that runtime is validated there
- frame export plus JSON metadata for generated clips
- optional end-frame morphing
- prompt morph / seed-sequence animation controls

## Troubleshooting

### Model Loading Issues

**Problem**: Model fails to load or crashes during loading
- **Solution**: Verify the model file exists and is not corrupted
- **Solution**: Check that the model format is compatible (`.safetensors`, `.ckpt`, or `.gguf`)
- **Solution**: Ensure sufficient RAM/VRAM (models typically require 4-8GB+)

### Out of Memory Errors

**Problem**: Generation fails with OOM error
- **Solution**: Reduce batch count (try `batchCount = 1`)
- **Solution**: Reduce image dimensions (try 512x512 instead of higher)
- **Solution**: Enable VAE tiling with `vaeTiling = true`
- **Solution**: Use quantized models (Q4_0, Q5_0, etc.) for lower memory usage

### Slow Generation

**Problem**: Image generation takes too long
- **Solution**: Increase thread count with `nThreads = -1` (auto-detect) or set manually
- **Solution**: Reduce sample steps (try 10-20 steps for faster results)
- **Solution**: Use smaller models like SD-Turbo for faster iteration
- **Solution**: Enable GPU backend if available (`--cuda` or `--vulkan` during build)

### Thread Safety

**Note**: The addon is designed for single-threaded use from the main thread. Do not call `generate()` or `generateVideo()` from multiple threads simultaneously. The background thread is managed internally.

### Invalid Input Dimensions

**Problem**: Generation fails with dimension errors
- **Solution**: Ensure width and height are positive multiples of 64 (e.g., 512, 768, 1024)
- **Solution**: Most SD models work best with dimensions between 256-1024

## Thread Safety Notes

- The addon manages its own background thread for generation
- All public API methods should be called from the main thread
- Do not call `generate()` or `generateVideo()` while a previous generation is running
- Use `isGenerating()` to check if generation is in progress
- Progress callbacks are fired from the background thread - use thread-safe operations in callbacks

## Documentation

Comprehensive documentation is available:

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation with classes, methods, and types
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Upgrade from legacy API to modern request-based API
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Architecture Notes](docs/ARCHITECTURE.md)** - Design decisions and integration patterns
- **[Native Build Guide](docs/NATIVE_BUILD.md)** - Building stable-diffusion.cpp from source

### Code Examples

The `examples/` directory contains working sample applications:

- **[basic_generation](examples/basic_generation/)** - Simple text-to-image generation
- **[cancellation_example](examples/cancellation_example/)** - Cancelling long-running operations

### Quick Links

- [Model Family Capabilities](docs/API_REFERENCE.md#model-families-and-capabilities)
- [Error Handling](docs/API_REFERENCE.md#error-handling)
- [Performance Tips](docs/API_REFERENCE.md#performance-optimization)
- [Platform Support](docs/API_REFERENCE.md#platform-support)
- [Cancellation API](docs/API_REFERENCE.md#cancellation)

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
