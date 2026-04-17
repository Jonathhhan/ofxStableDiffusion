# ofxStableDiffusion

`ofxStableDiffusion` is an openFrameworks addon that wraps
[`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) for
text-to-image, image-to-image, image-to-video, and upscaling workflows.

Current addon version: `1.0.0`

The addon is now structured more like a production addon:

- typed request/config/result objects
- background-thread generation
- owned image/video outputs
- explicit native-library staging
- wrapper-level integration seams for `ofxGgml` interaction
- lightweight unit tests for the new video helper layer

## Highlights

- Text-to-image and image-to-image generation
- Expanded image modes: `TextToImage`, `ImageToImage`, `InstructImage`, `Variation`, and `Restyle`
- Best-of-N image reranking through a callback seam that can be driven by `ofxGgml` CLIP scoring
- Image-to-video generation with `Standard`, `Loop`, `PingPong`, and `Boomerang` presentation modes
- ESRGAN upscaling support
- Progress callbacks for diffusion steps
- Legacy entry points still available for wrapper-level migration
- Standalone native runtime management instead of sharing `ggml` binaries across addons

## Repo Layout

- `src/`
  Addon wrapper, typed request/result model, and background-thread integration
- `src/core/`
  Enums and owned result types
- `src/video/`
  Video clip behavior and pure helper utilities
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
```

### Legacy compatibility API

The older `newSdCtx`, `txt2img`, `img2img`, and `img2vid` entry points are still
available so existing call sites, including addon-to-addon bridges, do not have
to migrate immediately.

## Video Behavior

Video generation returns owned frames through `ofxStableDiffusionVideoClip`.

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

## Image Modes

The typed image request layer now exposes addon-level image modes:

- `TextToImage`
- `ImageToImage`
- `InstructImage`
- `Variation`
- `Restyle`

`InstructImage` is implemented as a first-class wrapper mode in the addon layer,
with the wrapper mapping legacy-style request data onto the current upstream
parameter-struct API while still keeping the editing-oriented API clear for
callers.

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

Recommended architecture:

- keep `ofxStableDiffusion` standalone at the native-runtime layer
- integrate `ofxGgml` with it through the addon API
- do not share the low-level `ggml` binary directly across addons

Why:

- upstream `stable-diffusion.cpp` may require a different `ggml` revision
- backend flags and ABI expectations can diverge
- wrapper-level integration is more stable than native binary coupling

More detail: [docs/ARCHITECTURE.md](/C:/Users/Jonathan%20Frank/Desktop/of_v20260406_vs_64_release/addons/ofxStableDiffusion/docs/ARCHITECTURE.md)

## Native Runtime

The addon stages native artifacts into addon-local paths:

- `libs/stable-diffusion/include`
- `libs/stable-diffusion/lib/vs`
- `libs/stable-diffusion/lib/Linux64`

Rebuild helpers:

- `scripts/build-stable-diffusion.ps1`
- `scripts/build-stable-diffusion.bat`
- `scripts/build-stable-diffusion.sh`
- `scripts/download-stable-diffusion-release.ps1`
- `scripts/setup_addon.ps1`
- `scripts/setup_windows.bat`

More detail: [docs/NATIVE_BUILD.md](/C:/Users/Jonathan%20Frank/Desktop/of_v20260406_vs_64_release/addons/ofxStableDiffusion/docs/NATIVE_BUILD.md)

Backend flags now follow the same style as `ofxGgml`:

- `--auto` / `-Auto`
  Auto-detect supported GPU backends (default)
- `--cpu`, `--cpu-only` / `-CpuOnly`
  Force CPU-only native builds
- `--gpu`, `--cuda` / `-Cuda`
  Enable CUDA explicitly
- `--vulkan` / `-Vulkan`
  Enable Vulkan explicitly
- `--metal` / `-Metal`
  Enable Metal explicitly where supported

For Windows, `scripts/setup_windows.bat` and `scripts/setup_addon.ps1` also
support an optional prebuilt-runtime path:

- `--use-release`
  Stage a pinned upstream Windows release instead of compiling from source
- `--release-tag TAG`
  Override the upstream GitHub release tag used with `--use-release`
- `--release-variant auto|cpu|noavx|avx|avx2|avx512|cuda12`
  Choose the prebuilt runtime flavor

`--auto` remains the default setup behavior. The prebuilt-release path is
explicit opt-in and is currently intended for Windows CPU/CUDA staging. Windows
Vulkan still falls back to source builds.

When an upstream Windows zip does not include `stable-diffusion.lib`, the addon
setup script synthesizes the Visual Studio import library from the downloaded
DLL exports automatically.

## Current Native Source Status

The repo now includes a vendored upstream `stable-diffusion.cpp` source snapshot
under `libs/stable-diffusion/source`, pinned to:

- upstream repo: `https://github.com/leejet/stable-diffusion.cpp`
- upstream commit: `a564fdf642780d1df123f1c413b19961375b8346`
- vendored on: `2026-04-17`

The optional Windows prebuilt-runtime flow is currently pinned to the upstream
GitHub release tag `master-572-1b4e9be`, which was the latest upstream release
published on `2026-04-16`. Override it with `--release-tag` if you want a
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

Test notes: [tests/README.md](/C:/Users/Jonathan%20Frank/Desktop/of_v20260406_vs_64_release/addons/ofxStableDiffusion/tests/README.md)

## Example

The example project lives in `ofxStableDiffusionExample/` and now exposes:

- progress/error status
- busy-state gating
- video-mode selection
- frame export for generated clips

## Changelog

See [CHANGELOG.md](/C:/Users/Jonathan%20Frank/Desktop/of_v20260406_vs_64_release/addons/ofxStableDiffusion/CHANGELOG.md).
