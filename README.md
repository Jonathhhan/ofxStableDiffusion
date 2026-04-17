# ofxStableDiffusion

`ofxStableDiffusion` is an openFrameworks addon that wraps
[`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) for
text-to-image, image-to-image, image-to-video, and upscaling workflows.

The addon is now structured more like a production addon:

- typed request/config/result objects
- background-thread generation
- owned image/video outputs
- explicit native-library staging
- wrapper-level compatibility for existing `ofxGgml` interaction
- lightweight unit tests for the new video helper layer

## Highlights

- Text-to-image and image-to-image generation
- Expanded image modes: `TextToImage`, `ImageToImage`, `InstructImage`, `Variation`, and `Restyle`
- Best-of-N image reranking through a callback seam that can be driven by `ofxGgml` CLIP scoring
- Image-to-video generation with `Standard`, `Loop`, `PingPong`, and `Boomerang` presentation modes
- ESRGAN upscaling support
- Progress callbacks for diffusion steps
- Legacy compatibility surface kept intact for older code paths
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

`InstructImage` is implemented as a first-class wrapper mode on top of the
bundled native `img2img` path, which keeps the addon compatible with the
current bundled C API while still giving callers a clearer editing-oriented API.

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

More detail: [docs/NATIVE_BUILD.md](/C:/Users/Jonathan%20Frank/Desktop/of_v20260406_vs_64_release/addons/ofxStableDiffusion/docs/NATIVE_BUILD.md)

## Current Native Source Status

The repo currently includes staged native binaries and headers, but the full
vendored upstream `stable-diffusion.cpp` source snapshot is not yet present in
`libs/stable-diffusion/source`.

That means the rebuild scripts are ready, but full native rebuilds stay blocked
until a compatible upstream snapshot is pinned and vendored.

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
