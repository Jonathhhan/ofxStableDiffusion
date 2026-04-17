# Architecture

`ofxStableDiffusion` is structured as an openFrameworks-friendly wrapper around a
bundled `stable-diffusion.cpp` runtime, with a bias toward:

- A stable addon-facing API
- Background-thread generation
- Compatibility with existing `ofxGgml` integration points
- Native-runtime isolation so upstream `ggml` changes do not destabilize the addon

## Layers

### 1. Public addon wrapper

Files:

- `src/ofxStableDiffusion.h`
- `src/ofxStableDiffusion.cpp`

Responsibilities:

- Present a typed openFrameworks API
- Preserve the legacy `newSdCtx` / `txt2img` / `img2img` / `img2vid` surface
- Store owned image/video result objects
- Mirror enough public state to remain usable from older calling code

### 2. Typed addon data model

Files:

- `src/core/ofxStableDiffusionEnums.h`
- `src/core/ofxStableDiffusionTypes.h`

Responsibilities:

- Define addon-level tasks and video modes
- Separate context settings, image requests, video requests, upscaler settings
- Represent owned results (`ofxStableDiffusionResult`, `ofxStableDiffusionVideoClip`)

### 3. Background execution

Files:

- `src/ofxStableDiffusionThread.h`
- `src/ofxStableDiffusionThread.cpp`

Responsibilities:

- Own the native contexts
- Run load/generation/upscale work off the main thread
- Forward progress callbacks
- Hand raw native outputs back to the wrapper for conversion into owned results

### 4. Video helpers

Files:

- `src/video/ofxStableDiffusionVideo.cpp`
- `src/video/ofxStableDiffusionVideoHelpers.h`

Responsibilities:

- Build wrapper-owned frame sequences
- Apply presentation modes such as `Loop`, `PingPong`, and `Boomerang`
- Provide a small pure helper seam for unit tests

### 4b. Image ranking helpers

Files:

- `src/core/ofxStableDiffusionRankingHelpers.h`

Responsibilities:

- Define selection modes such as `KeepOrder`, `Rerank`, and `BestOnly`
- Support Best-of-N output scoring without importing a CLIP runtime directly
- Provide a clean callback seam for `ofxGgml` CLIP integration

### 5. Native runtime bundle

Files:

- `libs/stable-diffusion/include`
- `libs/stable-diffusion/lib`
- `libs/stable-diffusion/source`
- `scripts/build-stable-diffusion.*`

Responsibilities:

- Keep the upstream runtime version pinned inside the addon
- Rebuild and stage headers/libs into predictable addon paths
- Avoid sharing the low-level `ggml` build from `ofxGgml`

## `ofxGgml` Integration Policy

The recommended integration model is:

- `ofxStableDiffusion` owns its native `stable-diffusion.cpp` and `ggml` runtime
- `ofxGgml` talks to `ofxStableDiffusion` through the addon API
- Shared behavior should live at the wrapper layer, not in a shared native binary

This keeps the addon safer across upstream `ggml` changes, backend toggles, and
ABI differences.
