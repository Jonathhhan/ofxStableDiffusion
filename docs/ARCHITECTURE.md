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

### Integration Points

`ofxStableDiffusion` provides several well-defined integration points for `ofxGgml`:

#### 1. CLIP-Based Image Ranking

Use the image ranking callback system for Best-of-N workflows:

```cpp
// After generating a batch with ofxStableDiffusion
auto rankedResults = ofxStableDiffusionRankingHelpers::rankImages(
    generatedImages,
    [&clipModel](const ofImage& img) -> float {
        return clipModel.scoreImage(img, targetPrompt);
    },
    ofxStableDiffusionSelectionMode::BestOnly
);
```

This callback is defined in:
- `src/core/ofxStableDiffusionRankingHelpers.h`

#### 2. Model Path Coordination

Both addons support model discovery through file-based APIs:

- `ofxStableDiffusion::listModels()` - discover Stable Diffusion models
- `ofxStableDiffusion::listLoras()` - discover LoRA files
- `ofxGgml` equivalent model listing APIs

Keep model directories separate to avoid path conflicts:
```
data/models/
  sd/         # Stable Diffusion models
  clip/       # CLIP models for ofxGgml
  loras/      # LoRA adapters
```

#### 3. Prompt Engineering Workflow

Combine CLIP interrogation with generation:

1. Use `ofxGgml` CLIP to analyze reference images
2. Extract semantic features or style descriptors
3. Build enhanced prompts programmatically
4. Feed to `ofxStableDiffusion::generate()`

This workflow keeps the addons loosely coupled while enabling sophisticated
prompt-driven generation pipelines.

#### 4. Progress and Metadata Sharing

Both addons use similar patterns for async operations:

- Progress callbacks during long operations
- Result objects with metadata (seeds, timing, parameters)
- Error handling with codes and suggestions

This makes it easier to build unified UI layers that coordinate both addons.

### Why Not Share the Native Runtime?

While both addons use `ggml` at the native layer, sharing the binary creates
several risks:

1. **Version conflicts**: `stable-diffusion.cpp` may pin a specific `ggml` commit
   that differs from what `ofxGgml` or other GGML tools require

2. **Backend incompatibility**: One addon may need CUDA while another uses Vulkan,
   leading to conflicting compile-time flags

3. **ABI fragility**: Native structs, enum layouts, and calling conventions can
   change between `ggml` versions, breaking compatibility

4. **Build complexity**: Coordinating a single shared build across multiple
   addons is harder to maintain than independent per-addon builds

5. **Update friction**: Upgrading one addon's native runtime would force updates
   across all dependent addons

By keeping runtimes separate, each addon can:
- Update independently
- Use optimal backend configurations
- Maintain stable wrapper APIs
- Avoid cross-addon breakage during development

The small cost of duplicate native binaries is outweighed by the architectural
flexibility and stability gains.
