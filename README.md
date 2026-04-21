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
- Expanded image modes: `TextToImage`, `ImageToImage`, `InstructImage`, `Variation`, and `Restyle`
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
        ofxStableDiffusionImageMode::Restyle);

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

### Advanced Animation Features

**Interpolation Modes:**
- `Linear` - Simple linear interpolation
- `Smooth` - Cosine-based smooth transitions
- `EaseIn`, `EaseOut`, `EaseInOut` - Cubic easing functions
- `Cubic` - Smoothstep interpolation
- `Back` - Overshooting easing
- `Elastic` - Spring-like oscillation
- `Bounce` - Bouncing effect
- `Expo` - Exponential acceleration
- `CatmullRom`, `BSpline` - Spline-based interpolation (multi-keyframe)

**Seed Variation Modes:**
- `Sequential` - Linear increment (default)
- `Noise` - Pseudo-random noise-based variation for organic feel
- `Random` - Reproducible random variation per frame

**Temporal Coherence:**
Control frame-to-frame consistency with the `temporalCoherence` setting (0.0-1.0).
Higher values produce smoother transitions between frames.

**Frame Count:** Supports up to 300 frames (increased from 100) for longer video sequences.

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

### Video Model Support

The addon now detects video-specific model families:
- **SVD** (Stable Video Diffusion) - Dedicated video generation models
- **AnimateDiff** - Motion module-based animation
- **WAN** family (WANI2V, WANTI2V, WANFLF2V, WANVACE) - Specialized video models

Model capabilities are automatically detected and exposed through `getCapabilities()`.

### Workflow & Usability Features

The addon provides comprehensive workflow tools for efficient video generation:

**Quick Preview Modes:**
- `QuickPreview_8` - Every 8th frame at 256x384 (ultra-fast iteration)
- `QuickPreview_4` - Every 4th frame at 384x576 (fast preview)
- `QuickPreview_2` - Every 2nd frame at 512x768 (quick validation)

**Storyboard Generation:**
- `Storyboard_6` - Generate 6 evenly-spaced keyframes
- `Storyboard_12` - Generate 12 keyframes for detailed planning

**High Quality Modes:**
- `HighQuality_24fps` - 24 FPS cinematic quality
- `HighQuality_30fps` - 30 FPS smooth motion

**Dry-Run Estimation:**
```cpp
auto estimate = ofxStableDiffusionEstimateVideoRender(request);
ofLogNotice() << "Estimated time: " << estimate.estimatedMinutes << " minutes";
ofLogNotice() << "Memory usage: " << estimate.estimatedMemoryMB << " MB";
ofLogNotice() << "Recommendation: " << estimate.recommendation;
```

**Preset Composition:**
Combine multiple presets to create custom workflows:
```cpp
std::vector<ofxStableDiffusionVideoWorkflowPreset> presets = {
    ofxStableDiffusionVideoWorkflowPreset::LowVram,
    ofxStableDiffusionVideoWorkflowPreset::QuickPreview_4
};
auto composed = ofxStableDiffusionComposePresets(baseRequest, presets);
```

**Video Templates:**
Reusable animation patterns for common scenarios:
```cpp
// Fade between two prompts
auto fadeTemplate = ofxStableDiffusionCreateFadeTemplate(
    "sunrise over mountains",
    "sunset over ocean",
    24);
auto request = ofxStableDiffusionApplyTemplate(baseRequest, fadeTemplate);

// Pulsing strength variation
auto pulseTemplate = ofxStableDiffusionCreatePulseTemplate(24, 0.4f, 0.8f);
```

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

### Image Generation Workflow Features

The addon provides comprehensive workflow tools for efficient image generation, parallel to the video generation features:

#### Workflow Presets

Eight quality/speed presets for different use cases:

**Quick Iteration:**
- `QuickDraft` - 4 steps, fastest iteration for rapid experimentation
- `FastPreview` - 12 steps, quick feedback with reasonable quality

**Balanced Quality:**
- `Balanced` - 24 steps, balanced quality and speed (default)
- `HighQuality` - 50 steps, maximum detail and refinement

**Specialized:**
- `DetailEnhance` - Optimized for upscaling and subtle refinement (35-40 steps, light strength)
- `StyleTransfer` - Tuned for img2img artistic transformation (35 steps, heavier strength)
- `ProductionReady` - Conservative settings for reliable final output (45 steps, single image)
- `ExperimentalHigh` - Aggressive settings for creative exploration (max steps/CFG)

Apply presets:
```cpp
#include "image/ofxStableDiffusionImageWorkflowHelpers.h"

ofxStableDiffusionImageRequest request;
request.prompt = "cinematic portrait";
request.width = 512;
request.height = 512;

// Apply preset automatically adjusts parameters based on model family
ofxStableDiffusionImageWorkflowHelpers::applyWorkflowPreset(
    request,
    ofxStableDiffusionImageWorkflowPreset::HighQuality,
    contextSettings);
```

#### Dry-Run Estimation

Estimate generation time and resource requirements before starting:

```cpp
auto estimate = ofxStableDiffusionImageWorkflowHelpers::estimateImageGeneration(request, contextSettings);

ofLogNotice() << "Total images: " << estimate.totalImages;
ofLogNotice() << "Total steps: " << estimate.totalSteps;
ofLogNotice() << "Estimated time: " << estimate.estimatedMinutes << " minutes";
ofLogNotice() << "Estimated memory: " << estimate.estimatedMemoryMB << " MB";
ofLogNotice() << "Recommendation: " << estimate.recommendation;

if (!estimate.feasible) {
    ofLogWarning() << "Generation may not be feasible with current settings";
}

for (const auto& warning : estimate.warnings) {
    ofLogWarning() << warning;
}
```

#### Enhanced Validation with Auto-Correction

Validate requests and get automatic correction suggestions:

```cpp
auto validation = ofxStableDiffusionImageWorkflowHelpers::validateImageRequestWithCorrection(
    request, contextSettings);

if (!validation.isValid) {
    for (const auto& error : validation.errors) {
        ofLogError() << error;
    }
}

for (const auto& warning : validation.warnings) {
    ofLogWarning() << warning;
}

if (validation.hasCorrectedRequest) {
    for (const auto& suggestion : validation.suggestions) {
        ofLogNotice() << suggestion;
    }
    // Optionally use the corrected request
    request = validation.correctedRequest;
}
```

The validator automatically:
- Rounds dimensions to multiples of 64
- Scales down oversized images proportionally
- Clamps parameters to model-appropriate ranges
- Warns about suboptimal settings (e.g., high steps for turbo models)
- Checks mode requirements (e.g., input images for img2img modes)

#### Batch Diversity & Parameter Sweeps

Generate varied batches systematically:

```cpp
// Create multiple requests with diversity
std::vector<ofxStableDiffusionImageRequest> requests(5, baseRequest);

ofxStableDiffusionBatchDiversitySettings diversity;
diversity.mode = ofxStableDiffusionBatchDiversityMode::Sequential;
diversity.seedIncrement = 1;

ofxStableDiffusionImageWorkflowHelpers::applyBatchDiversity(requests, diversity);

// Parameter sweep across CFG scale
diversity.mode = ofxStableDiffusionBatchDiversityMode::ParameterSweep;
diversity.cfgScaleStart = 5.0f;
diversity.cfgScaleEnd = 10.0f;
ofxStableDiffusionImageWorkflowHelpers::applyBatchDiversity(requests, diversity);
```

**Diversity Modes:**
- `None` - All images use same seed
- `Sequential` - Increment seed by 1
- `LargeSteps` - Increment seed by 1000
- `Random` - Completely random seeds
- `ParameterSweep` - Vary CFG scale or strength systematically

#### Seed Exploration

Generate variations around a successful seed:

```cpp
ofxStableDiffusionSeedExplorationSettings exploration;
exploration.centerSeed = 42;
exploration.gridSize = 3;  // 3x3 = 9 variations
exploration.seedRadius = 100;
exploration.useRadialPattern = true;

auto seeds = ofxStableDiffusionImageWorkflowHelpers::generateSeedExplorationGrid(exploration);

// Create request for each seed
auto requests = ofxStableDiffusionImageCompositionHelpers::createSeedExplorationRequests(
    baseRequest, seeds);
```

#### Image Templates

Built-in templates for common workflows:

```cpp
// Portrait photography
auto portraitTemplate = ofxStableDiffusionImageWorkflowHelpers::getPortraitTemplate();
request.width = portraitTemplate.baseRequest.width;
request.height = portraitTemplate.baseRequest.height;
request.cfgScale = portraitTemplate.baseRequest.cfgScale;
request.prompt = portraitTemplate.promptSuggestions[0];
request.negativePrompt = portraitTemplate.negativePromptSuggestions[0];

// Other built-in templates:
auto landscapeTemplate = ofxStableDiffusionImageWorkflowHelpers::getLandscapeTemplate();
auto conceptArtTemplate = ofxStableDiffusionImageWorkflowHelpers::getConceptArtTemplate();
auto productShotTemplate = ofxStableDiffusionImageWorkflowHelpers::getProductShotTemplate();
```

#### Advanced Prompt Features

Prompt weighting and template system:

```cpp
#include "image/ofxStableDiffusionPromptHelpers.h"

// Emphasize/de-emphasize tokens
std::string prompt = ofxStableDiffusionPromptHelpers::emphasize("detailed face", 1.5f) + ", " +
                     ofxStableDiffusionPromptHelpers::deemphasize("background", 0.7f);
// Result: "(detailed face:1.5), [background:0.7]"

// Build weighted prompt
std::vector<ofxStableDiffusionPromptWeight> components = {
    {"portrait", 1.3f},
    {"professional lighting", 1.2f},
    {"detailed", 1.1f},
    {"background", 0.6f}
};
prompt = ofxStableDiffusionPromptHelpers::buildWeightedPrompt(components);

// Negative prompt presets
std::string negPrompt = ofxStableDiffusionPromptHelpers::getNegativePromptPreset("comprehensive");
// Includes: quality issues, distortions, artifacts, watermarks

// Available presets: "quality", "artifacts", "distortion", "style", "realistic", "comprehensive"

// Style mixing
auto styleMix = ofxStableDiffusionPromptHelpers::getStyleMixPreset("cinematic");
prompt += ", " + styleMix.toPrompt();
// Available styles: "cinematic", "anime", "photorealistic", "artistic", "fantasy", "scifi"

// Prompt templates with variables
auto charTemplate = ofxStableDiffusionPromptHelpers::getCharacterTemplate();
charTemplate.variables = {
    {"adjective", "mysterious"},
    {"subject", "wizard"},
    {"style", "fantasy art"},
    {"lighting", "dramatic lighting"},
    {"quality", "highly detailed, 4k"}
};
prompt = charTemplate.apply();

// Prompt cleanup and validation
prompt = ofxStableDiffusionPromptHelpers::cleanupPrompt(prompt);
if (ofxStableDiffusionPromptHelpers::isPromptTooLong(prompt, 75)) {
    prompt = ofxStableDiffusionPromptHelpers::truncatePrompt(prompt, 75);
}
```

#### Image Composition & Blending

Blend and compose images:

```cpp
#include "image/ofxStableDiffusionImageCompositionHelpers.h"

// Blend two images
ofPixels blended;
ofxStableDiffusionImageCompositionHelpers::blendImages(
    imageA.pixels, imageB.pixels, blended,
    0.5f,  // blend amount
    ofxStableDiffusionBlendMode::Smoothstep);

// Available blend modes: Linear, Smoothstep, Cosine, Cubic, Overlay, Screen, Multiply

// Create comparison grid
ofxStableDiffusionGridLayout layout;
layout.columns = 3;
layout.rows = 3;
layout.cellWidth = 512;
layout.cellHeight = 512;
layout.padding = 4;
layout.drawBorders = true;
layout.addLabels = true;

ofxStableDiffusionComparisonConfig config;
config.showParameters = true;
config.showSeeds = true;
config.labels = {"Seed 42", "Seed 43", "Seed 44", /* ... */};

ofPixels grid;
ofxStableDiffusionImageCompositionHelpers::createComparisonGrid(
    generatedFrames, grid, layout, config);

// A/B comparison (side-by-side)
ofPixels comparison;
ofxStableDiffusionImageCompositionHelpers::createABComparison(
    imageA.pixels, imageB.pixels, comparison, 8 /* padding */);

// Calculate similarity between images
float similarity = ofxStableDiffusionImageCompositionHelpers::calculateImageSimilarity(
    imageA.pixels, imageB.pixels);
ofLogNotice() << "Images are " << (similarity * 100.0f) << "% similar";

// Interpolate between images (morphing)
std::vector<ofPixels> morphFrames;
ofxStableDiffusionImageCompositionHelpers::interpolateImages(
    startImage.pixels, endImage.pixels, morphFrames,
    24,  // frame count
    ofxStableDiffusionBlendMode::Smoothstep);
```

#### Parameter Sweep Generation

Test parameter ranges systematically:

```cpp
// Create requests sweeping CFG scale from 5.0 to 10.0
auto requests = ofxStableDiffusionImageCompositionHelpers::createParameterSweepRequests(
    baseRequest,
    "cfgScale",  // parameter name: "cfgScale", "steps", or "strength"
    5.0f,        // min value
    10.0f,       // max value
    6);          // step count

// Generate all variations and create comparison grid
std::vector<ofxStableDiffusionImageFrame> results;
for (auto& req : requests) {
    auto result = sd.generate(req);
    if (result.success && !result.images.empty()) {
        results.push_back(result.images[0]);
    }
}

ofPixels comparisonGrid;
ofxStableDiffusionImageCompositionHelpers::createComparisonGrid(
    results, comparisonGrid, layout, config);
```

#### Progress Tracking

Enhanced progress information during generation:

```cpp
ofxStableDiffusionImageProgressInfo progress;
progress.currentImage = 3;
progress.totalImages = 8;
progress.currentStep = 15;
progress.totalSteps = 20;
progress.elapsedSeconds = 45.5f;
progress.estimatedRemainingSeconds = 60.2f;
progress.percentComplete = 0.75f;
progress.currentPhase = "generating";

ofLogNotice() << "Progress: " << (progress.percentComplete * 100.0f) << "%";
ofLogNotice() << "ETA: " << progress.getETA();
ofLogNotice() << "Phase: " << progress.currentPhase;
```

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

More detail: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

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

More detail: [docs/NATIVE_BUILD.md](docs/NATIVE_BUILD.md)

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

For Windows, `scripts/setup_windows.bat` and `scripts/setup_addon.ps1` now
always refresh the vendored source from the latest upstream release-tag source
snapshot, then build the native runtime locally.

- `--source-release-tag TAG`
  Override the upstream release tag used for the vendored source snapshot

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

## Video Performance & Quality Optimization

The addon provides comprehensive tools for optimizing video generation performance and quality.

### Performance Helpers

Include the performance helpers for advanced optimizations:

```cpp
#include "video/ofxStableDiffusionVideoPerformanceHelpers.h"
```

### Adaptive Quality Scaling

Automatically adjust quality parameters based on generation performance:

```cpp
ofxStableDiffusionAdaptiveQualitySettings adaptiveSettings;
adaptiveSettings.enabled = true;
adaptiveSettings.targetSecondsPerFrame = 2.0f;  // Target 2 seconds per frame
adaptiveSettings.minSampleSteps = 8;
adaptiveSettings.maxSampleSteps = 50;
adaptiveSettings.qualityAdjustmentRate = 0.2f;
adaptiveSettings.warmupFrames = 2;

ofxStableDiffusionVideoQualityMetrics metrics;

for (int frame = 0; frame < request.frameCount; ++frame) {
    float startTime = ofGetElapsedTimef();

    // Generate frame...

    float elapsed = ofGetElapsedTimef() - startTime;
    metrics.recordFrameTime(elapsed);

    // Adjust quality for next frame
    ofxStableDiffusionVideoPerformanceHelpers::adjustQualityForPerformance(
        request, adaptiveSettings, metrics);
}
```

### Temporal Consistency Enhancement

Improve frame-to-frame consistency for smoother videos:

```cpp
ofxStableDiffusionTemporalConsistencySettings temporalSettings;
temporalSettings.enabled = true;
temporalSettings.strengthModulation = 0.1f;  // Slight strength variation
temporalSettings.cfgScaleModulation = 0.2f;  // Slight CFG variation
temporalSettings.useFrameBlending = false;
temporalSettings.blendWeight = 0.15f;

for (int frame = 0; frame < request.frameCount; ++frame) {
    ofxStableDiffusionVideoPerformanceHelpers::applyTemporalConsistency(
        frameRequest, frame, temporalSettings, previousFrame);

    // Generate frame...
}
```

### Frame Caching

Cache generated frames for faster iteration:

```cpp
ofxStableDiffusionFrameCache cache;
cache.enabled = true;
cache.maxCachedFrames = 10;

for (int frame = 0; frame < request.frameCount; ++frame) {
    if (cache.hasFrame(frame)) {
        // Reuse cached frame
        const auto* cachedFrame = cache.getFrame(frame);
        // Use cached frame...
    } else {
        // Generate new frame
        auto newFrame = generateFrame(frame);
        cache.addFrame(frame, newFrame);
    }
}
```

### Memory Optimization

Optimize memory usage for large video generations:

```cpp
ofxStableDiffusionVideoMemorySettings memorySettings;
memorySettings.enableStreamingMode = true;  // Save frames as they're generated
memorySettings.clearIntermediateBuffers = true;
memorySettings.maxFramesInMemory = 30;
memorySettings.streamingOutputDirectory = "output/video_frames";

// Estimate memory usage
float estimatedMB = ofxStableDiffusionVideoPerformanceHelpers::estimateVideoMemoryUsageMB(
    request, memorySettings.maxFramesInMemory);

ofLogNotice() << "Estimated memory usage: " << estimatedMB << " MB";

// Calculate optimal batch strategy
auto batches = ofxStableDiffusionVideoPerformanceHelpers::calculateOptimalBatchStrategy(
    request.frameCount,
    memorySettings.maxFramesInMemory,
    memorySettings);
```

### Quality Metrics & Validation

Track and validate video generation quality:

```cpp
ofxStableDiffusionVideoQualityMetrics metrics;

// Record frame generation times
for (int frame = 0; frame < totalFrames; ++frame) {
    float startTime = ofGetElapsedTimef();
    // Generate frame...
    float elapsed = ofGetElapsedTimef() - startTime;
    metrics.recordFrameTime(elapsed);
}

// Analyze performance
ofLogNotice() << "Average time per frame: " << metrics.averageGenerationTimePerFrame << "s";
ofLogNotice() << "Min/Max time: " << metrics.minGenerationTime << "s / "
              << metrics.maxGenerationTime << "s";
ofLogNotice() << "Time variance: " << metrics.getVariance();
ofLogNotice() << "Std deviation: " << metrics.getStdDeviation();

// Calculate frame-to-frame similarity for temporal coherence
float similarity = ofxStableDiffusionVideoPerformanceHelpers::calculateFrameSimilarity(
    frames[i], frames[i+1], 8 /* sample step */);
ofLogNotice() << "Frame similarity: " << (similarity * 100.0f) << "%";

// Validate video parameters
auto warnings = ofxStableDiffusionVideoPerformanceHelpers::validateVideoPerformance(request);
for (const auto& warning : warnings) {
    ofLogWarning() << warning;
}
```

### Quick Optimization Presets

Apply performance/quality trade-offs with preset targets:

```cpp
ofxStableDiffusionVideoRequest request;
request.prompt = "cinematic sequence";
request.width = 768;
request.height = 1024;
request.frameCount = 48;

// Optimize for different targets
ofxStableDiffusionVideoPerformanceHelpers::optimizeVideoRequestForTarget(
    request, "ultrafast");  // ultrafast, fast, balanced, quality, highquality

// Or optimize for smooth playback
auto smoothRequest = ofxStableDiffusionVideoPerformanceHelpers::optimizeForSmoothPlayback(
    request, 3.0f /* target duration in seconds */);
```

### Performance Best Practices

**For Fast Iteration:**
- Use `QuickPreview_8` or `QuickPreview_4` presets
- Reduce resolution (384x576 or smaller)
- Lower sample steps (8-12)
- Reduce frame count (8-16 frames)
- Use `FastPreview` workflow preset

**For Balanced Quality:**
- Use `Balanced` workflow preset
- 512x768 resolution
- 20-24 sample steps
- 12-24 frames at 10-12 FPS
- Enable temporal coherence (0.5-0.7)

**For High Quality:**
- Use `HighQuality_24fps` or `HighQuality_30fps` presets
- 768x1024 or higher resolution
- 35-50 sample steps
- Enable temporal consistency enhancements
- Use smooth interpolation modes
- Higher temporal coherence (0.7-0.9)

**Memory Management:**
- Enable streaming mode for very long videos (>100 frames)
- Limit frames in memory to 30-50
- Clear intermediate buffers between frames
- Use batch processing for large projects

**Temporal Quality:**
- Set `temporalCoherence` to 0.7+ for smooth transitions
- Use `Noise` seed variation mode instead of `Sequential`
- Apply slight strength/CFG modulation (0.1-0.2)
- Enable prompt interpolation with `Smooth` mode

**Resolution vs Speed:**
- 256x384: Ultra-fast previews (~5-10s per frame)
- 384x576: Fast iteration (~15-30s per frame)
- 512x768: Balanced quality (~30-60s per frame)
- 768x1024: High quality (~60-120s per frame)
- 1024x1536: Maximum quality (~120-300s per frame)

**Sample Steps Trade-offs:**
- 4-8 steps: Very fast, draft quality
- 12-16 steps: Fast preview, decent quality
- 20-28 steps: Balanced, good quality
- 35-50 steps: High quality, diminishing returns beyond 50

### Frame Skip Strategies

For quick previews, generate every Nth frame:

```cpp
int frameSkip = ofxStableDiffusionVideoPerformanceHelpers::calculateOptimalFrameSkip(
    totalFrames, 2.0f /* target duration */, 12 /* target FPS */);

for (int i = 0; i < totalFrames; i += frameSkip) {
    // Generate frame i only
}
```

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

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
