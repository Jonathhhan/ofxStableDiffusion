# Changelog

## Unreleased

### Added

- `getCapabilities()` plus capability/model-family helpers for backend-aware UI gating and workflow routing.
- Parameter tuning helpers for image/video defaults, range clamping, and model-family-specific guidance.
- Video clip metadata export helpers on both `ofxStableDiffusionVideoClip` and the wrapper API.
- Request-owned thread task payloads for image/video generation so background work no longer depends on mutable wrapper state during a run.
- Optional `ofxStableDiffusionHoloscanBridge` scaffolding for live `frame -> conditioning -> diffusion -> preview` workflows, designed as an addon-side bridge instead of a hard Holoscan dependency.
  - The native Holoscan runtime path is Linux-only for now; other platforms stay on the addon fallback lane until that runtime is validated there.
- A small example-side `Holoscan Bridge` panel for starting/stopping the bridge MVP, reusing current prompts, submitting the loaded image, and previewing the most recent bridge result.
- `ofxStableDiffusionVideoWorkflowHelpers.h` with reusable video-generation presets, prompt-workflow validation, and richer render-manifest export helpers for request/result based pipelines.

### Changed

- Default stable-diffusion.cpp vendoring is pinned to upstream release `master-585-44cca3d` (published 2026-04-19) for compatibility stability.

## 1.2.0 - 2026-04-17

### Added

- **Multi-ControlNet Support** (Feature #5): Multiple ControlNet models simultaneously
  - `ofxStableDiffusionControlNet` structure with condition image, strength, and type fields
  - Vector-based API for managing multiple ControlNets
  - `addControlNet()`, `clearControlNets()`, and `getControlNets()` methods
  - Per-model strength control and type hints (canny, depth, pose, etc.)
  - Backward compatibility with legacy single ControlNet API
  - `controlNets` vector in `ofxStableDiffusionImageRequest`

- **Complete Inpainting Implementation** (Feature #8): Selective image editing with full validation
  - Mask dimension validation (must match input image dimensions)
  - Proper error handling with descriptive error messages
  - `maskImage` field validation in image request processing
  - Complete inpainting mode support with `ofxStableDiffusionImageMode::Inpainting`

- **LoRA Discovery** (Feature #6 - Partial): LoRA file discovery and listing
  - `listLoras()` method returns name and absolute path pairs
  - Supports multiple file formats: .safetensors, .ckpt, .pt, .bin
  - Scans loraModelDir for available LoRA files
  - Complements existing LoRA management API

- **Performance Profiling System** (Feature #16): Comprehensive performance tracking and analysis
  - `ofxStableDiffusionPerformanceProfiler` class with timing and memory tracking
  - RAII-style scoped timers for automatic instrumentation
  - Thread-safe operations with std::mutex protection
  - Bottleneck identification with configurable thresholds
  - JSON and CSV export formats for performance data
  - Statistics aggregation across multiple operations
  - Main class API: `setProfilingEnabled()`, `getPerformanceStats()`, `getPerformanceBottlenecks()`
  - `printPerformanceSummary()`, `exportPerformanceJSON()`, `exportPerformanceCSV()`

- **Advanced Sampling Options** (Feature #13): Comprehensive sampler and scheduler configuration
  - `ofxStableDiffusionSamplingHelpers` with metadata for 15+ samplers and 11+ schedulers
  - 7 predefined quality/speed presets: Ultra Quality, Quality, Balanced, Fast, Ultra Fast, LCM, TCD
  - Sampler information with descriptions and recommended step ranges
  - Scheduler information with use case recommendations
  - Name-based lookup (case-insensitive) for samplers and schedulers
  - Recommended sampler/scheduler combinations
  - Step count validation and quality-based recommendation system
  - Support for: Euler, Euler A, Heun, DPM2, DPM++ variants, iPNDM, LCM, TCD, and more
  - Scheduler support: Discrete, Karras, Exponential, AYS, GITS, LCM, and more

- **Video Animation and Interpolation** (Feature #18): Advanced video generation with keyframe support
  - Prompt interpolation between keyframes with smooth transitions
  - Parameter animation for CFG scale, strength, and other settings
  - Seed sequence support for incremental seed variation across frames
  - Multiple interpolation modes: Linear, Smooth (cosine), EaseIn, EaseOut, EaseInOut
  - `ofxStableDiffusionPromptKeyframe` structure for prompt-based animations
  - `ofxStableDiffusionKeyframe` structure for parameter animations
  - `ofxStableDiffusionVideoAnimationSettings` for animation configuration
  - Helper functions for creating animated video requests
  - Frame-specific parameter getter functions (prompt, cfgScale, strength, seed)

- **Video Export Enhancements** (Feature #19): Improved video metadata and export capabilities
  - Automatic frame metadata generation (JSON format)
  - Generation parameter embedding for each frame
  - Video parameter export to JSON file
  - Complete generation history tracking per frame
  - Export helper functions for metadata and parameters

- **Test Coverage Expansion**: Comprehensive unit tests for new features
  - 15 total tests (up from 9), 100% passing
  - Tests for LoRA discovery functionality
  - Tests for inpainting validation logic
  - Tests for model manager integration
  - Tests for Multi-ControlNet support
  - Tests for performance profiler
  - Tests for sampling helpers

## 1.1.0 - 2026-04-17

### Added

- **Seed Management** (Feature #7): Track generation seeds and reproducibility
  - `getLastUsedSeed()` returns the actual seed used in the last generation
  - `getSeedHistory()` returns up to 20 recent seed values
  - `clearSeedHistory()` clears the seed history
  - `hashStringToSeed()` static method to generate deterministic seeds from strings
  - `actualSeedUsed` field added to `ofxStableDiffusionResult` to capture the real seed (including auto-generated ones)

- **Inpainting Mode** (Feature #8 - Partial): Selective image editing with mask support
  - Added `Inpainting` to `ofxStableDiffusionImageMode` enum
  - Added `maskImage` field to `ofxStableDiffusionImageRequest` for mask input (white=inpaint, black=keep)
  - Added `maskBlur` parameter for mask edge softening (default: 4.0f)
  - Inpainting task enum added for proper mode handling

- **Model Management System** (Feature #2): Already implemented via `ofxStableDiffusionModelManager`
  - Model preloading and caching with LRU eviction
  - Model metadata extraction and validation
  - Cache size and count limits
  - Model scanning from directories

- **Generation Queue System** (Feature #3): Already implemented via `ofxStableDiffusionQueue`
  - Priority-based request queuing (Low, Normal, High, Critical)
  - Request cancellation by ID or tag
  - Per-request completion, error, and progress callbacks
  - Queue statistics and state management
  - Optional queue persistence to file

- **Advanced Progress Reporting** (Feature #4): Already implemented via `ofxStableDiffusionProgressTracker`
  - Stage-based progress (Idle, LoadingModel, Encoding, Diffusing, Decoding, Upscaling, Finalizing)
  - ETA estimation with adaptive algorithm
  - Performance metrics (steps/second, average step time)
  - Memory usage tracking
  - Batch progress support

### Changed

- Seed values are now tracked and stored in generation results
- Image request structure extended with mask support for future inpainting implementation

## 1.0.0 - 2026-04-17

### Added

- Typed addon-side request/result structs for context, image, video, and upscaler configuration
- Owned image/video result handling with `ofxStableDiffusionResult` and `ofxStableDiffusionVideoClip`
- Native image generation modes: `TextToImage`, `ImageToImage`, and `Inpainting`
- Best-of-N image ranking support with callback-based reranking and per-image score metadata
- Expanded video presentation modes: `Standard`, `Loop`, `PingPong`, and `Boomerang`
- Native rebuild scripts for Windows, batch-shell, and POSIX-shell workflows
- Windows setup entrypoints via `scripts/setup_addon.ps1` and `scripts/setup_windows.bat`
- Release-tag source snapshot refresh via `scripts/download-stable-diffusion-release.ps1`
- Architecture and native build documentation under `docs/`
- A lightweight CMake-based unit test suite for video helper behavior
- A vendored current-master `stable-diffusion.cpp` source snapshot, pinned to `a564fdf642780d1df123f1c413b19961375b8346`

### Changed

- Reworked the wrapper toward a more structured addon API while preserving the legacy `newSdCtx` / `txt2img` / `img2img` / `img2vid` surface
- Migrated the native bridge to current upstream `stable-diffusion.cpp` master using the newer parameter-struct API under the hood
- Removed the temporary enum compatibility shim so addon code now uses the current upstream sample/scheduler names directly
- Removed wrapper-only image modes so the addon image-mode surface now tracks native image generation paths
- Added a bridge-friendly CLIP rerank seam so `ofxGgml` can score and reorder outputs without native binary coupling
- Improved example-app UX with explicit busy states, progress/error feedback, video-mode controls, and frame-sequence export
- Updated native rebuild/staging so the script targets current upstream CMake flags and preserves the addon compatibility header while refreshing DLL/lib artifacts
- Aligned native build/setup flags with `ofxGgml` style, including `auto`, `cpu-only`, `cuda`, `vulkan`, and `metal` backend selection behavior
- Kept `--auto` as the default setup mode while adding an explicit `--use-release` path for pinned upstream Windows CPU/CUDA runtime staging
- Added automatic import-library synthesis from upstream Windows release DLL exports when the zip omits `stable-diffusion.lib`

### Notes

- The addon still stages prebuilt native artifacts, but the full upstream source snapshot is now vendored in-repo and rebuildable
- The intended integration path with `ofxGgml` remains wrapper-level coordination, not a shared native `ggml` binary
