# Changelog

## 1.0.0 - 2026-04-17

### Added

- Typed addon-side request/result structs for context, image, video, and upscaler configuration
- Owned image/video result handling with `ofxStableDiffusionResult` and `ofxStableDiffusionVideoClip`
- Expanded image generation modes: `TextToImage`, `ImageToImage`, `InstructImage`, `Variation`, and `Restyle`
- Best-of-N image ranking support with callback-based reranking and per-image score metadata
- Expanded video presentation modes: `Standard`, `Loop`, `PingPong`, and `Boomerang`
- Native rebuild scripts for Windows, batch-shell, and POSIX-shell workflows
- Windows setup entrypoints via `scripts/setup_addon.ps1` and `scripts/setup_windows.bat`
- Optional Windows prebuilt-runtime staging via `scripts/download-stable-diffusion-release.ps1`
- Architecture and native build documentation under `docs/`
- A lightweight CMake-based unit test suite for video helper behavior
- A vendored current-master `stable-diffusion.cpp` source snapshot, pinned to `a564fdf642780d1df123f1c413b19961375b8346`

### Changed

- Reworked the wrapper toward a more structured addon API while preserving the legacy `newSdCtx` / `txt2img` / `img2img` / `img2vid` surface
- Migrated the native bridge to current upstream `stable-diffusion.cpp` master using the newer parameter-struct API under the hood
- Removed the temporary enum compatibility shim so addon code now uses the current upstream sample/scheduler names directly
- Added a first-class `InstructImage` wrapper mode over the addon request layer instead of relying on the older native `img2img` entry point
- Added a bridge-friendly CLIP rerank seam so `ofxGgml` can score and reorder outputs without native binary coupling
- Improved example-app UX with explicit busy states, progress/error feedback, video-mode controls, and frame-sequence export
- Updated native rebuild/staging so the script targets current upstream CMake flags and preserves the addon compatibility header while refreshing DLL/lib artifacts
- Aligned native build/setup flags with `ofxGgml` style, including `auto`, `cpu-only`, `cuda`, `vulkan`, and `metal` backend selection behavior
- Kept `--auto` as the default setup mode while adding an explicit `--use-release` path for pinned upstream Windows CPU/CUDA runtime staging
- Added automatic import-library synthesis from upstream Windows release DLL exports when the zip omits `stable-diffusion.lib`

### Notes

- The addon still stages prebuilt native artifacts, but the full upstream source snapshot is now vendored in-repo and rebuildable
- The intended integration path with `ofxGgml` remains wrapper-level coordination, not a shared native `ggml` binary
