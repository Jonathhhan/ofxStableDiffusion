# Changelog

## Unreleased

### Added

- Typed addon-side request/result structs for context, image, video, and upscaler configuration
- Owned image/video result handling with `ofxStableDiffusionResult` and `ofxStableDiffusionVideoClip`
- Expanded image generation modes: `TextToImage`, `ImageToImage`, `InstructImage`, `Variation`, and `Restyle`
- Best-of-N image ranking support with callback-based reranking and per-image score metadata
- Expanded video presentation modes: `Standard`, `Loop`, `PingPong`, and `Boomerang`
- Native rebuild scripts for Windows, batch-shell, and POSIX-shell workflows
- Architecture and native build documentation under `docs/`
- A lightweight CMake-based unit test suite for video helper behavior

### Changed

- Reworked the wrapper toward a more structured addon API while preserving the legacy `newSdCtx` / `txt2img` / `img2img` / `img2vid` surface
- Added a first-class `InstructImage` wrapper mode over the existing native `img2img` backend
- Added a bridge-friendly CLIP rerank seam so `ofxGgml` can score and reorder outputs without native binary coupling
- Improved example-app UX with explicit busy states, progress/error feedback, video-mode controls, and frame-sequence export
- Staged native library handling in the example and addon config so DLL/lib locations are explicit and reproducible

### Notes

- The addon still ships prebuilt native artifacts, but the vendored upstream `stable-diffusion.cpp` source snapshot has not been added yet
- The intended integration path with `ofxGgml` remains wrapper-level coordination, not a shared native `ggml` binary
