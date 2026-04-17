# Native Build

`ofxStableDiffusion` stages the native runtime into addon-local folders instead
of relying on a global install.

## Layout

- `libs/stable-diffusion/source`
  A vendored snapshot of upstream `stable-diffusion.cpp`
- `libs/stable-diffusion/include`
  Thin passthrough include for the vendored upstream header
- `libs/stable-diffusion/lib/vs`
  Windows import library and runtime DLL
- `libs/stable-diffusion/lib/Linux64`
  Linux shared library staging location

## Why It Is Standalone

This addon intentionally does **not** link against the `ggml` build bundled by
`ofxGgml`.

Reasons:

- `stable-diffusion.cpp` often expects a specific `ggml` revision or patch set
- backend flags can differ across diffusion and language runtimes
- a shared native binary would make addon upgrades more fragile

The right place to integrate with `ofxGgml` is the addon/API layer.

## Rebuild Workflow

### Windows

```powershell
.\scripts\build-stable-diffusion.ps1 -Configuration Release
```

Or use the addon-level setup entrypoint:

```bat
scripts\setup_windows.bat --auto
```

Or stage a pinned upstream Windows release instead of building from source:

```bat
scripts\setup_windows.bat --use-release --auto
```

### Linux / macOS-style shell

```bash
./scripts/build-stable-diffusion.sh
```

## Backend Flags

The native build scripts now mirror the backend flag style used by `ofxGgml`.

- `--auto` / `-Auto`
  Auto-detect available GPU backends and enable them when the required SDKs are present
- `--cpu`, `--cpu-only` / `-CpuOnly`
  Disable GPU backends and build CPU-only
- `--gpu`, `--cuda` / `-Cuda`
  Enable CUDA explicitly
- `--vulkan` / `-Vulkan`
  Enable Vulkan explicitly
- `--metal` / `-Metal`
  Enable Metal explicitly where supported

On Windows, `scripts/setup_windows.bat` forwards the same flags into
`scripts/setup_addon.ps1`, which builds the native runtime and, by default,
also rebuilds `ofxStableDiffusionExample`.

## Optional Prebuilt Windows Runtime

For faster setup on Windows, the addon can stage a pinned upstream release from
`stable-diffusion.cpp` instead of compiling from source.

Available setup flags:

- `--use-release`
  Use the prebuilt upstream runtime path
- `--release-tag TAG`
  Override the pinned upstream release tag
- `--release-variant auto|cpu|noavx|avx|avx2|avx512|cuda12`
  Choose the Windows release package to stage

Behavior:

- `--auto` stays the default behavior overall
- `--use-release --auto` prefers `cuda12` when CUDA is available, otherwise `avx2`
- `--use-release --cuda` stages the `cuda12` release directly
- `--use-release --cpu` stages a CPU release
- Windows Vulkan is still a source-build path, because the upstream Windows release assets do not currently include a Vulkan package
- If the upstream Windows zip omits `stable-diffusion.lib`, the addon script synthesizes the import library from the downloaded DLL exports automatically

## Current Pin

The addon currently vendors upstream `stable-diffusion.cpp` from:

- repo: `https://github.com/leejet/stable-diffusion.cpp`
- commit: `a564fdf642780d1df123f1c413b19961375b8346`
- vendored on: `2026-04-17`

The optional Windows prebuilt-runtime path is currently pinned to:

- release tag: `master-572-1b4e9be`
- published: `2026-04-16`
- release page: [stable-diffusion.cpp releases](https://github.com/leejet/stable-diffusion.cpp/releases)

The vendored tree includes the required submodules so the native rebuild scripts
can run end-to-end.

## Header Notes

Current upstream master uses a newer parameter-struct based C API. The addon
keeps `libs/stable-diffusion/include/stable-diffusion.h` only as a passthrough
include to the vendored upstream header, so addon code should use the current
upstream names directly.

The native rebuild script stages:

- `stable-diffusion.dll`
- `stable-diffusion.lib`

It intentionally does not rewrite the addon include path.

## Recommended Pinning Policy

When vendoring upstream:

1. Choose a known-compatible upstream commit.
2. Record that commit hash in this document and the changelog.
3. Rebuild the staged header/library pair together.
4. Re-run the wrapper test suite and the example build.
