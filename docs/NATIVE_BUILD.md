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
`scripts/setup_addon.ps1`, which builds the native runtime.

## Windows Source Snapshot Flow

On Windows, the addon setup entrypoints now always fetch the source snapshot
for the latest upstream `stable-diffusion.cpp` release tag, then refresh
`libs/stable-diffusion/source/ggml` from the latest upstream `ggml` release,
replace `libs/stable-diffusion/source`, and compile locally.

Available setup flags:

- `--source-release-tag TAG`
  Override the upstream release tag used for the vendored source snapshot
- `--ggml-release-tag TAG`
  Override the upstream `ggml` release tag used for the vendored `ggml` subtree

Example:

```bat
scripts\setup_windows.bat --cuda
```

Pin a specific source snapshot:

```bat
scripts\setup_windows.bat --cuda --source-release-tag master-585-44cca3d
```

Pin both upstream trees explicitly:

```bat
scripts\setup_windows.bat --cuda --source-release-tag master-585-44cca3d --ggml-release-tag v0.9.11
```

The legacy-named helper below now does the same source-refresh job instead of
staging prebuilt runtime binaries:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download-stable-diffusion-release.ps1 -ReleaseTag master-585-44cca3d
```

## Current Pin

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
4. Re-run the wrapper test suite.
