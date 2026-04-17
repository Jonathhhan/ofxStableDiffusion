# Native Build

`ofxStableDiffusion` stages the native runtime into addon-local folders instead
of relying on a global install.

## Layout

- `libs/stable-diffusion/source`
  A vendored snapshot of upstream `stable-diffusion.cpp`
- `libs/stable-diffusion/include`
  The staged public header used by the addon wrapper
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

### Linux / macOS-style shell

```bash
./scripts/build-stable-diffusion.sh
```

## Current Blocker

This repository currently stages prebuilt native libraries, but does not yet
vendor the full upstream source tree into `libs/stable-diffusion/source`.

Until that snapshot is added, the build script will fail early with a clear
message instead of producing a partial or mismatched build.

## Recommended Pinning Policy

When vendoring upstream:

1. Choose a known-compatible upstream commit.
2. Record that commit hash in this document and the changelog.
3. Rebuild the staged header/library pair together.
4. Re-run the wrapper test suite and the example build.
