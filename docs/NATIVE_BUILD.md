# Native Build

`ofxStableDiffusion` stages the native runtime into addon-local folders instead
of relying on a global install.

## Layout

- `libs/stable-diffusion/source`
  A vendored snapshot of upstream `stable-diffusion.cpp`
- `libs/stable-diffusion/include`
  The addon compatibility header used by the wrapper and examples
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

## Current Pin

The addon currently vendors upstream `stable-diffusion.cpp` from:

- repo: `https://github.com/leejet/stable-diffusion.cpp`
- commit: `a564fdf642780d1df123f1c413b19961375b8346`
- vendored on: `2026-04-17`

The vendored tree includes the required submodules so the native rebuild scripts
can run end-to-end.

## Compatibility Notes

Current upstream master uses a newer parameter-struct based C API. The addon
keeps `libs/stable-diffusion/include/stable-diffusion.h` as a small wrapper
shim over `libs/stable-diffusion/source/include/stable-diffusion.h` so older
addon-facing enum names such as `EULER_A`, `DPMPP2Mv2`, `DEFAULT`, `DISCRETE`,
and `KARRAS` still compile.

The native rebuild script stages:

- `stable-diffusion.dll`
- `stable-diffusion.lib`

It intentionally does not overwrite the addon compatibility header.

## Recommended Pinning Policy

When vendoring upstream:

1. Choose a known-compatible upstream commit.
2. Record that commit hash in this document and the changelog.
3. Rebuild the staged header/library pair together.
4. Re-run the wrapper test suite and the example build.
