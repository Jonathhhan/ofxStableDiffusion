# `ofxGgml` Bridge Notes

This note captures the current integration seam between `ofxGgml` and
`ofxStableDiffusion`, plus the result of comparing their staged runtime
surfaces.

## Current Seam

`ofxGgml` already has an addon-level bridge in:

- `addons/ofxGgml/src/inference/ofxGgmlStableDiffusionAdapters.h`

That adapter:

- conditionally includes `ofxStableDiffusion.h`
- creates an image-generation backend named `ofxStableDiffusion`
- reloads `ofxStableDiffusion` contexts when model/runtime settings change
- forwards typed requests through the `ofxStableDiffusion` wrapper API

So the basic integration path already exists without sharing a low-level native
runtime.

## Staged Runtime Comparison

### `ofxGgml`

`ofxGgml` stages a full `ggml` SDK/runtime:

- public headers in `libs/ggml/include`
- public libs in `libs/ggml/lib`
- direct backend libs such as:
  - `ggml-cuda.lib`
  - `ggml-vulkan.lib`
  - `ggml-cpu.lib`

It also performs runtime backend selection in addon code.

### `ofxStableDiffusion`

`ofxStableDiffusion` stages a diffusion runtime surface only:

- public header in `libs/stable-diffusion/include/stable-diffusion.h`
- staged runtime in `libs/stable-diffusion/lib/...`
  - `stable-diffusion.dll`
  - `stable-diffusion.lib`

It does not stage a public `ggml` SDK surface for other addons to consume.

## Why Direct Runtime Sharing Is Not Clean

Even if `stable-diffusion.cpp` vendors an upstream-compatible `ggml`, the staged
surface in this addon is still not a drop-in replacement for `ofxGgml`:

- `ofxGgml` expects public `ggml` headers and backend-specific libs
- `ofxStableDiffusion` exports only the diffusion wrapper API
- backend selection policy differs
- ABI and build-flag drift would become a shared failure mode

So the safer architecture remains:

- `ofxGgml` keeps its own staged `ggml` runtime
- `ofxStableDiffusion` keeps its own staged diffusion runtime
- integration happens through the addon wrapper layer

## Safe Experiment Scope

For a shared-runtime experiment, the least risky path is:

1. keep both addons standalone
2. improve the addon-level bridge seam first
3. only attempt deeper runtime sharing if both staged public surfaces are
   intentionally redesigned to match

That keeps disk/build duplication as a known tradeoff, instead of turning it
into hidden coupling.
