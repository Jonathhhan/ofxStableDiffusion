# `ofxGgml` Bridge Notes

This note captures integration options between `ofxGgml` and `ofxStableDiffusion`.

## Integration Options

### 1. Addon-Level Integration (Recommended)

`ofxGgml` already has an addon-level bridge in:

- `addons/ofxGgml/src/inference/ofxGgmlStableDiffusionAdapters.h`

That adapter:

- conditionally includes `ofxStableDiffusion.h`
- creates an image-generation backend named `ofxStableDiffusion`
- reloads `ofxStableDiffusion` contexts when model/runtime settings change
- forwards typed requests through the `ofxStableDiffusion` wrapper API

This is the recommended integration path because it:
- Keeps native runtimes isolated and independently versioned
- Avoids ABI/version coupling between stable-diffusion.cpp and llama.cpp/whisper.cpp
- Allows each addon to upgrade independently
- Prevents the link mismatches like the `sd_cache_params_init` issue

### 2. System GGML Integration (Optional)

For advanced use cases, ofxStableDiffusion can now optionally consume GGML from ofxGgml at build time using `stable-diffusion.cpp`'s built-in `-DSD_USE_SYSTEM_GGML=ON` support.

**Build with system GGML:**

```bash
# Linux/macOS
./scripts/build-stable-diffusion.sh --use-system-ggml --ofxggml-path ../ofxGgml

# Windows
.\scripts\build-stable-diffusion.ps1 -UseSystemGgml -OfxGgmlPath ..\ofxGgml
```

**What this provides:**
- Single GGML binary shared between stable-diffusion and llama/whisper/etc.
- Reduced disk footprint (one set of GGML libs instead of duplicates)
- Consistent GGML version across all addons

**Important requirements:**
- ofxGgml must be built first
- Backend flags must match (CPU/CUDA/Vulkan)
- GGML versions must be compatible
- This is opt-in; default remains standalone for stability

See [docs/NATIVE_BUILD.md](NATIVE_BUILD.md) for complete system GGML build instructions.

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

**In standalone mode (default):**

`ofxStableDiffusion` stages a diffusion runtime surface only:

- public header in `libs/stable-diffusion/include/stable-diffusion.h`
- staged runtime in `libs/stable-diffusion/lib/...`
  - `stable-diffusion.dll`
  - `stable-diffusion.lib`

It does not stage a public `ggml` SDK surface for other addons to consume.

**In system GGML mode (optional):**

`ofxStableDiffusion` links against ofxGgml's GGML and only stages:
- public header in `libs/stable-diffusion/include/stable-diffusion.h`
- staged runtime in `libs/stable-diffusion/lib/...`
  - `stable-diffusion.dll` (linked against ofxGgml's GGML)
  - `stable-diffusion.lib`

GGML itself comes from ofxGgml in this mode.

## Why Addon-Level Integration Is The Default

Even though system GGML is now supported, addon-level integration remains the recommended default for most users:

**Stability:**
- Each addon can upgrade independently without breaking the other
- Version mismatches are isolated
- ABI changes in GGML don't require coordinated updates

**Flexibility:**
- Different GGML versions/patches can coexist
- Backend selection can differ (e.g., CPU diffusion with CUDA language models)
- Build flags and optimizations can be tuned per use case

**Simplicity:**
- No coordination needed between addon maintainers
- Users can update one addon without rebuilding the other
- Fallback is always available if system GGML causes issues

System GGML mode is available for users who:
- Need to minimize disk usage
- Want guaranteed version consistency
- Are comfortable managing GGML compatibility manually
- Have controlled environments where coordinated updates are feasible
