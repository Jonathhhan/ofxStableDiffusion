# ofxStableDiffusion

An [openFrameworks](https://openframeworks.cc/) addon that wraps [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), providing text-to-image, image-to-image, image-to-video generation, and ESRGAN upscaling from C++.

![Screenshot 2023-12-01 062447](https://github.com/Jonathhhan/ofxStableDiffusion/assets/41275844/4622905e-1fcb-4693-b2d0-48a464d2a95c)

## Features

- **Text-to-image** — generate images from text prompts
- **Image-to-image** — transform existing images guided by prompts
- **Image-to-video** — generate video frames from a source image
- **ESRGAN upscaling** — 4× super-resolution on generated images
- **ControlNet** support (pose, canny, etc.)
- **TAESD** — lightweight VAE decoder for faster previews
- **PhotoMaker** — identity-preserving generation
- **LoRA** — load LoRA adapter weights
- **Threaded generation** — all heavy work runs off the main thread
- **Progress callback** — monitor diffusion step progress in real time

## Dependencies

| Dependency | Branch / Notes |
|---|---|
| [openFrameworks](https://openframeworks.cc/) | 0.12+ recommended |
| [ofxImGui](https://github.com/jvcleave/ofxImGui/tree/develop) | **develop** branch (required by the example) |

The pre-built `stable-diffusion.cpp` shared library is bundled in `libs/`.

## Setup

1. Clone this repo into your openFrameworks `addons/` folder:
   ```bash
   cd openFrameworks/addons
   git clone https://github.com/Jonathhhan/ofxStableDiffusion.git
   ```
2. Download a compatible model and place it in the example's `bin/data/models/` directory.
   A ready-to-use model folder is available here:
   <https://huggingface.co/Jona0123456789/ofxStableDiffusion/tree/main>
3. Add `ofxStableDiffusion` (and `ofxImGui` for the example) to your project's `addons.make`.

## Quick Start (addon API)

```cpp
#include "ofxStableDiffusion.h"

ofxStableDiffusion sd;

// In setup():
sd.newSdCtx("data/models/sd_turbo.safetensors",
    "", "", "", "", "", "",       // vae, taesd, controlnet, lora, embed, stacked-id
    false, false, false,          // vaeDecodeOnly, vaeTiling, freeParamsImmediately
    8, SD_TYPE_F16,               // threads, weight type
    STD_DEFAULT_RNG, DEFAULT,     // rng, schedule
    false, false, false);         // keepClipOnCpu, keepControlNetCpu, keepVaeOnCpu

// Optional: receive progress updates
sd.setProgressCallback([](int step, int steps, float time) {
    ofLog() << "Step " << step << "/" << steps << " (" << time << "s)";
});

// Generate:
sd.txt2img("a cat in space", "",
    -1, 7.0, 512, 512,           // clipSkip, cfg, w, h
    EULER_A, 20, -1, 1,          // sampler, steps, seed, batch
    nullptr, 0.9, 20.0, true, "");

// In update():
if (sd.isDiffused()) {
    sd_image_t* imgs = sd.returnImages();
    // upload imgs[0].data to a texture …
    sd.setDiffused(false);
}
```

## API Reference

### `ofxStableDiffusion`

| Method | Description |
|---|---|
| `newSdCtx(...)` | Load a model (runs on a background thread). |
| `freeSdCtx()` | Free the loaded model context. |
| `txt2img(...)` | Start text-to-image generation (background thread). |
| `img2img(...)` | Start image-to-image generation (background thread). |
| `img2vid(...)` | Start image-to-video generation (background thread). |
| `newUpscalerCtx(path, threads, wtype)` | Load an ESRGAN upscaler model. |
| `freeUpscalerCtx()` | Free the upscaler context. |
| `upscaleImage(image, factor)` | Upscale a single `sd_image_t`. |
| `convert(in, vae, out, type)` | Convert a model to a different quantisation. |
| `preprocessCanny(...)` | Run Canny edge detection on a buffer. |
| `loadImage(pixels)` | Set the input image from `ofPixels` (by reference — pixels must stay alive). |
| `isDiffused()` | Returns `true` when generation is complete. |
| `setDiffused(bool)` | Reset the diffused flag after consuming results. |
| `returnImages()` | Get the array of generated `sd_image_t` results. |
| `isGenerating()` | Returns `true` while the background thread is running. |
| `setProgressCallback(cb)` | Register a callback for per-step progress. |
| `getSystemInfo()` | Print system / backend info. |
| `getNumPhysicalCores()` | Query physical CPU core count. |
| `typeName(type)` | Get the name string for an `sd_type_t` value. |

### Supported Samplers

`EULER_A`, `EULER`, `HEUN`, `DPM2`, `DPMPP2S_A`, `DPMPP2M`, `DPMPP2Mv2`, `LCM`

### Supported Schedules

`DEFAULT`, `DISCRETE`, `KARRAS`, `AYS`

## Bundled Library Version

The `libs/stable-diffusion/` directory ships a pre-built version of stable-diffusion.cpp.
The upstream project's API has evolved significantly since this version was bundled (the latest
master uses struct-based parameter passing). To update the bindings, rebuild the shared library
from upstream and update the header accordingly.

## License

See the [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) repo for library licensing.
This addon wrapper follows the same MIT license as openFrameworks addons.
