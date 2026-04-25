# Basic Generation Example

This example demonstrates the simplest way to generate images with ofxStableDiffusion.

## Features

- Load a Stable Diffusion model
- Generate images from text prompts
- Display progress during generation
- Save generated images

## Usage

1. Place a Stable Diffusion model (`.safetensors` or `.ckpt`) in `bin/data/models/`
2. Update the model path in `ofApp::setup()` if needed
3. Run the example
4. Press **SPACE** to generate an image
5. Press **S** to save the current image

## Code Walkthrough

### Setup (ofApp::setup)

```cpp
ofxStableDiffusionContextSettings settings;
settings.modelPath = ofToDataPath("models/sd_v1.5.safetensors");
settings.wType = SD_TYPE_F16;
sd.configureContext(settings);
```

This loads the model and configures the generation context.

### Generate (keyPressed)

```cpp
ofxStableDiffusionImageRequest request;
request.prompt = "A serene mountain landscape at sunset";
request.width = 512;
request.height = 512;
request.sampleSteps = 20;
sd.generate(request);
```

Creates a request with desired parameters and starts generation.

### Progress Tracking

```cpp
sd.setProgressCallback([](int step, int steps, float time) {
    float progress = (float)step / (float)steps;
    // Update UI
});
```

Callbacks fire on each diffusion step for progress updates.

### Get Results

```cpp
if (sd.hasImageResult()) {
    auto images = sd.getImages();
    resultImage.setFromPixels(images[0].pixels);
}
```

Results are available after generation completes.

## Next Steps

- See [cancellation_example](../cancellation_example/) for cancelling long operations
- See API Reference for more generation options

## Troubleshooting

**Model won't load?**
- Verify the model path exists
- Check the model format (`.safetensors` recommended)
- See [Troubleshooting Guide](../../docs/TROUBLESHOOTING.md#model-loading-issues)

**Out of memory?**
- Reduce image dimensions (e.g., 512x512)
- Use `SD_TYPE_F16` instead of `SD_TYPE_F32`
- See [Troubleshooting Guide](../../docs/TROUBLESHOOTING.md#memory-issues)

**Generation too slow?**
- Enable Flash Attention: `settings.flashAttn = true;`
- Reduce sample steps: `request.sampleSteps = 15;`
- See [Troubleshooting Guide](../../docs/TROUBLESHOOTING.md#performance-issues)
