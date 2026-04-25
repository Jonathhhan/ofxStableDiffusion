# API Reference Guide

## Overview

This guide provides a high-level overview of the ofxStableDiffusion API. For detailed documentation of all classes and methods, see the [generated Doxygen documentation](api/html/index.html).

## Core Classes

### ofxStableDiffusion

The main class for Stable Diffusion generation. Provides methods for:
- Text-to-image generation
- Image-to-image transformation
- Video generation
- Model management
- Performance profiling

**Key Methods:**
- `generate(const ofxStableDiffusionImageRequest& request)` - Generate images
- `generateVideo(const ofxStableDiffusionVideoRequest& request)` - Generate videos
- `configureContext(const ofxStableDiffusionContextSettings& settings)` - Load models
- `requestCancellation()` - Cancel ongoing generation
- `isGenerating()` - Check if generation is in progress

**Thread Safety:** Most methods are NOT thread-safe. Always call from the main thread unless otherwise documented.

### ofxStableDiffusionRealtimeSession

Real-time generation pipeline for interactive applications.

**Key Features:**
- Adaptive step count with progressive refinement
- Low-latency mode for performance
- Frame dropping when busy
- Live parameter updates

**Usage:**
```cpp
ofxStableDiffusionRealtimeSession session;
session.setGenerator(&sd);
session.start(settings);

// In update loop
session.update();
```

### ofxStableDiffusionBatchProcessor

Batch processing utilities for systematic parameter exploration.

**Features:**
- Grid generation (X/Y/Z plots)
- Parameter sweeping
- A/B comparison
- Organized export

### ofxStableDiffusionModelManager

Model caching and preloading system.

**Features:**
- LRU cache with size limits
- Model metadata extraction
- Hot-swapping between models
- Directory scanning

## Request/Response Types

### ofxStableDiffusionImageRequest

Configuration for image generation requests.

**Key Fields:**
- `prompt` - Text prompt
- `negativePrompt` - Negative prompt
- `width`, `height` - Output dimensions
- `sampleSteps` - Diffusion steps
- `cfgScale` - Guidance scale
- `seed` - Random seed (-1 for random)
- `batchCount` - Number of images
- `mode` - Generation mode
- `controlNets` - ControlNet configurations

### ofxStableDiffusionVideoRequest

Configuration for video generation requests.

**Key Fields:**
- Same as ImageRequest, plus:
- `frameCount` - Number of frames
- `fps` - Frames per second
- `endImage` - Target image for morphing
- `controlFrames` - Per-frame control images

### ofxStableDiffusionResult

Generation results container.

**Key Fields:**
- `success` - Generation succeeded
- `error` - Error information
- `images` - Generated image frames
- `video` - Video clip data
- `actualSeedUsed` - Seed value used
- `elapsedMilliseconds` - Generation time

## Context Settings

### ofxStableDiffusionContextSettings

Model and runtime configuration.

**Key Fields:**
- `modelPath` - Path to model file
- `vaePath` - Optional VAE model
- `clipLPath`, `t5xxlPath` - Split model components
- `loraModelDir` - LoRA directory
- `embedDir` - Embeddings directory
- `controlNetPath` - ControlNet model
- `stackedIdEmbedDir` - PhotoMaker embeddings
- `schedule` - Noise scheduler
- `wType` - Weight precision type
- `clipSkip` - CLIP layers to skip
- `vaeDecodeOnly` - VAE decode-only mode
- `vaeTiling` - Enable VAE tiling for large images
- `freeParamsImmediately` - Free memory after generation
- `rngType` - Random number generator type
- `diffusionFlashAttn` - Flash attention
- `cacheMode` - Cache optimization mode

## Error Handling

### Error Codes

- `None` - No error
- `ModelNotFound` - Model file not found
- `ModelCorrupted` - Model file corrupted
- `ModelLoadFailed` - Failed to load model
- `OutOfMemory` - Insufficient memory
- `InvalidDimensions` - Invalid width/height
- `InvalidBatchCount` - Invalid batch count
- `InvalidFrameCount` - Invalid frame count
- `InvalidParameter` - Invalid parameter value
- `MissingInputImage` - Required input image missing
- `GenerationFailed` - Generation failed
- `ThreadBusy` - Another operation in progress
- `UpscaleFailed` - Upscaling failed
- `Cancelled` - Operation cancelled by user
- `Unknown` - Unknown error

### Error Information

```cpp
ofxStableDiffusionError error = sd.getLastErrorInfo();
if (error.code != ofxStableDiffusionErrorCode::None) {
    ofLogError() << error.message;
    ofLogNotice() << "Suggestion: " << error.suggestion;
}
```

## Cancellation Support

New in this release: Support for cancelling long-running operations.

**API:**
- `requestCancellation()` - Request cancellation
- `isCancellationRequested()` - Check if cancellation pending
- `wasCancelled()` - Check if last operation was cancelled

**Notes:**
- Cancellation is checked between diffusion steps
- Operation stops gracefully after current step
- Error code will be `Cancelled` if cancelled

**Example:**
```cpp
sd.generate(request);

// In another thread or timer:
if (userPressedCancel) {
    sd.requestCancellation();
}

// After generation completes:
if (sd.wasCancelled()) {
    ofLogNotice() << "Generation was cancelled";
}
```

## Platform Support

### macOS

Fully supported with Metal backend for optimal performance.

**Requirements:**
- macOS 10.15+
- Xcode with Metal support
- Build stable-diffusion.cpp with Metal backend

### iOS

Basic support available (static linking required).

**Requirements:**
- iOS 13.0+
- Reduced model sizes recommended
- Limited by device memory

### Linux

Fully supported with CUDA, Vulkan, or CPU backends.

### Windows

Fully supported with CUDA, Vulkan, or CPU backends.

## Performance Tips

1. **Use appropriate model size** - Smaller models for real-time, larger for quality
2. **Enable VAE tiling** - For images larger than 1024x1024
3. **Adjust sample steps** - Fewer steps for speed, more for quality
4. **Use LCM/Turbo models** - For real-time applications
5. **Enable profiling** - To identify bottlenecks
6. **Preload models** - Use ModelManager to cache frequently-used models
7. **Batch processing** - Generate multiple images in one call

## Thread Safety

**Main Thread Only:**
- `generate()`, `generateVideo()`
- `configureContext()`
- All model loading methods

**Thread-Safe:**
- `isGenerating()`
- `requestCancellation()`
- `isCancellationRequested()`
- `wasCancelled()`

## See Also

- [Migration Guide](MIGRATION_GUIDE.md) - Upgrading from older versions
- [Troubleshooting Guide](TROUBLESHOOTING.md) - Common issues and solutions
- [Examples](examples/) - Code examples
- [Generated API Docs](api/html/index.html) - Complete API reference

## Generating Documentation

To generate the complete API documentation:

```bash
# Install Doxygen if needed
# Linux: sudo apt-get install doxygen graphviz
# macOS: brew install doxygen graphviz
# Windows: Download from doxygen.org

# Generate documentation
doxygen Doxyfile

# Open documentation
open docs/api/html/index.html
```
