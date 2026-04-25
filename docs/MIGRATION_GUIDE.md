# Migration Guide

## Overview

This guide helps you migrate to the latest version of ofxStableDiffusion and adopt modern API patterns.

## From Legacy API to Modern API

### Text-to-Image Generation

**Old (Deprecated):**
```cpp
sd.txt2img(
    prompt,
    negativePrompt,
    clipSkip,
    cfgScale,
    width,
    height,
    sampleMethod,
    sampleSteps,
    seed,
    batchCount,
    nullptr,  // controlCond
    0.9f,     // controlStrength
    0.0f,     // styleStrength
    false,    // normalizeInput
    ""        // inputIdImagesPath
);
```

**New (Recommended):**
```cpp
ofxStableDiffusionImageRequest request;
request.prompt = prompt;
request.negativePrompt = negativePrompt;
request.clipSkip = clipSkip;
request.cfgScale = cfgScale;
request.width = width;
request.height = height;
request.sampleMethod = sampleMethod;
request.sampleSteps = sampleSteps;
request.seed = seed;
request.batchCount = batchCount;

sd.generate(request);
```

**Benefits:**
- Clear, self-documenting code
- Optional parameters with sensible defaults
- Easier to extend with new features
- Type-safe with compile-time checking

### Image-to-Image Generation

**Old:**
```cpp
sd.loadImage(pixels);
sd.img2img(prompt, negativePrompt, /* ... many parameters ... */);
```

**New:**
```cpp
ofxStableDiffusionImageRequest request;
request.mode = ofxStableDiffusionImageMode::ImageToImage;
request.prompt = prompt;
request.negativePrompt = negativePrompt;
request.initImage = /* sd_image_t from pixels */;
request.strength = 0.75f;
// ... other parameters ...

sd.generate(request);
```

### Video Generation

**Old:**
```cpp
sd.img2vid(prompt, negativePrompt, /* ... */);
```

**New:**
```cpp
ofxStableDiffusionVideoRequest request;
request.prompt = prompt;
request.negativePrompt = negativePrompt;
request.initImage = /* starting frame */;
request.frameCount = 24;
request.fps = 8;
// ... other parameters ...

sd.generateVideo(request);
```

## Breaking Changes by Version

### Version 1.3.0

**New:**
- Cancellation support API
- Platform support for macOS/iOS
- Enhanced error handling with `Cancelled` error code

**Breaking Changes:**
- None (backward compatible)

**Deprecations:**
- None

### Version 1.2.0

**New:**
- Multi-ControlNet support
- Inpainting mode
- Performance profiling
- Sampling helpers

**Breaking Changes:**
- `controlCond` parameter replaced with `controlNets` vector
- Old single-ControlNet API still supported but deprecated

**Migration:**
```cpp
// Old
request.controlCond = &controlImage;
request.controlStrength = 0.9f;

// New
ofxStableDiffusionControlNet controlNet;
controlNet.conditionImage = controlImage;
controlNet.strength = 0.9f;
controlNet.type = "canny";
request.controlNets.push_back(controlNet);
```

### Version 1.1.0

**New:**
- Model manager with caching
- Generation queue system
- Progress tracking with ETA
- Seed history

**Breaking Changes:**
- None (backward compatible)

## Migrating to Request-Based API

### Step 1: Update Includes

No changes needed - all APIs are in the same headers.

### Step 2: Replace Function Calls

Use find-and-replace or refactor gradually:

```cpp
// Pattern to find:
sd.txt2img(

// Replace with:
ofxStableDiffusionImageRequest request;
// ... set parameters ...
sd.generate(
```

### Step 3: Set Request Parameters

Map old positional parameters to request fields:

```cpp
// Old parameter order:
// prompt, negativePrompt, clipSkip, cfgScale, width, height,
// sampleMethod, sampleSteps, seed, batchCount

request.prompt = prompt;
request.negativePrompt = negativePrompt;
request.clipSkip = clipSkip;
request.cfgScale = cfgScale;
request.width = width;
request.height = height;
request.sampleMethod = sampleMethod;
request.sampleSteps = sampleSteps;
request.seed = seed;
request.batchCount = batchCount;
```

### Step 4: Test Thoroughly

The new API should produce identical results, but always test:
- Same prompts produce same outputs (with same seed)
- Error handling works correctly
- Performance is unchanged

## Common Migration Patterns

### Pattern 1: Default Parameters

**Old:**
```cpp
// Must specify all parameters, even defaults
sd.txt2img(prompt, "", -1, 7.0f, 512, 512,
           EULER_A_SAMPLE_METHOD, 20, -1, 1,
           nullptr, 0.9f, 0.0f, false, "");
```

**New:**
```cpp
// Only specify what you need
ofxStableDiffusionImageRequest request;
request.prompt = prompt;
request.width = 512;
request.height = 512;
sd.generate(request);
```

### Pattern 2: ControlNet

**Old:**
```cpp
sd_image_t controlImage = /* ... */;
sd.txt2img(/* ... */, &controlImage, 0.9f, /* ... */);
```

**New:**
```cpp
ofxStableDiffusionControlNet controlNet;
controlNet.conditionImage = controlImage;
controlNet.strength = 0.9f;

ofxStableDiffusionImageRequest request;
request.controlNets.push_back(controlNet);
sd.generate(request);
```

### Pattern 3: Multiple ControlNets

**New Capability:**
```cpp
// Add multiple ControlNets with different types
ofxStableDiffusionControlNet canny, depth;

canny.conditionImage = cannyImage;
canny.strength = 0.9f;
canny.type = "canny";

depth.conditionImage = depthImage;
depth.strength = 0.7f;
depth.type = "depth";

request.controlNets = {canny, depth};
sd.generate(request);
```

## Migrating Error Handling

### Old Pattern

**Old:**
```cpp
sd.generate(request);
if (!sd.hasImageResult()) {
    std::string error = sd.getLastError();
    ofLogError() << error;
}
```

### New Pattern (Enhanced)

**New:**
```cpp
sd.generate(request);
ofxStableDiffusionError error = sd.getLastErrorInfo();
if (error.code != ofxStableDiffusionErrorCode::None) {
    ofLogError() << error.message;
    ofLogNotice() << "Suggestion: " << error.suggestion;

    // Handle specific errors
    switch (error.code) {
        case ofxStableDiffusionErrorCode::OutOfMemory:
            // Reduce dimensions or batch count
            break;
        case ofxStableDiffusionErrorCode::Cancelled:
            // User cancelled
            break;
        default:
            break;
    }
}
```

## Using New Features

### Cancellation (v1.3.0+)

```cpp
// Start generation
sd.generate(request);

// In another context (timer, button, etc.):
if (userWantsToCan cel) {
    sd.requestCancellation();
}

// Check result
if (sd.wasCancelled()) {
    ofLogNotice() << "User cancelled generation";
}
```

### Performance Profiling (v1.2.0+)

```cpp
sd.setProfilingEnabled(true);

sd.generate(request);

// Get performance stats
auto stats = sd.getPerformanceStats();
ofLogNotice() << "Total time: " << stats.totalTimeMs << "ms";

// Find bottlenecks
auto bottlenecks = sd.getPerformanceBottlenecks(10.0f);
for (const auto& name : bottlenecks) {
    auto entry = sd.getPerformanceEntry(name);
    ofLogWarning() << name << " took " << entry.avgTimeMs << "ms";
}
```

### Model Caching (v1.1.0+)

```cpp
// Scan and preload models
auto models = sd.scanModels("data/models/");
for (const auto& model : models) {
    std::string error;
    if (sd.preloadModel(model.path, error)) {
        ofLogNotice() << "Preloaded: " << model.name;
    }
}

// Switch between cached models instantly
sd.configureContext(settings1);  // Fast
sd.configureContext(settings2);  // Fast
```

## Compatibility Notes

### Backward Compatibility

- All legacy `txt2img()`, `img2img()`, etc. methods still work
- No breaking changes in v1.3.0
- Plan to use request-based API for new code

### Forward Compatibility

- Request-based API is the future
- New features will be added to request structures
- Legacy methods may be deprecated in future versions

## Getting Help

If you encounter migration issues:

1. Check the [API Reference](API_REFERENCE.md)
2. See [Troubleshooting Guide](TROUBLESHOOTING.md)
3. Look at [examples/](examples/) for patterns
4. Open a GitHub issue with your specific case

## Migration Checklist

- [ ] Update to latest ofxStableDiffusion version
- [ ] Read this migration guide
- [ ] Identify all generation calls in your code
- [ ] Create backup/branch before changes
- [ ] Migrate one file at a time
- [ ] Test each migrated component
- [ ] Update error handling to use new patterns
- [ ] Consider adopting new features (profiling, caching, etc.)
- [ ] Update your documentation
- [ ] Test thoroughly with various inputs

## Timeline

- **Legacy API**: Supported indefinitely, no plans to remove
- **Request API**: Current standard, use for all new code
- **Future**: New features only added to request API
