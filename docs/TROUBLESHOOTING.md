# Troubleshooting Guide

## Overview

This guide helps you diagnose and resolve common issues with ofxStableDiffusion. Issues are organized by symptom with step-by-step solutions.

## Quick Diagnostic Decision Tree

```
Generation not working?
├─ Check error code (getLastErrorInfo())
│  ├─ ModelLoadFailed → See "Model Loading Issues"
│  ├─ OutOfMemory → See "Memory Issues"
│  ├─ InvalidDimensions → See "Dimension Issues"
│  ├─ Cancelled → See "Cancellation Issues"
│  └─ Unknown → See "General Debugging"
├─ Slow or hanging? → See "Performance Issues"
├─ Thread safety issues? → See "Threading Issues"
└─ Platform-specific? → See "Platform Issues"
```

## Model Loading Issues

### Symptom: `ModelLoadFailed` Error

**Common Causes:**
1. Invalid model file path
2. Corrupted model file
3. Incompatible model format
4. Missing dependent files (VAE, ControlNet, etc.)

**Solutions:**

#### 1. Verify File Path
```cpp
ofxStableDiffusionContextSettings settings;
settings.modelPath = "data/models/sd_v1.5.safetensors";

// Check if file exists
ofFile modelFile(settings.modelPath);
if (!modelFile.exists()) {
    ofLogError() << "Model file not found: " << settings.modelPath;
    ofLogError() << "Absolute path: " << modelFile.getAbsolutePath();
}
```

**Fix:** Use absolute paths or verify relative paths from working directory:
```cpp
settings.modelPath = ofToDataPath("models/sd_v1.5.safetensors");
```

#### 2. Check Model Format
Supported formats:
- `.safetensors` (recommended)
- `.ckpt` (legacy)
- `.pth` (PyTorch)

**Fix:** Convert unsupported formats using the convert API:
```cpp
sd.convert(
    "input.ckpt",           // input
    "",                     // vae (optional)
    "output.safetensors",   // output
    SD_TYPE_F16             // precision
);
```

#### 3. Verify Model Integrity
```bash
# Check file size (should be > 2GB for SD 1.5, > 5GB for SDXL)
ls -lh data/models/

# Check file permissions
chmod 644 data/models/*.safetensors
```

#### 4. Check Dependent Files
For models requiring VAE, ControlNet, or other components:

```cpp
settings.vaePath = "data/models/vae.safetensors";
settings.controlNetPath = "data/models/control_canny.safetensors";

// Verify all paths
std::vector<std::string> paths = {
    settings.modelPath,
    settings.vaePath,
    settings.controlNetPath
};

for (const auto& path : paths) {
    if (!path.empty() && !ofFile::doesFileExist(path)) {
        ofLogError() << "Missing file: " << path;
    }
}
```

### Symptom: Model Loads But Generation Fails

**Causes:**
- Model family mismatch
- Incorrect wType for model precision
- Missing required components (CLIP, VAE)

**Solution:**

```cpp
// Get model info first
auto modelInfo = sd.getModelInfo("data/models/sd_v1.5.safetensors");
ofLogNotice() << "Model family: " << (int)modelInfo.family;
ofLogNotice() << "Recommended wType: " << (int)modelInfo.recommendedWType;

// Apply recommended settings
settings.modelPath = modelInfo.path;
settings.wType = modelInfo.recommendedWType;

// Check capabilities
auto caps = sd.getCapabilities();
if (caps.modelFamily == ofxStableDiffusionModelFamily::Unknown) {
    ofLogWarning() << "Unknown model family - may not work correctly";
}
```

## Memory Issues

### Symptom: `OutOfMemory` Error

**Common Causes:**
1. Image dimensions too large
2. Batch count too high
3. Model too large for available VRAM/RAM
4. Memory fragmentation

**Solutions:**

#### 1. Reduce Image Dimensions
```cpp
// VRAM usage scales with pixel count
// SD 1.5: 512x512 = ~2GB, 768x768 = ~4GB, 1024x1024 = ~8GB
// SDXL: 1024x1024 = ~6GB, 1536x1536 = ~12GB

ofxStableDiffusionImageRequest request;
request.width = 512;   // Reduce from 1024
request.height = 512;  // Reduce from 1024
```

**Reference VRAM Requirements:**

| Model | Resolution | Batch=1 | Batch=4 |
|-------|-----------|---------|---------|
| SD 1.5 | 512x512 | ~2GB | ~6GB |
| SD 1.5 | 768x768 | ~4GB | ~12GB |
| SDXL | 1024x1024 | ~6GB | ~18GB |
| SDXL | 1536x1536 | ~12GB | ~36GB |

#### 2. Reduce Batch Count
```cpp
// Process in smaller batches
request.batchCount = 1;  // Down from 4

// Or process serially
for (int i = 0; i < 4; i++) {
    request.batchCount = 1;
    request.seed = baseSeed + i;
    sd.generate(request);
    // Process result before next generation
}
```

#### 3. Use Lower Precision
```cpp
settings.wType = SD_TYPE_F16;  // Instead of SD_TYPE_F32
// Reduces memory by ~50% with minimal quality loss
```

#### 4. Offload to CPU
```cpp
settings.keepVaeOnCpu = true;           // Offload VAE
settings.keepClipOnCpu = true;          // Offload CLIP
settings.keepControlNetCpu = true;      // Offload ControlNet
settings.offloadParamsToCpu = true;     // Offload inactive params
```

#### 5. Enable Tiling (for large images)
```cpp
settings.vaeTiling = true;
// Processes image in tiles, reduces peak memory
```

#### 6. Free Resources Immediately
```cpp
settings.freeParamsImmediately = true;
// Aggressively frees unused model parameters
```

#### 7. Clear Model Cache
```cpp
sd.clearModelCache();
// Frees cached models if switching between models
```

### Symptom: Memory Leak

**Check for:**
```cpp
// Are you holding references to results?
std::vector<ofxStableDiffusionImageFrame> savedImages;
// This holds full pixel data in memory!

// Solution: Extract what you need, then clear
for (auto& frame : sd.getImages()) {
    // Save or process
    frame.pixels.saveImage("output.png");
}
// Result is cleared on next generation
```

## Dimension Issues

### Symptom: `InvalidDimensions` Error

**Causes:**
- Dimensions not multiples of 64
- Dimensions too small or too large
- Aspect ratio incompatible with model

**Solutions:**

#### 1. Use 64-pixel Multiples
```cpp
// Wrong
request.width = 500;   // Not divisible by 64
request.height = 700;  // Not divisible by 64

// Correct
request.width = 512;   // 512 = 64 * 8 ✓
request.height = 704;  // 704 = 64 * 11 ✓
```

#### 2. Check Model Limits

**SD 1.5/2.1:**
- Trained on: 512x512
- Recommended: 512x512 to 768x768
- Max practical: 1024x1024

**SDXL:**
- Trained on: 1024x1024
- Recommended: 1024x1024 to 1536x1536
- Max practical: 2048x2048

```cpp
// Helper: Round to nearest 64
auto roundTo64 = [](int value) {
    return ((value + 31) / 64) * 64;
};

request.width = roundTo64(desiredWidth);
request.height = roundTo64(desiredHeight);
```

#### 3. Validate Before Generation
```cpp
bool validateDimensions(int width, int height) {
    if (width % 64 != 0 || height % 64 != 0) {
        ofLogError() << "Dimensions must be multiples of 64";
        return false;
    }
    if (width < 256 || height < 256) {
        ofLogError() << "Dimensions too small (min 256)";
        return false;
    }
    if (width > 2048 || height > 2048) {
        ofLogWarning() << "Large dimensions may cause OOM";
    }
    return true;
}
```

## Performance Issues

### Symptom: Generation Very Slow

**Diagnostic Steps:**

#### 1. Enable Profiling
```cpp
sd.setProfilingEnabled(true);
sd.generate(request);

auto stats = sd.getPerformanceStats();
ofLogNotice() << "Total time: " << stats.totalTimeMs << "ms";
ofLogNotice() << "Samples/sec: " << stats.samplesPerSecond;

// Find bottlenecks (>10% of total time)
auto bottlenecks = sd.getPerformanceBottlenecks(10.0f);
for (const auto& name : bottlenecks) {
    auto entry = sd.getPerformanceEntry(name);
    ofLogWarning() << name << ": "
                   << entry.avgTimeMs << "ms avg, "
                   << entry.percentage << "% of total";
}
```

#### 2. Common Bottlenecks & Fixes

**Bottleneck: VAE Decode**
```cpp
// Fix 1: Keep VAE on GPU
settings.keepVaeOnCpu = false;

// Fix 2: Use TAESD (tiny VAE, 10x faster)
settings.taesdPath = "data/models/taesd.safetensors";
```

**Bottleneck: CLIP Text Encoding**
```cpp
// Fix: Keep CLIP on GPU
settings.keepClipOnCpu = false;
```

**Bottleneck: Sampling Steps**
```cpp
// Reduce steps for faster generation
request.sampleSteps = 20;  // Down from 50
// Quality difference minimal for most samplers above 20 steps
```

**Bottleneck: High Resolution**
```cpp
// Generate at lower res, then upscale
request.width = 512;
request.height = 512;
// ... generate ...

// Then upscale with ESRGAN
sd.setUpscalerSettings(upscalerSettings);
auto upscaled = sd.upscaleImage(image, 2);
```

#### 3. Optimize Settings

```cpp
// Enable Flash Attention (requires compatible GPU)
settings.flashAttn = true;
settings.diffusionFlashAttn = true;

// Enable memory mapping (faster model loading)
settings.enableMmap = true;

// Use optimal thread count
settings.nThreads = sd.getNumPhysicalCores();

// Choose fast sampler
request.sampleMethod = EULER_A_SAMPLE_METHOD;  // Fast
// Avoid: DDIM (slow), PLMS (slow)
```

#### 4. Platform-Specific

**macOS/iOS:**
```cpp
// Metal may be slower than CPU for some operations
// Try CPU-only for smaller models
settings.nThreads = 8;
settings.keepVaeOnCpu = true;
```

**Linux/Windows with CUDA:**
```cpp
// Use CUDA RNG for better performance
settings.rngType = CUDA_RNG;
```

### Symptom: Model Loading Slow

**Solutions:**

#### 1. Preload Models
```cpp
// At startup
std::string error;
if (sd.preloadModel("data/models/sd_v1.5.safetensors", error)) {
    ofLogNotice() << "Model cached";
}

// Later switches are instant
sd.configureContext(settings1);  // Fast
sd.configureContext(settings2);  // Fast
```

#### 2. Enable mmap
```cpp
settings.enableMmap = true;
// Models load faster and use less RAM
```

## Threading Issues

### Symptom: Crashes or Data Races

**Causes:**
- Calling non-thread-safe methods during generation
- Accessing results while generation in progress
- Multiple simultaneous generations

**Thread-Safe Methods:**
```cpp
// Safe to call anytime from any thread:
bool generating = sd.isGenerating();
bool busy = sd.isBusy();
sd.requestCancellation();
bool cancelRequested = sd.isCancellationRequested();
bool cancelled = sd.wasCancelled();
```

**Not Thread-Safe (main thread only):**
```cpp
sd.generate(request);              // Start generation
sd.getImages();                    // Get results
sd.getLastError();                 // Get error
sd.configureContext(settings);     // Change settings
```

**Safe Pattern:**
```cpp
// Main thread: start generation
sd.generate(request);

// Any thread: monitor progress
void update() {
    if (sd.isGenerating()) {
        // Show spinner
    } else {
        // Generation done, fetch results on main thread
        auto images = sd.getImages();
    }
}

// UI thread: cancel button
void onCancelButton() {
    if (sd.isGenerating()) {
        sd.requestCancellation();
    }
}
```

**Wrong Pattern:**
```cpp
// ❌ WRONG: Accessing results from background thread
std::thread([&sd]() {
    sd.generate(request);
    auto images = sd.getImages();  // NOT THREAD-SAFE!
}).detach();
```

## Cancellation Issues

### Symptom: Cancellation Doesn't Work

**Check:**

1. **Are you checking the flag?**
```cpp
// The cancellation flag must be checked by the native library
// If using custom integration, check periodically:
while (generating) {
    if (sd.isCancellationRequested()) {
        break;
    }
    // ... generation step ...
}
```

2. **Timing expectations:**
```cpp
sd.requestCancellation();
// Cancellation happens after current step completes
// May take 1-5 seconds depending on step complexity
while (sd.isGenerating()) {
    ofSleepMillis(100);  // Wait for graceful stop
}
```

3. **Check result:**
```cpp
if (sd.wasCancelled()) {
    ofLogNotice() << "Generation cancelled";
} else {
    auto error = sd.getLastErrorInfo();
    if (error.code == ofxStableDiffusionErrorCode::Cancelled) {
        ofLogNotice() << "Cancelled via error code";
    }
}
```

## Platform-Specific Issues

### macOS

**Issue: "Library not loaded" error**
```
dyld: Library not loaded: @rpath/libstable-diffusion.dylib
```

**Fix:**
```bash
# Check rpath
otool -l libs/stable-diffusion/lib/osx/libstable-diffusion.dylib | grep RPATH

# Set correct rpath
install_name_tool -add_rpath @loader_path/../../../addons/ofxStableDiffusion/libs/stable-diffusion/lib/osx bin/myApp.app/Contents/MacOS/myApp
```

**Issue: Slow on Metal**
- Metal backend may be slower for some operations
- Try CPU-only mode for testing
- Update to latest stable-diffusion.cpp

### iOS

**Issue: Build fails with dylib error**
- iOS requires static linking
- Ensure `addon_config.mk` uses `.a` not `.dylib`:
```makefile
ios:
    ADDON_LIBS += libs/stable-diffusion/lib/ios/libstable-diffusion.a
```

**Issue: App crashes on device but not simulator**
- Check code signing
- Verify all libraries are ARM64
- Check memory limits (devices have less RAM than Mac)

### Linux

**Issue: CUDA not found**
```cpp
// Verify CUDA build
auto info = sd.getSystemInfo();
ofLogNotice() << info;  // Should show CUDA version

// Fallback to CPU
settings.nThreads = 8;
settings.rngType = PHILOX_RNG;  // CPU RNG
```

### Windows

**Issue: DLL not found**
```
The code execution cannot proceed because stable-diffusion.dll was not found.
```

**Fix:**
- Copy DLL to `bin/` directory
- Or add to PATH
- Or use `SetDllDirectory()` at startup

## ControlNet Issues

### Symptom: ControlNet Has No Effect

**Check:**

1. **Model compatibility:**
```cpp
auto caps = sd.getCapabilities();
if (!caps.supportsControlNet) {
    ofLogError() << "Model doesn't support ControlNet";
}
```

2. **Strength too low:**
```cpp
controlNet.strength = 0.9f;  // Not 0.1f
```

3. **Condition image format:**
```cpp
// Ensure proper preprocessing
controlNet.conditionImage = preprocessedImage;
// e.g., Canny edges for canny ControlNet
```

4. **ControlNet path:**
```cpp
settings.controlNetPath = "data/models/control_canny.safetensors";
// Verify file exists and matches model version (SD 1.5 vs SDXL)
```

## Video Generation Issues

### Symptom: `VideoGenerationFailed` Error

**Check:**

1. **Model supports video:**
```cpp
auto caps = sd.getCapabilities();
if (!caps.supportsImg2Vid) {
    ofLogError() << "Model doesn't support video generation";
    // Requires SVD or similar video model
}
```

2. **Frame count limits:**
```cpp
request.frameCount = 24;  // Keep reasonable (≤30)
// More frames = much more memory
```

3. **Init image provided:**
```cpp
request.initImage = /* valid sd_image_t */;
// Video generation requires starting frame
```

## LoRA Issues

### Symptom: LoRA Has No Effect

**Check:**

1. **Model compatibility:**
```cpp
// LoRA must match model (SD 1.5 LoRA won't work with SDXL)
auto lora = ofxStableDiffusionLora();
lora.path = "data/models/lora_sd15.safetensors";
lora.strength = 0.8f;

// List available LoRAs
auto available = sd.listLoras();
```

2. **Strength:**
```cpp
lora.strength = 0.8f;  // Typical range: 0.5-1.0
// Too low: no effect
// Too high: overpowering, artifacts
```

3. **Apply mode:**
```cpp
settings.loraApplyMode = LORA_APPLY_AUTO;  // Let library decide
// Or LORA_APPLY_UNET_ONLY for faster but less accurate
```

## Debugging Tools

### Enable Detailed Logging

```cpp
// Enable native library logging
sd.setNativeLoggingEnabled(true);
sd.setNativeLogLevel(SD_LOG_DEBUG);

// Enable openFrameworks logging
ofSetLogLevel(OF_LOG_VERBOSE);
```

### Validate Configuration

```cpp
void validateConfig() {
    // Check file paths
    auto checkFile = [](const std::string& path, const std::string& name) {
        if (!path.empty() && !ofFile::doesFileExist(path)) {
            ofLogError() << name << " not found: " << path;
            return false;
        }
        return true;
    };

    auto settings = sd.getContextSettings();
    checkFile(settings.modelPath, "Model");
    checkFile(settings.vaePath, "VAE");
    checkFile(settings.controlNetPath, "ControlNet");

    // Check memory
    auto caps = sd.getCapabilities();
    ofLogNotice() << "Model family: " << (int)caps.modelFamily;
    ofLogNotice() << "Supports ControlNet: " << caps.supportsControlNet;
    ofLogNotice() << "Supports video: " << caps.supportsImg2Vid;

    // Check dimensions
    auto request = /* your request */;
    if (request.width % 64 != 0 || request.height % 64 != 0) {
        ofLogWarning() << "Dimensions not multiples of 64";
    }

    // Estimate VRAM
    int64_t pixelCount = request.width * request.height;
    int64_t estimatedVRAM = pixelCount * request.batchCount * 4;  // Rough estimate
    ofLogNotice() << "Estimated VRAM: ~" << (estimatedVRAM / 1024 / 1024) << " MB";
}
```

### Error History

```cpp
// Check recent errors
auto errors = sd.getErrorHistory();
for (const auto& err : errors) {
    ofLogNotice() << "[" << (int)err.code << "] "
                  << err.message << " | " << err.suggestion;
}
```

### Performance Analysis

```cpp
sd.setProfilingEnabled(true);
sd.generate(request);

// Detailed analysis
sd.printPerformanceSummary();

// Export for analysis
std::string json = sd.exportPerformanceJSON();
ofSaveJson("performance.json", ofJson::parse(json));
```

## Getting More Help

If you're still stuck:

1. **Check error code and message:**
```cpp
auto error = sd.getLastErrorInfo();
ofLogError() << "Code: " << (int)error.code;
ofLogError() << "Message: " << error.message;
ofLogError() << "Suggestion: " << error.suggestion;
```

2. **Gather diagnostic info:**
```cpp
ofLogNotice() << "System: " << sd.getSystemInfo();
ofLogNotice() << "Capabilities: " << (int)sd.getCapabilities().modelFamily;
ofLogNotice() << "Last seed: " << sd.getLastUsedSeed();
```

3. **Check the docs:**
- [API Reference](API_REFERENCE.md) - Complete API documentation
- [Migration Guide](MIGRATION_GUIDE.md) - Upgrading from old API
- [Examples](../examples/) - Working code samples

4. **Ask for help:**
- Include error codes and messages
- Provide model info (family, size, format)
- Share relevant settings and code
- Mention platform (macOS/iOS/Linux/Windows)
- Include system info from `getSystemInfo()`

## Common Error Codes Reference

| Code | Meaning | Common Cause | First Step |
|------|---------|--------------|------------|
| `None` | Success | - | - |
| `ModelLoadFailed` | Model file issue | Wrong path, corrupted file | Verify `modelPath` exists |
| `ContextNotInitialized` | No model loaded | Forgot `configureContext()` | Call `configureContext()` first |
| `OutOfMemory` | Insufficient VRAM/RAM | Image too large, batch too big | Reduce dimensions or batch |
| `InvalidDimensions` | Bad width/height | Not multiple of 64 | Round to 64 |
| `InvalidParameters` | Bad settings | Out-of-range value | Check all request fields |
| `GenerationFailed` | Generation error | Various | Check logs, reduce complexity |
| `UpscaleFailed` | Upscale error | Missing ESRGAN, bad input | Verify `esrganPath` |
| `Cancelled` | User cancelled | `requestCancellation()` called | Expected behavior |
| `Unknown` | Unexpected error | Various | Enable debug logging |

## Performance Expectations

**Typical Generation Times (SD 1.5, 512x512, 20 steps):**

| Hardware | Time | Notes |
|----------|------|-------|
| RTX 4090 | 1-2s | Optimal |
| RTX 3080 | 3-5s | Good |
| RTX 2060 | 8-12s | Acceptable |
| M1 Max | 15-25s | CPU fallback slower |
| CPU (12-core) | 60-120s | Very slow |

If your times are significantly worse, see "Performance Issues" section.
