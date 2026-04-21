# Feature Suggestions for ofxStableDiffusion

This document contains suggested features and enhancements for future versions of the ofxStableDiffusion addon.

## High Priority Features

### 1. Advanced Error Handling ✅ IMPLEMENTED (v1.0.0)

**Status**: ✅ **COMPLETED** in version 1.0.0

This feature has been fully implemented with error code enumeration, error categories, error history tracking, and recovery suggestions.

### 2. Model Preloading and Management ✅ IMPLEMENTED (v1.1.0)

**Status**: ✅ **COMPLETED** in version 1.1.0

Implemented via `ofxStableDiffusionModelManager` class with:
- Async model preloading
- Model metadata extraction and validation
- LRU cache eviction
- Model scanning from directories
- Cache size and count limits
- Progress callbacks during loading

**API Added**:
```cpp
ofxStableDiffusionModelManager modelManager;
auto models = modelManager.scanModelsInDirectory("data/models");
std::string errorMsg;
bool success = modelManager.preloadModel(modelInfo, errorMsg);
sd_ctx_t* ctx = modelManager.getModelContext(modelPath, modelInfo, errorMsg);
```

### 3. Generation Queue System ✅ IMPLEMENTED (v1.1.0)

**Status**: ✅ **COMPLETED** in version 1.1.0

Implemented via `ofxStableDiffusionQueue` class with:
- Priority-based request queuing (Low, Normal, High, Critical)
- Request cancellation by ID or tag
- Per-request completion, error, and progress callbacks
- Queue statistics and state management
- Optional queue persistence to file

**API Added**:
```cpp
ofxStableDiffusionQueue queue;
int requestId = queue.addImageRequest(request, ofxStableDiffusionPriority::High, "batch1");
queue.setCompletionCallback(requestId, [](const ofxStableDiffusionResult& result) {
    // Handle completion
});
queue.cancelRequest(requestId);
auto stats = queue.getStats();
```

## Medium Priority Features

### 4. Advanced Progress Reporting ✅ IMPLEMENTED (v1.1.0)

**Status**: ✅ **COMPLETED** in version 1.1.0

Implemented via `ofxStableDiffusionProgressTracker` class with:
- Stage-based progress (Loading, Encoding, Diffusing, Decoding, Upscaling, Finalizing)
- ETA estimation with adaptive algorithm
- Throughput metrics (steps/second)
- Memory usage reporting
- Batch progress support

**API Added**:
```cpp
ofxStableDiffusionProgressTracker tracker;
tracker.reset(totalSteps, totalBatches);
tracker.update(currentStep, currentBatch, elapsedSeconds);
tracker.setPhase(ofxStableDiffusionPhase::Diffusing);
auto progressInfo = tracker.getProgressInfo();
// Access: percentComplete, estimatedTimeRemainingSeconds, stepsPerSecond, etc.
```

### 5. ControlNet Multi-Model Support ✅ IMPLEMENTED (v1.2.0)

**Status**: ✅ **COMPLETED** in version 1.2.0

Implemented via `ofxStableDiffusionControlNet` structure and vector-based API:
- Multiple ControlNet models simultaneously
- Different control types (canny, depth, pose, etc.) via type field
- Per-model strength control
- Backward compatibility with legacy single ControlNet API

**API Added**:
```cpp
struct ofxStableDiffusionControlNet {
    sd_image_t conditionImage{0, 0, 0, nullptr};
    float strength = 0.9f;
    std::string type;  // Optional type hint: "canny", "depth", "pose", etc.

    bool isValid() const;
};

void addControlNet(const ofxStableDiffusionControlNet& controlNet);
void clearControlNets();
std::vector<ofxStableDiffusionControlNet> getControlNets() const;
// ImageRequest includes: std::vector<ofxStableDiffusionControlNet> controlNets;
```

### 6. LoRA Management System 🚧 PARTIALLY IMPLEMENTED (v1.2.0)

**Status**: 🚧 **PARTIALLY COMPLETED** in version 1.2.0

Implemented LoRA discovery functionality:
- LoRA discovery and listing from directory
- Returns name and absolute path pairs
- Supports multiple file formats (.safetensors, .ckpt, .pt, .bin)

**API Added**:
```cpp
std::vector<std::pair<std::string, std::string>> listLoras() const;
```

**Already Available**:
```cpp
void setLoras(const std::vector<ofxStableDiffusionLora>& loras);
std::vector<ofxStableDiffusionLora> getLoras() const;
// ofxStableDiffusionLora includes name, path, and multiplier (weight) fields
```

**Remaining Work**:
- Convenience methods for adding/removing individual LoRAs
- Better weight control documentation

### 7. Image Seed Management ✅ IMPLEMENTED (v1.1.0)

**Status**: ✅ **COMPLETED** in version 1.1.0

Implemented with seed history tracking and deterministic seed generation:
- Seed history tracking (up to 20 recent seeds)
- Deterministic seed from string hash
- Actual seed capture in results

**API Added**:
```cpp
int64_t getLastUsedSeed() const;
const std::vector<int64_t>& getSeedHistory() const;
void clearSeedHistory();
static int64_t hashStringToSeed(const std::string& text);
// Result structure includes actualSeedUsed field
```

### 8. Inpainting and Outpainting ✅ IMPLEMENTED (v1.2.0)

**Status**: ✅ **COMPLETED** in version 1.2.0

Implemented complete inpainting support with validation:
- Dedicated `Inpainting` mode enum
- Mask support (`maskImage` field in request)
- Mask dimension validation (must match input image)
- Proper error handling with descriptive messages

**API**:
```cpp
ofxStableDiffusionImageRequest request;
request.mode = ofxStableDiffusionImageMode::Inpainting;
request.initImage = sourceImage;  // Source image
request.maskImage = maskImage;     // White=inpaint, Black=keep (must match initImage dimensions)
request.prompt = "replace with ocean view";
sd.generate(request);
```

**Remaining Work**:
- Add outpainting with automatic padding
- Add mask preprocessing helpers (blur, feather, dilate)
- Add mask generation utilities

## Low Priority / Future Enhancements

### 9. Real-time Generation Modes

**Suggested Enhancement**:
- Low-latency streaming mode (LCM, Turbo optimized)
- Frame-by-frame refinement
- Interactive parameter adjustment during generation

### 10. Model Quantization Support

**Suggested Enhancement**:
- Support for different quantization levels
- Runtime quantization conversion
- Quality vs. speed tradeoffs

### 11. Prompt Engineering Helpers

**Suggested Enhancement**:
- Prompt templates
- Negative prompt presets
- Prompt token counting and analysis
- Emphasis syntax support

### 12. Batch Processing Utilities

**Suggested Enhancement**:
- Grid generation (X/Y/Z plots)
- Parameter sweeping
- A/B comparison tools
- Batch export with metadata

### 13. Advanced Sampling Options ✅ IMPLEMENTED (v1.2.0)

**Status**: ✅ **COMPLETED** in version 1.2.0

Implemented comprehensive sampling configuration utilities:
- 15+ sampling methods with metadata and descriptions
- 11+ schedulers with use case recommendations
- 7 predefined quality/speed presets
- Sampler/scheduler combination recommendations
- Step count validation and recommendation system
- Name-based lookup (case-insensitive)

**API Added**:
```cpp
// Get all available samplers and schedulers
auto samplers = ofxStableDiffusionSamplingHelpers::getAllSamplers();
auto schedulers = ofxStableDiffusionSamplingHelpers::getAllSchedulers();

// Lookup by name
auto method = ofxStableDiffusionSamplingHelpers::getSamplerByName("DPM++ 2M");
auto sched = ofxStableDiffusionSamplingHelpers::getSchedulerByName("Karras");

// Use presets
auto preset = ofxStableDiffusionSamplingPreset::Quality();  // DPM++ 2M + Karras, 30 steps
auto preset = ofxStableDiffusionSamplingPreset::Fast();     // Euler A + Discrete, 15 steps
auto preset = ofxStableDiffusionSamplingPreset::LCM();      // LCM + LCM scheduler, 4 steps

// Validate and recommend steps
bool valid = ofxStableDiffusionSamplingHelpers::isValidStepCount(method, 20);
int steps = ofxStableDiffusionSamplingHelpers::getRecommendedSteps(method, 0.8f);  // quality level 0-1
```

**Supported Samplers**: Euler, Euler A, Heun, DPM2, DPM++ 2S A, DPM++ 2M, DPM++ 2M v2, iPNDM, iPNDM_v, LCM, DDIM Trailing, TCD, Restart Multistep, Restart 2S, ER-SDE

**Supported Schedulers**: Discrete, Karras, Exponential, AYS, GITS, SGM Uniform, Simple, Smoothstep, KL Optimal, LCM, Bong Tangent

### 14. Textual Inversion Support

**Suggested Enhancement**:
- Textual inversion/embedding loading
- Per-prompt embedding management
- Embedding discovery

### 15. Safety and Content Filtering

**Suggested Enhancement**:
- Optional NSFW filter
- Content classification
- Watermarking support

### 16. Performance Profiling ✅ IMPLEMENTED (v1.2.0)

**Status**: ✅ **COMPLETED** in version 1.2.0

Implemented comprehensive performance profiling system:
- Built-in performance metrics with timing and memory tracking
- Bottleneck identification with configurable thresholds
- Thread-safe operations with std::mutex
- RAII-style scoped timers for automatic instrumentation
- JSON and CSV export formats
- Statistics aggregation across multiple calls

**API Added**:
```cpp
// Enable/disable profiling
sd.setProfilingEnabled(true);

// Get performance data
auto stats = sd.getPerformanceStats();
auto entry = sd.getPerformanceEntry("diffusion");

// Identify bottlenecks (operations taking >10% of total time)
auto bottlenecks = sd.getPerformanceBottlenecks(10.0f);

// Export data
std::string json = sd.exportPerformanceJSON();
std::string csv = sd.exportPerformanceCSV();

// Print summary to console
sd.printPerformanceSummary();

// Reset profiling data
sd.resetProfiling();

// Direct profiler usage
ofxStableDiffusionPerformanceProfiler profiler;
profiler.begin("operation_name");
// ... do work ...
profiler.end("operation_name");

// Or use scoped timers
{
    auto timer = profiler.scopedTimer("scoped_operation");
    // ... automatically timed until scope exit ...
}
```

**Remaining Work**:
- GPU utilization tracking (requires backend-specific integration)

## Integration Features

### 17. Enhanced ofxGgml Integration

**Suggested Enhancement**:
- Automatic CLIP interrogation
- Image-to-prompt reverse engineering
- Style extraction from reference images
- Semantic search in generated image batches

### 18. Animation and Interpolation ✅ **COMPLETED (v1.2.0)**

**Implementation Status**: Fully implemented with comprehensive keyframe and interpolation support.

**Implemented Features**:
- ✅ Prompt keyframe interpolation with smooth transitions
- ✅ Parameter keyframe animation (CFG scale, strength)
- ✅ Seed sequence support for frame variation
- ✅ Multiple interpolation modes (Linear, Smooth, EaseIn, EaseOut, EaseInOut)
- ✅ Helper functions for creating animated video requests
- ✅ Frame-specific parameter getters

**Suggested Future Enhancements**:
- Video-to-video support (requires stable-diffusion.cpp updates)

### 19. Export and Metadata ✅ **COMPLETED (v1.2.0)**

**Implementation Status**: Comprehensive metadata and export features implemented.

**Implemented Features**:
- ✅ Frame-by-frame metadata generation (JSON format)
- ✅ Generation parameter tracking per frame
- ✅ Video parameter export to JSON file
- ✅ Complete generation history tracking

**Suggested Future Enhancements**:
- PNG metadata embedding (generation parameters)
- Reproducible generation from metadata files

### 20. Platform-Specific Optimizations

**Suggested Enhancement**:
- Metal acceleration improvements for macOS
- Mobile platform support (iOS/Android with reduced models)
- WebAssembly build for browser use

## Community Requests

This section is reserved for features requested by addon users. Please submit feature requests via GitHub issues with the `enhancement` label.

---

## Implementation Status Summary

### ✅ Completed Features (v1.0.0 - v1.2.0)
1. ✅ Advanced Error Handling (v1.0.0)
2. ✅ Model Preloading and Management (v1.1.0)
3. ✅ Generation Queue System (v1.1.0)
4. ✅ Advanced Progress Reporting (v1.1.0)
5. ✅ ControlNet Multi-Model Support (v1.2.0)
7. ✅ Image Seed Management (v1.1.0)
8. ✅ Inpainting and Outpainting (v1.2.0)
13. ✅ Advanced Sampling Options (v1.2.0)
16. ✅ Performance Profiling (v1.2.0)
18. ✅ Animation and Interpolation (v1.2.0)
19. ✅ Export and Metadata (v1.2.0)

### 🚧 Partially Completed
6. 🚧 LoRA Management System (v1.2.0 - discovery implemented, convenience methods pending)

### 📋 Planned
- Real-time Generation Modes
- Model Quantization Support
- Prompt Engineering Helpers
- Batch Processing Utilities
- Textual Inversion Support
- Safety and Content Filtering
- Enhanced ofxGgml Integration
- Platform-Specific Optimizations
- Real-time Generation Modes
- And more...

---

## Contributing

Feature suggestions are welcome! Please:
1. Open a GitHub issue with detailed description
2. Explain the use case and benefits
3. Provide example code/API if possible
4. Consider backwards compatibility

Priority is given to features that:
- Have clear use cases
- Align with the addon's design philosophy
- Don't break existing functionality
- Can be implemented cleanly
