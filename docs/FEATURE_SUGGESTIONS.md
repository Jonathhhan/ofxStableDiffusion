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

### 5. ControlNet Multi-Model Support

**Current State**: Single ControlNet model

**Suggested Enhancement**:
- Multiple ControlNet models simultaneously
- Different control types (canny, depth, pose, etc.)
- Per-model strength control
- Auto-preprocessing for common types

**Example API**:
```cpp
struct ofxStableDiffusionControlNet {
    std::string modelPath;
    std::string type; // canny, depth, pose, seg, etc.
    sd_image_t conditionImage;
    float strength;
    bool autoPreprocess;
};

void addControlNet(const ofxStableDiffusionControlNet& controlNet);
void clearControlNets();
std::vector<std::string> getSupportedControlNetTypes() const;
```

### 6. LoRA Management System

**Current State**: LoRA directory specified, but no fine control

**Suggested Enhancement**:
- Per-generation LoRA selection
- LoRA strength/weight control
- Multiple LoRAs with different weights
- LoRA discovery and listing

**Example API**:
```cpp
struct ofxStableDiffusionLoRA {
    std::string name;
    std::string path;
    float weight; // -2.0 to 2.0
};

void addLoRA(const std::string& name, float weight);
void removeLoRA(const std::string& name);
std::vector<std::string> discoverLoRAs(const std::string& directory);
```

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

### 8. Inpainting and Outpainting 🚧 IN PROGRESS (v1.1.0)

**Status**: 🚧 **PARTIALLY COMPLETED** in version 1.1.0

Infrastructure added for inpainting mode:
- Dedicated `Inpainting` mode enum
- Mask support (`maskImage` field in request)
- Mask blur control (`maskBlur` parameter)

**Remaining Work**:
- Implement inpainting generation logic in thread
- Add mask preprocessing helpers
- Add outpainting with automatic padding
- Add feathering/blur control for mask edges

**Current API**:
```cpp
ofxStableDiffusionImageRequest request;
request.mode = ofxStableDiffusionImageMode::Inpainting;
request.initImage = sourceImage;  // Source image
request.maskImage = maskImage;     // White=inpaint, Black=keep
request.maskBlur = 4.0f;          // Edge softening
request.prompt = "replace with ocean view";
sd.generate(request);
```

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

### 13. Advanced Sampling Options

**Suggested Enhancement**:
- Custom sampling schedules
- Karras sigma support
- CFG rescale
- SMEA/DYN support for SDXL

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

### 16. Performance Profiling

**Suggested Enhancement**:
- Built-in performance metrics
- Bottleneck identification
- Memory profiling
- GPU utilization tracking

## Integration Features

### 17. Enhanced ofxGgml Integration

**Suggested Enhancement**:
- Automatic CLIP interrogation
- Image-to-prompt reverse engineering
- Style extraction from reference images
- Semantic search in generated image batches

### 18. Animation and Interpolation

**Suggested Enhancement**:
- Keyframe-based animation
- Prompt interpolation
- Smooth parameter transitions
- Video-to-video

### 19. Export and Metadata

**Suggested Enhancement**:
- PNG metadata embedding (generation parameters)
- JSON export of settings
- Reproducible generation from metadata

### 20. Platform-Specific Optimizations

**Suggested Enhancement**:
- Metal acceleration improvements for macOS
- Mobile platform support (iOS/Android with reduced models)
- WebAssembly build for browser use

## Community Requests

This section is reserved for features requested by addon users. Please submit feature requests via GitHub issues with the `enhancement` label.

---

## Implementation Status Summary

### ✅ Completed Features (v1.0.0 - v1.1.0)
1. ✅ Advanced Error Handling (v1.0.0)
2. ✅ Model Preloading and Management (v1.1.0)
3. ✅ Generation Queue System (v1.1.0)
4. ✅ Advanced Progress Reporting (v1.1.0)
7. ✅ Image Seed Management (v1.1.0)

### 🚧 In Progress
8. 🚧 Inpainting and Outpainting (v1.1.0 - partial)

### 📋 Planned
- ControlNet Multi-Model Support
- LoRA Management System
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
