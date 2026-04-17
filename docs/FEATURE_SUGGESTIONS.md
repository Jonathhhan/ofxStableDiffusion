# Feature Suggestions for ofxStableDiffusion

This document contains suggested features and enhancements for future versions of the ofxStableDiffusion addon.

## High Priority Features

### 1. Advanced Error Handling

**Current State**: String-based error messages via `getLastError()`

**Suggested Enhancement**:
- Add error code enumeration for programmatic error handling
- Implement error categories (model loading, generation, validation, memory)
- Support error history/stack for debugging complex workflows
- Add error recovery suggestions

**Example API**:
```cpp
enum class ofxStableDiffusionErrorCode {
    None = 0,
    ModelNotFound,
    ModelCorrupted,
    OutOfMemory,
    InvalidDimensions,
    InvalidBatchCount,
    MissingInputImage,
    GenerationFailed,
    ThreadBusy
};

struct ofxStableDiffusionError {
    ofxStableDiffusionErrorCode code;
    std::string message;
    std::string suggestion;
    uint64_t timestamp;
};

std::vector<ofxStableDiffusionError> getErrorHistory() const;
ofxStableDiffusionErrorCode getLastErrorCode() const;
```

### 2. Model Preloading and Management

**Current State**: Model loads synchronously on first generation

**Suggested Enhancement**:
- Async model preloading
- Multiple model management (switch between models without reloading)
- Model info/metadata query
- Model validation before loading

**Example API**:
```cpp
struct ofxStableDiffusionModelInfo {
    std::string path;
    std::string type; // SD1.5, SD2.1, SDXL, SD-Turbo, etc.
    uint64_t sizeBytes;
    int width, height; // default dimensions
    bool isLoaded;
    bool isValid;
};

void preloadModel(const std::string& modelPath, std::function<void(bool)> callback);
ofxStableDiffusionModelInfo queryModelInfo(const std::string& path);
std::vector<std::string> getLoadedModels() const;
void switchModel(const std::string& modelPath);
```

### 3. Generation Queue System

**Current State**: Only one generation at a time, subsequent calls ignored

**Suggested Enhancement**:
- Queue multiple generation requests
- Priority-based scheduling
- Cancel pending requests
- Batch processing optimization

**Example API**:
```cpp
struct ofxStableDiffusionQueueEntry {
    uint64_t id;
    ofxStableDiffusionImageRequest request;
    int priority;
    float progress;
};

uint64_t queueGeneration(const ofxStableDiffusionImageRequest& request, int priority = 0);
void cancelQueuedGeneration(uint64_t id);
std::vector<ofxStableDiffusionQueueEntry> getQueue() const;
void setMaxConcurrentGenerations(int count);
```

## Medium Priority Features

### 4. Advanced Progress Reporting

**Current State**: Simple step/steps callback

**Suggested Enhancement**:
- Stage-based progress (loading, encoding, diffusion, decoding)
- ETA estimation
- Throughput metrics (it/s)
- Memory usage reporting

**Example API**:
```cpp
struct ofxStableDiffusionProgress {
    enum Stage { Loading, Encoding, Diffusion, Decoding, Upscaling };
    Stage currentStage;
    int currentStep;
    int totalSteps;
    float etaSeconds;
    float iterationsPerSecond;
    uint64_t memoryUsageBytes;
    uint64_t memoryPeakBytes;
};

void setProgressCallback(std::function<void(const ofxStableDiffusionProgress&)> cb);
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

### 7. Image Seed Management

**Current State**: Basic seed support via int64

**Suggested Enhancement**:
- Seed variations (subseed, subseed_strength)
- Seed travel/interpolation for video
- Deterministic seed from string hash
- Seed history tracking

**Example API**:
```cpp
struct ofxStableDiffusionSeedConfig {
    int64_t seed;
    int64_t subseed;
    float subseedStrength;
};

int64_t hashStringToSeed(const std::string& text);
std::vector<int64_t> getSeedHistory() const;
int64_t getLastUsedSeed() const;
```

### 8. Inpainting and Outpainting

**Current State**: Not directly supported

**Suggested Enhancement**:
- Dedicated inpainting mode
- Mask support (grayscale or alpha channel)
- Outpainting with automatic padding
- Feathering/blur control for mask edges

**Example API**:
```cpp
struct ofxStableDiffusionInpaintRequest {
    sd_image_t sourceImage;
    sd_image_t maskImage; // white = inpaint, black = keep
    std::string prompt;
    float maskBlur;
    int padding; // for outpainting
    // ... other common parameters
};

void generateInpaint(const ofxStableDiffusionInpaintRequest& request);
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
