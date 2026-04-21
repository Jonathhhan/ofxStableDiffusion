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

### 9. Real-time Generation Modes ✅ IMPLEMENTED (v1.3.0)

**Status**: ✅ **COMPLETED** in version 1.3.0

**Overview**:
Real-time generation modes enable low-latency, interactive image generation workflows optimized for live performance, VJ applications, and interactive installations. This feature would leverage fast sampling methods (LCM, Turbo, TCD) and streaming inference patterns to achieve sub-second generation times.

**Suggested Features**:
- **Low-Latency Streaming Mode**: Optimized pipeline for LCM/Turbo models with minimal overhead
  - Target: <1 second for 512x512 generation on modern GPUs
  - Streaming inference with progressive quality improvement
  - Frame-by-frame refinement for real-time parameter tweaking

- **Interactive Parameter Adjustment**: Live parameter updates during generation
  - Real-time CFG scale, strength, and prompt adjustments
  - Smooth parameter interpolation between frames
  - Live preview with progressive refinement

- **Performance Optimizations**:
  - Model warmup and caching to eliminate first-run latency
  - Optimized memory management for continuous generation
  - GPU/CPU pipelining to maximize throughput
  - Batch processing for multiple real-time streams

**Potential API**:
```cpp
// Configure real-time mode
ofxStableDiffusionRealtimeSettings rtSettings;
rtSettings.mode = ofxStableDiffusionRealtimeMode::Streaming;
rtSettings.targetLatencyMs = 500;  // Target generation time
rtSettings.enableWarmup = true;
rtSettings.maxQueueDepth = 2;  // Limit pending requests
sd.configureRealtime(rtSettings);

// Start real-time generation session
sd.startRealtimeSession();

// Submit requests with minimal latency
ofxStableDiffusionImageRequest request;
request.prompt = "live performance visuals";
request.width = 512;
request.height = 512;
request.sampleSteps = 4;  // LCM optimized
request.samplerMethod = LCM_SAMPLE_METHOD;
sd.generateRealtime(request);  // Non-blocking, returns immediately

// Update parameters on the fly
sd.updateRealtimePrompt("evolving abstract patterns");
sd.updateRealtimeCfgScale(1.5f);

// Stop real-time session
sd.stopRealtimeSession();
```

**Integration Points**:
- Leverages existing queue system (Feature #3) for request management
- Uses performance profiler (Feature #16) for latency monitoring
- Compatible with existing progress tracking (Feature #4)
- Works with model quantization (Feature #10) for faster inference

**Hardware Requirements**:
- Recommended: GPU with 8GB+ VRAM
- LCM/Turbo models for optimal performance
- CPU fallback with reduced quality/resolution

**Use Cases**:
- Live VJ performances and installations
- Interactive art applications
- Real-time style transfer
- Live video filtering and effects
- Interactive character generation for games

### 10. Model Quantization Support ✅ IMPLEMENTED (v1.3.0)

**Status**: ✅ **COMPLETED** in version 1.3.0

**Overview**:
Model quantization support would enable using quantized Stable Diffusion models (Q4_0, Q5_0, Q8_0, etc.) for reduced memory usage and faster inference on resource-constrained hardware. This feature would complement real-time generation modes by enabling deployment on lower-end GPUs and embedded systems.

**Suggested Features**:
- **Multiple Quantization Levels**:
  - Q4_0, Q4_1: Ultra-compressed (4-bit), ~75% memory reduction
  - Q5_0, Q5_1: Balanced (5-bit), ~65% memory reduction
  - Q8_0: High quality (8-bit), ~50% memory reduction
  - F16: Half precision (16-bit), ~50% memory reduction

- **Runtime Quantization**:
  - On-the-fly model quantization from full precision
  - Quantization presets (Ultra Fast, Balanced, High Quality)
  - Memory usage estimation before loading

- **Quality vs. Performance Tradeoffs**:
  - Quality metrics and comparison tools
  - Automatic quantization level selection based on available VRAM
  - Per-layer quantization for optimal quality/speed balance

**Potential API**:
```cpp
// Load quantized model
ofxStableDiffusionContextSettings context;
context.modelPath = "data/models/sd_turbo_q4_0.safetensors";
context.quantizationLevel = ofxStableDiffusionQuantization::Q4_0;
context.autoQuantize = false;  // Use pre-quantized model
sd.configureContext(context);

// Auto-quantize on load
context.modelPath = "data/models/sd_turbo.safetensors";
context.autoQuantize = true;
context.targetQuantization = ofxStableDiffusionQuantization::Q5_0;
sd.configureContext(context);

// Query quantization info
auto quantInfo = sd.getQuantizationInfo();
ofLogNotice() << "Current quantization: " << quantInfo.level;
ofLogNotice() << "Memory saved: " << quantInfo.memoryReductionPercent << "%";
ofLogNotice() << "Quality estimate: " << quantInfo.qualityScore;

// List available quantization levels
auto levels = ofxStableDiffusionQuantizationHelpers::getAvailableLevels();
for (const auto& level : levels) {
    ofLogNotice() << level.name << ": " << level.description;
}
```

**Integration Points**:
- Essential for real-time generation modes (Feature #9)
- Reduces memory requirements for queue system (Feature #3)
- Enables batch processing on limited hardware (Feature #12)
- Compatible with existing model manager (Feature #2)

**Performance Impact**:
- Memory: 50-75% reduction depending on quantization level
- Speed: 1.2-2x faster inference on most hardware
- Quality: Minimal visual difference for Q5_0 and higher

**Use Cases**:
- Running models on consumer GPUs (4-6GB VRAM)
- Embedded and mobile deployment
- Batch processing with limited memory
- Real-time generation on mid-range hardware

### 11. Prompt Engineering Helpers ✅ IMPLEMENTED (v1.3.0)

**Status**: ✅ **COMPLETED** in version 1.3.0

**Overview**:
Prompt engineering helpers would provide utilities for crafting, analyzing, and managing prompts to achieve better generation results. This includes templates, token analysis, emphasis syntax, and prompt optimization tools.

**Suggested Features**:
- **Prompt Templates**:
  - Pre-built templates for common styles (cinematic, portrait, landscape, anime, etc.)
  - Variable substitution for dynamic prompts
  - Template library with categorization

- **Negative Prompt Presets**:
  - Common negative prompts for quality improvement
  - Style-specific negative prompts
  - Automatic negative prompt suggestions

- **Token Analysis**:
  - Token counting and limit warnings
  - Token importance scoring
  - Prompt truncation preview

- **Emphasis Syntax**:
  - Support for (word:weight) emphasis syntax
  - Bracket-based emphasis ((word)), [word]
  - Attention visualization and editing

**Potential API**:
```cpp
// Use prompt template
auto promptHelper = ofxStableDiffusionPromptHelpers::getInstance();
auto prompt = promptHelper.applyTemplate("cinematic_portrait", {
    {"subject", "astronaut"},
    {"lighting", "rim lighting"},
    {"mood", "dramatic"}
});
// Result: "cinematic portrait of astronaut, rim lighting, dramatic mood, ..."

// Analyze prompt
auto analysis = promptHelper.analyzePrompt("beautiful landscape with mountains");
ofLogNotice() << "Token count: " << analysis.tokenCount;
ofLogNotice() << "Estimated impact: " << analysis.estimatedImpact;
for (const auto& warning : analysis.warnings) {
    ofLogWarning() << warning;
}

// Get negative prompt preset
auto negativePrompt = promptHelper.getNegativePreset("quality_boost");
// Result: "low quality, blurry, distorted, watermark, ..."

// Parse emphasis syntax
auto parsed = promptHelper.parseEmphasis("(masterpiece:1.2), detailed face");
for (const auto& token : parsed.tokens) {
    ofLogNotice() << token.text << " (weight: " << token.weight << ")";
}

// Template management
promptHelper.loadTemplates("data/prompt_templates/");
auto categories = promptHelper.getTemplateCategories();
auto templates = promptHelper.getTemplatesInCategory("portraits");
```

**Integration Points**:
- Complements existing seed management (Feature #7)
- Works with batch processing (Feature #12) for prompt variations
- Enhances animation/interpolation (Feature #18) with prompt templates

**Template Categories**:
- Photography: cinematic, portrait, landscape, macro
- Art Styles: anime, oil painting, watercolor, sketch
- Scenes: interior, exterior, fantasy, sci-fi
- Quality: high quality, photorealistic, artistic

**Use Cases**:
- Beginners learning effective prompting
- Consistent style across multiple generations
- A/B testing different prompt formulations
- Educational tools for prompt engineering

### 12. Batch Processing Utilities ✅ IMPLEMENTED (v1.3.0)

**Status**: ✅ **COMPLETED** in version 1.3.0

**Overview**:
Batch processing utilities would enable systematic exploration of generation parameters through grid generation, parameter sweeps, and automated comparison workflows. This feature is essential for testing, experimentation, and finding optimal parameter combinations.

**Suggested Features**:
- **Grid Generation (X/Y/Z Plots)**:
  - 2D grids varying two parameters (e.g., CFG scale vs. steps)
  - 3D grids with multiple variations
  - Automatic grid layout and labeling
  - Export as single composite image or individual files

- **Parameter Sweeping**:
  - Automated parameter range exploration
  - Linear and logarithmic stepping
  - Multi-parameter combinatorial sweeps
  - Result ranking and sorting

- **A/B Comparison Tools**:
  - Side-by-side comparison generation
  - Automatic difference highlighting
  - Quality metrics and scoring
  - User preference tracking

- **Batch Export**:
  - Organized file naming schemes
  - Comprehensive metadata for each variation
  - CSV/JSON export of all parameters
  - Automatic gallery generation

**Potential API**:
```cpp
// Create X/Y grid (CFG scale vs. sample steps)
ofxStableDiffusionBatchProcessor batchProcessor;
ofxStableDiffusionGridSettings gridSettings;
gridSettings.baseRequest = baseRequest;
gridSettings.xAxis = ofxStableDiffusionParameter::CfgScale;
gridSettings.xValues = {1.5f, 3.0f, 5.0f, 7.5f, 10.0f};
gridSettings.yAxis = ofxStableDiffusionParameter::SampleSteps;
gridSettings.yValues = {10, 20, 30, 40, 50};
gridSettings.outputPath = "output/grid_cfg_steps.png";

auto gridResult = batchProcessor.generateGrid(gridSettings);
// Creates 5x5 grid (25 images) with labeled axes

// Parameter sweep
ofxStableDiffusionSweepSettings sweepSettings;
sweepSettings.baseRequest = baseRequest;
sweepSettings.parameter = ofxStableDiffusionParameter::Strength;
sweepSettings.rangeMin = 0.1f;
sweepSettings.rangeMax = 1.0f;
sweepSettings.steps = 10;
sweepSettings.stepMode = ofxStableDiffusionStepMode::Linear;

auto sweepResults = batchProcessor.parameterSweep(sweepSettings);
for (const auto& result : sweepResults.results) {
    ofLogNotice() << "Strength: " << result.parameterValue
                  << " Quality: " << result.qualityScore;
}

// A/B comparison
ofxStableDiffusionImageRequest requestA = baseRequest;
requestA.samplerMethod = EULER_A_SAMPLE_METHOD;

ofxStableDiffusionImageRequest requestB = baseRequest;
requestB.samplerMethod = DPMPP2Mv2_SAMPLE_METHOD;

auto comparison = batchProcessor.compareAB(requestA, requestB);
comparison.exportComparison("output/sampler_comparison.png");

// Batch with custom variations
std::vector<ofxStableDiffusionImageRequest> requests;
for (float cfg : {1.5f, 3.0f, 5.0f, 7.5f}) {
    auto req = baseRequest;
    req.cfgScale = cfg;
    requests.push_back(req);
}

auto batchResult = batchProcessor.processBatch(requests, "output/cfg_sweep/");
batchResult.exportMetadata("output/cfg_sweep/metadata.json");
```

**Integration Points**:
- Uses existing queue system (Feature #3) for batch management
- Leverages performance profiler (Feature #16) for batch timing
- Works with prompt helpers (Feature #11) for prompt variations
- Compatible with existing metadata export (Feature #19)

**Export Formats**:
- Grid images with parameter labels
- Individual images with systematic naming
- JSON/CSV metadata files
- HTML gallery pages
- Comparison reports with statistics

**Use Cases**:
- Parameter optimization and tuning
- Sampler/scheduler comparisons
- Prompt effectiveness testing
- Model capability exploration
- Research and documentation

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

**Status**: 📋 **PLANNED** (Note: Basic embedding support already exists via `reloadEmbeddings()` and `listEmbeddings()`)

**Overview**:
Enhanced textual inversion support would provide comprehensive management and discovery of textual embeddings/concepts, enabling users to easily incorporate custom concepts, styles, and characters into their generations. While basic embedding support exists, this feature would add advanced management, preview, and integration capabilities.

**Existing Features**:
- `reloadEmbeddings(path)`: Hot-reload embeddings from directory
- `listEmbeddings()`: Discover available embeddings with name/path pairs

**Suggested Enhancements**:
- **Advanced Embedding Management**:
  - Per-prompt embedding selection and weighting
  - Embedding conflict detection and resolution
  - Embedding combination recommendations
  - Automatic embedding discovery and indexing

- **Embedding Metadata**:
  - Embedding preview thumbnails
  - Trigger word suggestions
  - Compatible model information
  - Quality ratings and usage statistics

- **Embedding Discovery**:
  - Search and filter by category (character, style, concept, object)
  - Preview gallery with example generations
  - Community embedding repositories
  - Automatic updates and versioning

**Potential API**:
```cpp
// Enhanced embedding management (beyond existing reloadEmbeddings)
ofxStableDiffusionEmbeddingManager embManager;
embManager.scanDirectory("data/embeddings/");

// Get embedding info (new feature)
auto embInfo = embManager.getEmbeddingInfo("mycharacter");
ofLogNotice() << "Trigger word: " << embInfo.triggerWord;
ofLogNotice() << "Category: " << embInfo.category;
ofLogNotice() << "Preview: " << embInfo.previewImagePath;

// Use embedding in prompt with explicit weighting
ofxStableDiffusionImageRequest request;
request.prompt = "portrait of <mycharacter:1.2>, detailed face";
request.embeddings = embManager.getActiveEmbeddings();  // New field
sd.generate(request);

// Search and filter embeddings
auto characterEmbeddings = embManager.searchEmbeddings({
    {"category", "character"},
    {"style", "anime"}
});

// Get usage recommendations
auto recommendations = embManager.getRecommendations(request.prompt);
for (const auto& rec : recommendations) {
    ofLogNotice() << "Suggested: " << rec.name << " (" << rec.reason << ")";
}

// Embedding preview generation
auto preview = embManager.generatePreview("mycharacter", {512, 512});
```

**Integration Points**:
- Works with existing `reloadEmbeddings()` and `listEmbeddings()` API
- Integrates with prompt helpers (Feature #11) for trigger word suggestions
- Compatible with batch processing (Feature #12) for embedding comparisons
- Uses existing metadata export (Feature #19) for embedding tracking

**File Format Support**:
- .pt (PyTorch)
- .safetensors
- .bin
- Auto-detection of embedding type and version

**Use Cases**:
- Character consistency across generations
- Custom style applications
- Concept art and design workflows
- Brand identity and product visualization

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

### 20. Outpainting Helpers (Mask Preprocessing)

**Status**: 📋 **PLANNED** (Note: Basic inpainting implemented in v1.2.0)

**Overview**:
Outpainting helpers would provide utilities for extending images beyond their original boundaries with intelligent mask preprocessing. This builds on the existing inpainting implementation (Feature #8) by adding automatic padding, mask generation, and preprocessing tools.

**Suggested Features**:
- **Automatic Padding**:
  - Expand canvas in any direction (top, bottom, left, right)
  - Smart padding with edge extension or blur
  - Configurable padding size and mode
  - Automatic mask generation for padded regions

- **Mask Preprocessing Utilities**:
  - Gaussian blur for smooth transitions
  - Feathering for soft edges
  - Dilation and erosion for mask refinement
  - Edge detection and enhancement
  - Gradient masks for natural blending

- **Mask Generation Tools**:
  - Generate masks from transparency
  - Create masks from color selection
  - Geometric mask shapes (rectangles, ellipses, custom paths)
  - Invert and combine masks

- **Outpainting Presets**:
  - Extend to standard aspect ratios
  - Symmetric expansion
  - Directional expansion (panorama mode)
  - Auto-crop after generation

**Potential API**:
```cpp
// Outpainting with automatic padding
ofxStableDiffusionOutpaintingHelper outpaintHelper;
auto outpaintSettings = outpaintHelper.createOutpaintSettings(
    sourceImage,
    ofxStableDiffusionOutpaintDirection::Right,
    512  // Pixels to extend
);

ofxStableDiffusionImageRequest request;
request.mode = ofxStableDiffusionImageMode::Inpainting;
request.initImage = outpaintSettings.paddedImage;
request.maskImage = outpaintSettings.mask;
request.prompt = "continue the landscape";
sd.generate(request);

// Mask preprocessing
ofxStableDiffusionMaskProcessor maskProcessor;
sd_image_t mask = /* ... */;

// Blur mask edges for smooth transitions
mask = maskProcessor.blur(mask, 8.0f);  // Gaussian blur radius

// Feather mask edges
mask = maskProcessor.feather(mask, 16);  // Feather pixels

// Dilate/erode for mask refinement
mask = maskProcessor.dilate(mask, 4);
mask = maskProcessor.erode(mask, 2);

// Gradient mask for natural blending
mask = maskProcessor.createGradientMask(width, height,
    ofxStableDiffusionMaskGradient::LinearHorizontal);

// Generate mask from transparency
auto alphaMask = maskProcessor.fromAlphaChannel(imageWithAlpha);

// Combine masks
auto combinedMask = maskProcessor.combine(mask1, mask2,
    ofxStableDiffusionMaskOp::Union);

// Outpainting presets
auto panoramaSettings = outpaintHelper.createPanorama(
    sourceImage,
    ofxStableDiffusionAspectRatio::Ratio_21_9
);

auto symmetricSettings = outpaintHelper.expandSymmetric(
    sourceImage,
    256,  // Pixels on all sides
    ofxStableDiffusionPaddingMode::EdgeExtend
);
```

**Integration Points**:
- Extends existing inpainting support (Feature #8)
- Works with existing image request validation
- Compatible with batch processing (Feature #12) for multiple expansions
- Uses existing error handling for dimension validation

**Mask Preprocessing Operations**:
- Gaussian blur (smooth transitions)
- Feathering (soft edge falloff)
- Dilation/erosion (morphological operations)
- Edge detection (Sobel, Canny)
- Gradient generation (linear, radial)

**Use Cases**:
- Extending compositions beyond frame
- Creating panoramic views
- Fixing cropped images
- Generating seamless patterns
- Canvas expansion for flexible aspect ratios

### 21. Platform-Specific Optimizations

**Suggested Enhancement**:
- Metal acceleration improvements for macOS
- Mobile platform support (iOS/Android with reduced models)
- WebAssembly build for browser use

## Community Requests

This section is reserved for features requested by addon users. Please submit feature requests via GitHub issues with the `enhancement` label.

---

## Implementation Status Summary

### ✅ Completed Features (v1.0.0 - v1.3.0)
1. ✅ Advanced Error Handling (v1.0.0)
2. ✅ Model Preloading and Management (v1.1.0)
3. ✅ Generation Queue System (v1.1.0)
4. ✅ Advanced Progress Reporting (v1.1.0)
5. ✅ ControlNet Multi-Model Support (v1.2.0)
7. ✅ Image Seed Management (v1.1.0)
8. ✅ Inpainting and Outpainting (v1.2.0)
9. ✅ Real-time Generation Modes (v1.3.0)
10. ✅ Model Quantization Support (v1.3.0)
11. ✅ Prompt Engineering Helpers (v1.3.0)
12. ✅ Batch Processing Utilities (v1.3.0)
13. ✅ Advanced Sampling Options (v1.2.0)
16. ✅ Performance Profiling (v1.2.0)
18. ✅ Animation and Interpolation (v1.2.0)
19. ✅ Export and Metadata (v1.2.0)

### 🚧 Partially Completed
6. 🚧 LoRA Management System (v1.2.0 - discovery implemented, convenience methods pending)

### 📋 Planned
- Textual Inversion Support (Feature #14)
- Safety and Content Filtering (Feature #15)
- Enhanced ofxGgml Integration (Feature #17)
- Outpainting Helpers / Mask Preprocessing (Feature #20)
- Platform-Specific Optimizations (Feature #21)
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
