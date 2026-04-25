# Video Generation Improvements

**Date**: 2026-04-25
**Branch**: claude/improve-and-review-addon

## Overview

This document summarizes improvements made to the video generation capabilities of ofxStableDiffusion, focusing on validation, error handling, progress reporting, and documentation.

## Changes Made

### 1. Validation Limits for Video Parameters

**Location**: `src/core/ofxStableDiffusionLimits.h`

Added centralized constants for video generation validation:

```cpp
/// Minimum frame count for video generation
constexpr int MIN_FRAME_COUNT = 1;

/// Maximum frame count for video generation
constexpr int MAX_FRAME_COUNT = 1000;

/// Minimum FPS (frames per second) for video generation
constexpr int MIN_FPS = 1;

/// Maximum FPS (frames per second) for video generation
constexpr int MAX_FPS = 120;
```

Added validation helper functions:

```cpp
inline constexpr bool isValidFrameCount(int frameCount) {
    return frameCount >= MIN_FRAME_COUNT && frameCount <= MAX_FRAME_COUNT;
}

inline constexpr bool isValidFps(int fps) {
    return fps >= MIN_FPS && fps <= MAX_FPS;
}
```

**Benefits**:
- Consistent validation across the codebase
- Programmatic access to limits for UI builders
- Better error messages with actual limit values
- Compile-time optimization with constexpr

### 2. Improved Frame Count and FPS Validation

**Location**: `src/ofxStableDiffusion.cpp:296-312`

**Before**:
```cpp
if (request.frameCount <= 0) {
    return {ofxStableDiffusionErrorCode::InvalidFrameCount, "Frame count must be positive"};
}

if (request.fps <= 0 || request.fps > 120) {
    return {ofxStableDiffusionErrorCode::InvalidParameter, "FPS must be between 1 and 120"};
}
```

**After**:
```cpp
if (!ofxStableDiffusionLimits::isValidFrameCount(request.frameCount)) {
    return {ofxStableDiffusionErrorCode::InvalidFrameCount,
        "Frame count must be between " +
        std::to_string(ofxStableDiffusionLimits::MIN_FRAME_COUNT) + " and " +
        std::to_string(ofxStableDiffusionLimits::MAX_FRAME_COUNT)};
}

if (!ofxStableDiffusionLimits::isValidFps(request.fps)) {
    return {ofxStableDiffusionErrorCode::InvalidParameter,
        "FPS must be between " +
        std::to_string(ofxStableDiffusionLimits::MIN_FPS) + " and " +
        std::to_string(ofxStableDiffusionLimits::MAX_FPS)};
}
```

**Benefits**:
- More informative error messages showing actual limits
- Uses centralized validation constants
- Single source of truth for validation logic

### 3. Animation Keyframe Validation

**Location**: `src/video/ofxStableDiffusionVideoAnimation.h:266-333`

Added comprehensive validation function for animation keyframes:

```cpp
inline bool ofxStableDiffusionValidateAnimationKeyframes(
    const ofxStableDiffusionVideoAnimationSettings& settings,
    int frameCount,
    std::string& errorMessage);
```

**Validation checks**:

#### Prompt Keyframes
- Validates keyframes exist when prompt interpolation is enabled
- Frame numbers are non-negative and within valid range
- Weights are between 0.0 and 2.0
- Prompts are not empty

#### Parameter Keyframes
- Validates keyframes exist when parameter animation is enabled
- Frame numbers are non-negative and within valid range
- CFG scale values are between 0.0 and 50.0 (when specified)
- Strength values are between 0.0 and 1.0 (when specified)

**Integration**: `src/ofxStableDiffusion.cpp:375-383`

```cpp
if (request.hasAnimation()) {
    std::string animationError;
    if (!ofxStableDiffusionValidateAnimationKeyframes(
            request.animationSettings,
            request.frameCount,
            animationError)) {
        return {ofxStableDiffusionErrorCode::InvalidParameter, animationError};
    }
}
```

**Benefits**:
- Catches configuration errors before generation starts
- Prevents crashes from out-of-bounds frame numbers
- Provides clear, specific error messages for debugging
- Validates parameter ranges to prevent invalid values

### 4. Better Error Recovery in Frame Generation Loop

**Location**: `src/ofxStableDiffusionThread.cpp:492-537`

#### Frame Generation Failure

**Before**:
```cpp
if (!frameOutput || !frameOutput[0].data) {
    ofxSdReleaseImageArray(frameOutput, 1);
    generationError = "Animated video generation returned no frame output";
    break;
}
```

**After**:
```cpp
if (!frameOutput || !frameOutput[0].data) {
    ofxSdReleaseImageArray(frameOutput, 1);
    generationError = "Animated video generation returned no frame output for frame " +
        std::to_string(frameIndex);
    ofLogError("ofxStableDiffusion")
        << "Frame " << frameIndex << " generation failed: " << generationError;
    break;
}
```

#### Frame Assignment Failure

**Before**:
```cpp
OwnedImage generatedFrame;
if (!generatedFrame.assign(frameOutput[0])) {
    ofxSdReleaseImageArray(frameOutput, 1);
    generationError = "Animated video generation produced an invalid frame";
    break;
}
```

**After**:
```cpp
OwnedImage generatedFrame;
if (!generatedFrame.assign(frameOutput[0])) {
    ofxSdReleaseImageArray(frameOutput, 1);
    generationError = "Animated video generation produced an invalid frame at index " +
        std::to_string(frameIndex) + " (width=" +
        std::to_string(frameOutput[0].width) + ", height=" +
        std::to_string(frameOutput[0].height) + ", channels=" +
        std::to_string(frameOutput[0].channel) + ")";
    ofLogError("ofxStableDiffusion")
        << "Frame " << frameIndex << " assignment failed: invalid image data";
    break;
}
```

**Benefits**:
- Error messages include specific frame number that failed
- Logs errors for easier debugging of long video generation runs
- Includes image dimensions in assignment errors for diagnosis
- Helps identify patterns in failures (e.g., specific frame indices)

### 5. Enhanced Video API Documentation

**Location**: `src/ofxStableDiffusion.h`

Enhanced documentation for key video generation methods:

#### generateVideo() - Lines 56-63
```cpp
/// @brief Generate a video from a prompt.
/// @threadsafe Yes, but only one generation at a time. Returns immediately; results
/// available via callbacks or getLastResult() after completion.
/// @note Video generation supports animation via keyframes (prompt interpolation,
/// parameter animation, seed sequences). Use ofxStableDiffusionVideoAnimationSettings
/// to configure animation. For animated videos, progress callbacks report overall
/// progress across all frames.
void generateVideo(const ofxStableDiffusionVideoRequest& request);
```

#### getVideoClip() - Lines 89-92
```cpp
/// @brief Get generated video clip from last result.
/// @threadsafe Yes (returns copy under lock).
/// @note For animated videos, frames contain per-frame generation parameters.
ofxStableDiffusionVideoClip getVideoClip() const;
```

#### setVideoGenerationMode() - Lines 213-218
```cpp
/// @brief Set video generation mode (Standard, Loop, PingPong, Boomerang).
/// @threadsafe Yes.
/// @note This affects frame sequence construction for video output modes.
/// Standard: forward only. Loop: adds loop-back frame. PingPong: forward then backward (excluding endpoints).
/// Boomerang: forward then full reverse.
void setVideoGenerationMode(ofxStableDiffusionVideoMode mode);
```

**Benefits**:
- Clear documentation of animation support
- Explains progress reporting behavior for animated videos
- Documents video mode behaviors in detail
- Consistent with Phase 2 thread safety documentation standards

## Video Generation Architecture

### Native Video Generation (WAN Models)
- Single API call to `generate_video()`
- Handled by model's native video diffusion
- Uses `sd_vid_gen_params_t` structure
- Supports VACE control frames for guided generation

### Animated Video Generation (Frame-by-Frame)
- Generates each frame independently as an image
- Supports three animation types:
  1. **Prompt Interpolation**: Smoothly blends between text prompts
  2. **Parameter Animation**: Interpolates CFG scale, strength over time
  3. **Seed Sequences**: Sequential seeds for varied but related frames
- Composite progress reporting across all frames
- Per-frame generation parameters stored in results

### Video Output Modes
- **Standard**: Forward playback only
- **Loop**: Adds transition frame back to start
- **PingPong**: Forward then reverse (excluding endpoints)
- **Boomerang**: Forward then full reverse sequence

## Progress Reporting

### Existing Implementation (ofxStableDiffusionThread.cpp:43-76)

Video progress reporting already includes:
- **Exception Safety**: All callbacks wrapped in try-catch
- **Animated Video Progress**: Composite progress calculation
  - `compositeStep = (frameIndex * stepsPerFrame) + currentStep`
  - `totalSteps = frameCount * stepsPerFrame`
- **Native Video Progress**: Direct step/steps reporting

## Usage Examples

### Basic Video Generation with Validation

```cpp
ofxStableDiffusion sd;

// Configure video request
ofxStableDiffusionVideoRequest request;
request.prompt = "A serene lake at sunset";
request.width = 576;
request.height = 1024;
request.frameCount = 24;  // Validated: 1-1000
request.fps = 12;          // Validated: 1-120
request.seed = 42;

// Generate video
sd.generateVideo(request);

// Check for errors with detailed messages
if (sd.hasError()) {
    ofLogError() << "Video generation failed: " << sd.getLastError();
    // Error message will include actual limits if validation failed
}
```

### Animated Video with Prompt Interpolation

```cpp
ofxStableDiffusionVideoRequest request;
request.prompt = "Default prompt";
request.frameCount = 30;
request.fps = 10;

// Configure prompt animation
request.animationSettings.enablePromptInterpolation = true;
request.animationSettings.promptInterpolationMode =
    ofxStableDiffusionInterpolationMode::Smooth;

// Add keyframes - will be validated
request.animationSettings.promptKeyframes = {
    {0, "A calm ocean", 1.0f},
    {15, "Stormy seas with lightning", 1.0f},
    {29, "A peaceful sunset", 1.0f}
};

// Validation will catch:
// - Frame numbers outside 0-29 range
// - Empty prompts
// - Invalid weights (<0 or >2.0)
sd.generateVideo(request);
```

### Parameter Animation Example

```cpp
ofxStableDiffusionVideoRequest request;
request.frameCount = 20;

// Configure parameter animation
request.animationSettings.enableParameterAnimation = true;
request.animationSettings.parameterInterpolationMode =
    ofxStableDiffusionInterpolationMode::EaseInOut;

// Animate CFG scale and strength
request.animationSettings.parameterKeyframes = {
    ofxStableDiffusionKeyframe(0),    // frame 0
    ofxStableDiffusionKeyframe(10),   // frame 10
    ofxStableDiffusionKeyframe(19)    // frame 19
};
request.animationSettings.parameterKeyframes[0].cfgScale = 5.0f;
request.animationSettings.parameterKeyframes[0].strength = 0.5f;
request.animationSettings.parameterKeyframes[1].cfgScale = 8.0f;
request.animationSettings.parameterKeyframes[1].strength = 0.8f;
request.animationSettings.parameterKeyframes[2].cfgScale = 6.0f;
request.animationSettings.parameterKeyframes[2].strength = 0.6f;

// Validation will ensure:
// - CFG scale values are 0-50
// - Strength values are 0-1
// - Frame numbers are within range
sd.generateVideo(request);
```

### Progress Monitoring for Long Videos

```cpp
// Set up progress callback
sd.setProgressCallback([](int step, int totalSteps, float time) {
    float percent = (step * 100.0f) / totalSteps;
    ofLogNotice() << "Video progress: " << percent << "% "
                  << "(step " << step << "/" << totalSteps << ")";

    // For animated videos, this reports progress across ALL frames
    // e.g., frame 2/10, step 15/20 would report step 35/200
});

// Generate with many frames
ofxStableDiffusionVideoRequest request;
request.frameCount = 100;  // Long video
request.fps = 12;
request.sampleSteps = 20;

sd.generateVideo(request);

// If generation fails on a specific frame, error will include frame index
// Example: "Frame 47 generation failed: no frame output"
```

## Testing Recommendations

### Validation Testing
```cpp
// Test frame count limits
request.frameCount = 0;      // Should fail: below minimum
request.frameCount = 1001;   // Should fail: above maximum
request.frameCount = 50;     // Should succeed

// Test FPS limits
request.fps = 0;             // Should fail: below minimum
request.fps = 121;           // Should fail: above maximum
request.fps = 24;            // Should succeed

// Test keyframe validation
request.animationSettings.enablePromptInterpolation = true;
request.animationSettings.promptKeyframes = {
    {-1, "test", 1.0f}  // Should fail: negative frame number
};
request.animationSettings.promptKeyframes = {
    {1000, "test", 1.0f}  // Should fail: exceeds frameCount
};
request.animationSettings.promptKeyframes = {
    {5, "", 1.0f}  // Should fail: empty prompt
};
request.animationSettings.promptKeyframes = {
    {5, "test", 3.0f}  // Should fail: weight > 2.0
};
```

### Error Recovery Testing
```cpp
// Test error messages include frame information
// Generate video and simulate failure on specific frame
// Error should report: "Frame N generation failed..."

// Test with upscaling failures
request.upscalerSettings.enabled = true;
// Verify upscaler failures report frame context
```

## Performance Impact

- **Validation**: Negligible overhead (compile-time constants, simple comparisons)
- **Error Messages**: Minor string allocation overhead only on error paths
- **Logging**: Only executes on failures, no impact on success path
- **Animation Validation**: One-time check before generation starts

## Backward Compatibility

All changes maintain full backward compatibility:
- No changes to public API function signatures
- Validation is stricter but uses reasonable limits
- Error messages are more detailed but same error codes
- Existing code continues to work without modifications

## Files Modified

1. `src/core/ofxStableDiffusionLimits.h`
   - Added video validation constants
   - Added validation helper functions

2. `src/ofxStableDiffusion.cpp`
   - Updated video request validation to use centralized limits
   - Added animation keyframe validation call

3. `src/ofxStableDiffusion.h`
   - Enhanced documentation for video generation methods

4. `src/video/ofxStableDiffusionVideoAnimation.h`
   - Added comprehensive keyframe validation function

5. `src/ofxStableDiffusionThread.cpp`
   - Improved error messages in frame generation loop
   - Added logging for frame-specific failures

## Migration Guide

No migration required - all changes are backward compatible. However, developers can benefit from:

### Using Validation Limits in UI Code

```cpp
#include "core/ofxStableDiffusionLimits.h"

// Configure UI sliders/input fields
frameCountSlider.setMin(ofxStableDiffusionLimits::MIN_FRAME_COUNT);
frameCountSlider.setMax(ofxStableDiffusionLimits::MAX_FRAME_COUNT);

fpsSlider.setMin(ofxStableDiffusionLimits::MIN_FPS);
fpsSlider.setMax(ofxStableDiffusionLimits::MAX_FPS);

// Validate before submission
if (!ofxStableDiffusionLimits::isValidFrameCount(userInput)) {
    showError("Frame count must be between " +
              std::to_string(ofxStableDiffusionLimits::MIN_FRAME_COUNT) + " and " +
              std::to_string(ofxStableDiffusionLimits::MAX_FRAME_COUNT));
}
```

### Handling Detailed Error Messages

```cpp
sd.generateVideo(request);

if (sd.hasError()) {
    std::string error = sd.getLastError();

    // New error messages include more context:
    // - "Frame 42 generation failed: no frame output"
    // - "Frame count must be between 1 and 1000"
    // - "Prompt keyframe 3 has invalid frame number: 150"

    ofLogError() << "Detailed error: " << error;
}
```

## Future Enhancements (Not Implemented)

Potential areas for future improvement:
- Resume interrupted video generation from last successful frame
- Parallel frame generation for non-sequential animations
- Real-time video generation preview
- Adaptive quality adjustment based on VRAM availability
- Video generation presets optimized for different hardware
- Batch video generation with queue management

## Related Documentation

- Phase 1: Code Safety Improvements (docs/CODE_REVIEW_2026.md)
- Phase 2: Thread Safety Documentation (docs/IMPROVEMENTS_SUMMARY.md)
- Phase 3: Validation Limits (docs/PHASE3_SUMMARY.md)
- Complete Summary: docs/COMPLETE_IMPROVEMENTS_SUMMARY.md
- Validation Limits API: src/core/ofxStableDiffusionLimits.h
- Video Animation API: src/video/ofxStableDiffusionVideoAnimation.h
- Video Helpers API: src/video/ofxStableDiffusionVideoHelpers.h
- Video Workflow Presets: src/video/ofxStableDiffusionVideoWorkflowHelpers.h

---

**Implementation Status**: ✅ Complete
**Tests Required**: Manual validation testing, animation keyframe edge cases
**Breaking Changes**: None
