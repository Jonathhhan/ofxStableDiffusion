# Video Animation Examples

This document demonstrates the new video animation and interpolation features added in v1.2.0.

## Basic Prompt Interpolation

Create smooth transitions between different prompts:

```cpp
#include "ofxStableDiffusion.h"

void ofApp::setup() {
    ofxStableDiffusion sd;

    // Define prompt keyframes
    std::vector<ofxStableDiffusionPromptKeyframe> keyframes;
    keyframes.push_back({0, "a peaceful mountain landscape"});
    keyframes.push_back({30, "a stormy ocean with lightning"});
    keyframes.push_back({60, "a vibrant sunset over a desert"});

    // Create video request with prompt interpolation
    auto request = ofxStableDiffusionCreatePromptInterpolationRequest(
        keyframes,
        60,  // 60 frames
        512, 512,
        ofxStableDiffusionInterpolationMode::Smooth
    );

    // Set other parameters
    request.fps = 10;
    request.cfgScale = 7.5f;
    request.sampleSteps = 20;

    // Generate video
    sd.generateVideo(request);
}
```

## Parameter Animation

Animate generation parameters over time:

```cpp
void ofApp::setupParameterAnimation() {
    ofxStableDiffusion sd;

    // Define parameter keyframes
    std::vector<ofxStableDiffusionKeyframe> keyframes;

    // Frame 0: Low CFG, high strength
    ofxStableDiffusionKeyframe kf1(0);
    kf1.cfgScale = 3.0f;
    kf1.strength = 0.8f;
    kf1.prompt = "a dreamy forest";
    keyframes.push_back(kf1);

    // Frame 30: Medium CFG and strength
    ofxStableDiffusionKeyframe kf2(30);
    kf2.cfgScale = 7.0f;
    kf2.strength = 0.5f;
    kf2.prompt = "a mystical forest";
    keyframes.push_back(kf2);

    // Frame 60: High CFG, low strength
    ofxStableDiffusionKeyframe kf3(60);
    kf3.cfgScale = 12.0f;
    kf3.strength = 0.2f;
    kf3.prompt = "a realistic forest";
    keyframes.push_back(kf3);

    // Create request with parameter animation
    auto request = ofxStableDiffusionCreateParameterAnimationRequest(
        keyframes,
        60,
        512, 512,
        ofxStableDiffusionInterpolationMode::EaseInOut
    );

    request.fps = 10;
    sd.generateVideo(request);
}
```

## Seed Sequence Animation

Generate variations by incrementing the seed:

```cpp
void ofApp::setupSeedSequence() {
    ofxStableDiffusion sd;

    // Create request with seed sequence
    auto request = ofxStableDiffusionCreateSeedSequenceRequest(
        42,      // Starting seed
        100,     // 100 frames
        1,       // Increment seed by 1 each frame
        512, 512
    );

    request.fps = 15;
    request.cfgScale = 7.5f;
    request.sampleSteps = 25;

    sd.generateVideo(request);
}
```

## Combined Animation Features

Combine prompt interpolation, parameter animation, and seed sequences:

```cpp
void ofApp::setupCombinedAnimation() {
    ofxStableDiffusion sd;

    ofxStableDiffusionVideoRequest request;
    request.width = 512;
    request.height = 512;
    request.frameCount = 100;
    request.fps = 12;
    request.seed = 1000;

    // Enable prompt interpolation
    request.animationSettings.enablePromptInterpolation = true;
    request.animationSettings.promptInterpolationMode = ofxStableDiffusionInterpolationMode::Smooth;
    request.animationSettings.promptKeyframes = {
        {0, "sunrise over mountains"},
        {50, "midday in the mountains"},
        {100, "sunset over mountains"}
    };

    // Enable parameter animation
    request.animationSettings.enableParameterAnimation = true;
    request.animationSettings.parameterInterpolationMode = ofxStableDiffusionInterpolationMode::EaseInOut;

    ofxStableDiffusionKeyframe kf1(0);
    kf1.cfgScale = 5.0f;
    kf1.strength = 0.7f;

    ofxStableDiffusionKeyframe kf2(50);
    kf2.cfgScale = 8.0f;
    kf2.strength = 0.4f;

    ofxStableDiffusionKeyframe kf3(100);
    kf3.cfgScale = 6.0f;
    kf3.strength = 0.6f;

    request.animationSettings.parameterKeyframes = {kf1, kf2, kf3};

    // Enable seed sequence for subtle variations
    request.animationSettings.useSeedSequence = true;
    request.animationSettings.seedIncrement = 1;

    sd.generateVideo(request);
}
```

## Accessing Interpolated Values

Get interpolated values for specific frames:

```cpp
void ofApp::printFrameParameters() {
    ofxStableDiffusionVideoRequest request;
    // ... setup request with animation settings ...

    // Get parameters for frame 25
    std::string prompt = ofxStableDiffusionGetFramePrompt(request, 25);
    float cfgScale = ofxStableDiffusionGetFrameCfgScale(request, 25);
    float strength = ofxStableDiffusionGetFrameStrength(request, 25);
    int64_t seed = ofxStableDiffusionGetFrameSeed(request, 25);

    ofLogNotice() << "Frame 25:";
    ofLogNotice() << "  Prompt: " << prompt;
    ofLogNotice() << "  CFG Scale: " << cfgScale;
    ofLogNotice() << "  Strength: " << strength;
    ofLogNotice() << "  Seed: " << seed;
}
```

## Export Video Metadata

Export complete generation parameters for reproducibility:

```cpp
void ofApp::exportVideoMetadata() {
    ofxStableDiffusionVideoRequest request;
    // ... setup and generate video ...

    // Build frame metadata
    std::vector<ofJson> frameMetadata;
    for (int i = 0; i < request.frameCount; ++i) {
        ofJson metadata = ofxStableDiffusionGenerateFrameMetadata(
            i,
            ofxStableDiffusionGetFramePrompt(request, i),
            request.negativePrompt,
            ofxStableDiffusionGetFrameCfgScale(request, i),
            ofxStableDiffusionGetFrameStrength(request, i),
            ofxStableDiffusionGetFrameSeed(request, i),
            request.width,
            request.height,
            "path/to/model.safetensors"
        );
        frameMetadata.push_back(metadata);
    }

    // Export to JSON file
    request.animationSettings.exportParametersJson = true;
    request.animationSettings.outputDirectory = "output/";

    ofxStableDiffusionExportVideoParametersJson(
        "output/video_parameters.json",
        request.animationSettings,
        frameMetadata
    );

    ofLogNotice() << "Video metadata exported to video_parameters.json";
}
```

## Interpolation Modes Comparison

Different interpolation modes produce different motion feels:

```cpp
void ofApp::compareInterpolationModes() {
    std::vector<ofxStableDiffusionPromptKeyframe> keyframes = {
        {0, "calm water"},
        {30, "turbulent waves"}
    };

    // Linear - constant speed
    auto linearRequest = ofxStableDiffusionCreatePromptInterpolationRequest(
        keyframes, 30, 512, 512,
        ofxStableDiffusionInterpolationMode::Linear
    );

    // Smooth - gentle acceleration/deceleration (cosine)
    auto smoothRequest = ofxStableDiffusionCreatePromptInterpolationRequest(
        keyframes, 30, 512, 512,
        ofxStableDiffusionInterpolationMode::Smooth
    );

    // EaseIn - starts slow, accelerates
    auto easeInRequest = ofxStableDiffusionCreatePromptInterpolationRequest(
        keyframes, 30, 512, 512,
        ofxStableDiffusionInterpolationMode::EaseIn
    );

    // EaseOut - starts fast, decelerates
    auto easeOutRequest = ofxStableDiffusionCreatePromptInterpolationRequest(
        keyframes, 30, 512, 512,
        ofxStableDiffusionInterpolationMode::EaseOut
    );

    // EaseInOut - smooth S-curve
    auto easeInOutRequest = ofxStableDiffusionCreatePromptInterpolationRequest(
        keyframes, 30, 512, 512,
        ofxStableDiffusionInterpolationMode::EaseInOut
    );
}
```

## Tips and Best Practices

### 1. Prompt Interpolation
- Use 3-5 keyframes for smooth transitions
- Place keyframes at natural transition points
- Similar prompts blend better than drastically different ones
- Test with shorter videos first (10-30 frames)

### 2. Parameter Animation
- Subtle parameter changes work best (avoid extreme jumps)
- CFG scale range: 3.0-12.0 is usually safe
- Strength range: 0.2-0.8 for most use cases
- Use EaseInOut for the most natural feel

### 3. Seed Sequences
- Increment by 1 for subtle variations
- Larger increments (10+) create more dramatic changes
- Combine with fixed prompts for exploration
- Good for creating "evolution" style videos

### 4. Performance Considerations
- Longer videos take proportionally more time
- Each frame is a separate generation
- Consider generating preview with lower steps first
- Use appropriate sample steps (15-25 for most cases)

### 5. Export and Reproducibility
- Always export metadata for important videos
- Include all keyframe information
- Store the model path used
- Document any custom settings or parameters
