#pragma once

#include "../core/ofxStableDiffusionTypes.h"
#include "ofxStableDiffusionVideoAnimation.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

// Adaptive Quality Scaling Settings
struct ofxStableDiffusionAdaptiveQualitySettings {
	bool enabled = false;
	float targetSecondsPerFrame = 2.0f;  // Target generation time per frame
	int minSampleSteps = 8;
	int maxSampleSteps = 50;
	int minResolution = 256;
	int maxResolution = 1024;
	float qualityAdjustmentRate = 0.2f;  // How quickly to adjust (0-1)
	int warmupFrames = 2;  // Number of frames to measure before adjusting
};

// Temporal Consistency Settings
struct ofxStableDiffusionTemporalConsistencySettings {
	bool enabled = true;
	float strengthModulation = 0.1f;  // Vary strength slightly between frames
	float cfgScaleModulation = 0.2f;  // Vary CFG scale slightly
	bool useFrameBlending = false;  // Blend with previous frame
	float blendWeight = 0.15f;  // Weight of previous frame when blending
	int lookbackFrames = 1;  // How many frames back to consider
};

// Frame Interpolation Cache
struct ofxStableDiffusionFrameCache {
	std::unordered_map<int, ofxStableDiffusionImageFrame> frames;
	int maxCachedFrames = 10;
	bool enabled = true;

	void addFrame(int frameNumber, const ofxStableDiffusionImageFrame& frame) {
		if (!enabled) return;

		// LRU eviction if cache is full
		if (static_cast<int>(frames.size()) >= maxCachedFrames) {
			// Remove oldest frame (simple strategy - can be improved)
			auto oldest = frames.begin();
			frames.erase(oldest);
		}

		frames[frameNumber] = frame;
	}

	bool hasFrame(int frameNumber) const {
		return enabled && frames.find(frameNumber) != frames.end();
	}

	const ofxStableDiffusionImageFrame* getFrame(int frameNumber) const {
		if (!enabled) return nullptr;
		auto it = frames.find(frameNumber);
		return (it != frames.end()) ? &it->second : nullptr;
	}

	void clear() {
		frames.clear();
	}

	int size() const {
		return static_cast<int>(frames.size());
	}
};

// Quality Metrics for Video
struct ofxStableDiffusionVideoQualityMetrics {
	float averageGenerationTimePerFrame = 0.0f;
	float minGenerationTime = 0.0f;
	float maxGenerationTime = 0.0f;
	int totalFrames = 0;
	float totalGenerationTime = 0.0f;
	std::vector<float> perFrameTimes;
	float temporalCoherenceScore = 0.0f;  // 0-1, higher is better
	int droppedFrames = 0;
	int cachedFrames = 0;

	void recordFrameTime(float seconds) {
		perFrameTimes.push_back(seconds);
		totalFrames++;
		totalGenerationTime += seconds;

		if (totalFrames == 1) {
			minGenerationTime = maxGenerationTime = seconds;
		} else {
			minGenerationTime = std::min(minGenerationTime, seconds);
			maxGenerationTime = std::max(maxGenerationTime, seconds);
		}

		averageGenerationTimePerFrame = totalGenerationTime / totalFrames;
	}

	float getVariance() const {
		if (perFrameTimes.size() < 2) return 0.0f;

		float variance = 0.0f;
		for (float time : perFrameTimes) {
			float diff = time - averageGenerationTimePerFrame;
			variance += diff * diff;
		}
		return variance / perFrameTimes.size();
	}

	float getStdDeviation() const {
		return std::sqrt(getVariance());
	}
};

// Memory-Efficient Video Generation Settings
struct ofxStableDiffusionVideoMemorySettings {
	bool enableStreamingMode = false;  // Generate and save frames as we go
	bool clearIntermediateBuffers = true;  // Clear buffers between frames
	int maxFramesInMemory = 30;  // Maximum frames to keep in memory at once
	bool useReducedPrecision = false;  // Use lower precision for intermediate steps
	std::string streamingOutputDirectory;  // Where to save frames in streaming mode
};

namespace ofxStableDiffusionVideoPerformanceHelpers {

// Adjust quality parameters based on performance
inline void adjustQualityForPerformance(
	ofxStableDiffusionVideoRequest& request,
	const ofxStableDiffusionAdaptiveQualitySettings& settings,
	const ofxStableDiffusionVideoQualityMetrics& metrics) {

	if (!settings.enabled || metrics.totalFrames < settings.warmupFrames) {
		return;
	}

	const float avgTime = metrics.averageGenerationTimePerFrame;
	const float targetTime = settings.targetSecondsPerFrame;
	const float ratio = avgTime / targetTime;

	// If we're taking too long, reduce quality
	if (ratio > 1.2f) {
		// Reduce sample steps
		const int reduction = static_cast<int>(request.sampleSteps * settings.qualityAdjustmentRate);
		request.sampleSteps = std::max(
			settings.minSampleSteps,
			request.sampleSteps - reduction);

		// Optionally reduce resolution
		if (request.width > settings.minResolution && ratio > 1.5f) {
			const float scale = 0.9f;
			request.width = static_cast<int>(request.width * scale / 64) * 64;
			request.height = static_cast<int>(request.height * scale / 64) * 64;
			request.width = std::max(settings.minResolution, request.width);
			request.height = std::max(settings.minResolution, request.height);
		}
	}
	// If we're going fast enough, increase quality
	else if (ratio < 0.8f && request.sampleSteps < settings.maxSampleSteps) {
		const int increase = static_cast<int>(request.sampleSteps * settings.qualityAdjustmentRate * 0.5f);
		request.sampleSteps = std::min(
			settings.maxSampleSteps,
			request.sampleSteps + increase);
	}
}

// Apply temporal consistency adjustments to frame parameters
inline void applyTemporalConsistency(
	ofxStableDiffusionVideoRequest& frameRequest,
	int frameNumber,
	const ofxStableDiffusionTemporalConsistencySettings& settings,
	const ofxStableDiffusionImageFrame* previousFrame = nullptr) {

	if (!settings.enabled) return;

	// Modulate strength slightly for smoother transitions
	if (settings.strengthModulation > 0.0f && frameNumber > 0) {
		const float variation = std::sin(frameNumber * 0.5f) * settings.strengthModulation;
		frameRequest.strength = std::max(0.1f, std::min(0.95f,
			frameRequest.strength + variation));
	}

	// Modulate CFG scale for temporal coherence
	if (settings.cfgScaleModulation > 0.0f && frameNumber > 0) {
		const float variation = std::cos(frameNumber * 0.3f) * settings.cfgScaleModulation;
		frameRequest.cfgScale = std::max(1.0f,
			frameRequest.cfgScale + variation);
	}

	// Use higher temporal coherence for sequential frames
	if (frameRequest.hasAnimation() && frameNumber > 0) {
		// Adjust seed variation to be smoother
		if (frameRequest.animationSettings.useSeedSequence) {
			// Prefer noise mode for smoother transitions
			if (frameRequest.animationSettings.seedVariationMode ==
				ofxStableDiffusionSeedVariationMode::Sequential) {
				frameRequest.animationSettings.seedVariationMode =
					ofxStableDiffusionSeedVariationMode::Noise;
			}
		}

		// Increase temporal coherence setting
		frameRequest.animationSettings.temporalCoherence = std::min(1.0f,
			frameRequest.animationSettings.temporalCoherence + 0.1f);
	}
}

// Calculate optimal batch processing strategy
inline std::vector<std::vector<int>> calculateOptimalBatchStrategy(
	int totalFrames,
	int maxFramesPerBatch,
	const ofxStableDiffusionVideoMemorySettings& memorySettings) {

	std::vector<std::vector<int>> batches;

	const int framesPerBatch = memorySettings.enableStreamingMode ?
		memorySettings.maxFramesInMemory : maxFramesPerBatch;

	for (int i = 0; i < totalFrames; i += framesPerBatch) {
		std::vector<int> batch;
		const int batchEnd = std::min(i + framesPerBatch, totalFrames);
		for (int j = i; j < batchEnd; ++j) {
			batch.push_back(j);
		}
		batches.push_back(batch);
	}

	return batches;
}

// Estimate memory usage for video generation
inline float estimateVideoMemoryUsageMB(
	const ofxStableDiffusionVideoRequest& request,
	int framesInMemory = -1) {

	if (framesInMemory < 0) {
		framesInMemory = request.frameCount;
	}

	// Base memory for each frame (RGB, 8-bit per channel)
	const float bytesPerPixel = 3.0f;
	const float pixelsPerFrame = request.width * request.height;
	const float bytesPerFrame = pixelsPerFrame * bytesPerPixel;
	const float mbPerFrame = bytesPerFrame / (1024.0f * 1024.0f);

	// Account for intermediate buffers (diffusion process)
	const float intermediateMultiplier = 3.0f;  // Rough estimate

	// Total memory estimate
	const float totalMB = (mbPerFrame * framesInMemory * intermediateMultiplier);

	return totalMB;
}

// Calculate optimal frame skip for preview generation
inline int calculateOptimalFrameSkip(
	int totalFrames,
	float targetPreviewDuration = 2.0f,
	int targetFPS = 12) {

	const int targetFrameCount = static_cast<int>(targetPreviewDuration * targetFPS);

	if (totalFrames <= targetFrameCount) {
		return 1;  // No skipping needed
	}

	// Calculate skip factor
	const int skip = (totalFrames + targetFrameCount - 1) / targetFrameCount;
	return std::max(1, skip);
}

// Optimize video request for quality/performance trade-off
inline void optimizeVideoRequestForTarget(
	ofxStableDiffusionVideoRequest& request,
	const std::string& target) {

	if (target == "ultrafast") {
		request.sampleSteps = std::min(8, request.sampleSteps);
		request.cfgScale = std::min(4.0f, request.cfgScale);
		request.width = std::min(384, request.width);
		request.height = std::min(576, request.height);
		request.frameCount = std::min(12, request.frameCount);
	}
	else if (target == "fast") {
		request.sampleSteps = std::min(12, request.sampleSteps);
		request.cfgScale = std::min(5.0f, request.cfgScale);
		request.width = std::min(512, request.width);
		request.height = std::min(768, request.height);
	}
	else if (target == "balanced") {
		request.sampleSteps = std::min(20, request.sampleSteps);
		request.cfgScale = std::min(7.0f, request.cfgScale);
	}
	else if (target == "quality") {
		request.sampleSteps = std::max(28, std::min(40, request.sampleSteps));
		request.cfgScale = std::max(6.0f, std::min(8.0f, request.cfgScale));
		request.strength = std::max(0.7f, request.strength);
	}
	else if (target == "highquality") {
		request.sampleSteps = std::max(35, std::min(50, request.sampleSteps));
		request.cfgScale = std::max(7.0f, std::min(9.0f, request.cfgScale));
		request.strength = std::max(0.75f, request.strength);
	}
}

// Calculate frame-to-frame similarity score (0-1)
inline float calculateFrameSimilarity(
	const ofxStableDiffusionImageFrame& frame1,
	const ofxStableDiffusionImageFrame& frame2,
	int sampleStep = 8) {

	if (!frame1.isAllocated() || !frame2.isAllocated()) {
		return 0.0f;
	}

	if (frame1.width() != frame2.width() || frame1.height() != frame2.height()) {
		return 0.0f;
	}

	// Sample pixels for efficiency
	uint64_t totalDiff = 0;
	int sampleCount = 0;

	for (int y = 0; y < frame1.height(); y += sampleStep) {
		for (int x = 0; x < frame1.width(); x += sampleStep) {
			const ofColor c1 = frame1.pixels.getColor(x, y);
			const ofColor c2 = frame2.pixels.getColor(x, y);

			const int diffR = std::abs(static_cast<int>(c1.r) - static_cast<int>(c2.r));
			const int diffG = std::abs(static_cast<int>(c1.g) - static_cast<int>(c2.g));
			const int diffB = std::abs(static_cast<int>(c1.b) - static_cast<int>(c2.b));

			totalDiff += (diffR + diffG + diffB);
			sampleCount++;
		}
	}

	if (sampleCount == 0) return 0.0f;

	const float avgDiff = static_cast<float>(totalDiff) / (sampleCount * 3.0f * 255.0f);
	const float similarity = 1.0f - avgDiff;

	return std::max(0.0f, std::min(1.0f, similarity));
}

// Validate video generation parameters for quality/performance
inline std::vector<std::string> validateVideoPerformance(
	const ofxStableDiffusionVideoRequest& request) {

	std::vector<std::string> warnings;

	// Check for potentially slow configuration
	const int totalSteps = request.frameCount * request.sampleSteps;
	if (totalSteps > 2000) {
		warnings.push_back("High total step count (" + std::to_string(totalSteps) +
			") may result in very long generation time");
	}

	// Check resolution
	const int totalPixels = request.width * request.height;
	if (totalPixels > 786432) {  // 1024x768
		warnings.push_back("High resolution may significantly impact performance");
	}

	// Check frame count
	if (request.frameCount > 100) {
		warnings.push_back("High frame count (" + std::to_string(request.frameCount) +
			") may require substantial memory");
	}

	// Check animation complexity
	if (request.hasAnimation()) {
		const auto& anim = request.animationSettings;
		if (anim.enablePromptInterpolation && anim.promptKeyframes.size() > 10) {
			warnings.push_back("Large number of prompt keyframes may impact performance");
		}
		if (anim.enableParameterAnimation && anim.parameterKeyframes.size() > 10) {
			warnings.push_back("Large number of parameter keyframes may impact performance");
		}
	}

	// Check CFG scale
	if (request.cfgScale > 12.0f) {
		warnings.push_back("Very high CFG scale may reduce quality and performance");
	}

	// Check sample steps
	if (request.sampleSteps > 50) {
		warnings.push_back("High sample steps have diminishing returns on quality");
	}

	// Check temporal coherence settings
	if (request.animationSettings.temporalCoherence < 0.3f) {
		warnings.push_back("Low temporal coherence may result in flickering between frames");
	}

	return warnings;
}

// Calculate optimal settings for smooth video playback
inline ofxStableDiffusionVideoRequest optimizeForSmoothPlayback(
	const ofxStableDiffusionVideoRequest& request,
	float targetDurationSeconds = 3.0f) {

	ofxStableDiffusionVideoRequest optimized = request;

	// Calculate optimal FPS and frame count
	const int targetFPS = 24;  // Cinematic standard
	const int targetFrameCount = static_cast<int>(targetDurationSeconds * targetFPS);

	optimized.fps = targetFPS;
	optimized.frameCount = targetFrameCount;

	// Adjust for smooth interpolation
	if (optimized.animationSettings.enablePromptInterpolation ||
		optimized.animationSettings.enableParameterAnimation) {
		// Use smooth interpolation mode
		optimized.animationSettings.promptInterpolationMode =
			ofxStableDiffusionInterpolationMode::Smooth;
		optimized.animationSettings.parameterInterpolationMode =
			ofxStableDiffusionInterpolationMode::Smooth;
	}

	// Enable temporal coherence for smoothness
	optimized.animationSettings.temporalCoherence = 0.7f;

	// Use noise-based seed variation for organic transitions
	if (optimized.animationSettings.useSeedSequence) {
		optimized.animationSettings.seedVariationMode =
			ofxStableDiffusionSeedVariationMode::Noise;
	}

	return optimized;
}

} // namespace ofxStableDiffusionVideoPerformanceHelpers
