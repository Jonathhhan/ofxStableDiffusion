#pragma once

#include "../core/ofxStableDiffusionEnums.h"
#include "ofxStableDiffusionVideoAnimation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

inline const char * ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode mode) {
	switch (mode) {
	case ofxStableDiffusionVideoMode::Loop: return "Loop";
	case ofxStableDiffusionVideoMode::PingPong: return "PingPong";
	case ofxStableDiffusionVideoMode::Boomerang: return "Boomerang";
	case ofxStableDiffusionVideoMode::Standard:
	default:
		return "Standard";
	}
}

inline float ofxStableDiffusionVideoDurationSeconds(std::size_t frameCount, int fps) {
	if (frameCount == 0 || fps <= 0) {
		return 0.0f;
	}
	return static_cast<float>(frameCount) / static_cast<float>(fps);
}

inline int ofxStableDiffusionVideoFrameIndexForTime(std::size_t frameCount, int fps, float seconds) {
	if (frameCount == 0) {
		return -1;
	}
	if (fps <= 0) {
		return 0;
	}
	const float clampedSeconds = std::max(0.0f, seconds);
	const int index = static_cast<int>(std::floor(clampedSeconds * static_cast<float>(fps)));
	return std::min<int>(index, static_cast<int>(frameCount) - 1);
}

inline std::vector<int> ofxStableDiffusionBuildVideoFrameSequence(
	int sourceFrameCount,
	ofxStableDiffusionVideoMode mode) {
	std::vector<int> sequence;
	if (sourceFrameCount <= 0) {
		return sequence;
	}

	sequence.reserve(static_cast<std::size_t>(sourceFrameCount) * 2);
	for (int i = 0; i < sourceFrameCount; ++i) {
		sequence.push_back(i);
	}

	if (sourceFrameCount == 1) {
		return sequence;
	}

	switch (mode) {
	case ofxStableDiffusionVideoMode::Loop:
		sequence.push_back(0);
		break;
	case ofxStableDiffusionVideoMode::PingPong:
		for (int i = sourceFrameCount - 2; i > 0; --i) {
			sequence.push_back(i);
		}
		break;
	case ofxStableDiffusionVideoMode::Boomerang:
		for (int i = sourceFrameCount - 1; i >= 0; --i) {
			sequence.push_back(i);
		}
		break;
	case ofxStableDiffusionVideoMode::Standard:
	default:
		break;
	}

	return sequence;
}

//--------------------------------------------------------------
// Animation Helper Functions
//--------------------------------------------------------------

/// Create a video request with prompt interpolation between keyframes
inline ofxStableDiffusionVideoRequest ofxStableDiffusionCreatePromptInterpolationRequest(
	const std::vector<ofxStableDiffusionPromptKeyframe>& keyframes,
	int frameCount,
	int width = 576,
	int height = 1024,
	ofxStableDiffusionInterpolationMode mode = ofxStableDiffusionInterpolationMode::Smooth) {

	ofxStableDiffusionVideoRequest request;
	request.width = width;
	request.height = height;
	request.frameCount = frameCount;

	request.animationSettings.enablePromptInterpolation = true;
	request.animationSettings.promptKeyframes = keyframes;
	request.animationSettings.promptInterpolationMode = mode;

	return request;
}

/// Create a video request with parameter animation keyframes
inline ofxStableDiffusionVideoRequest ofxStableDiffusionCreateParameterAnimationRequest(
	const std::vector<ofxStableDiffusionKeyframe>& keyframes,
	int frameCount,
	int width = 576,
	int height = 1024,
	ofxStableDiffusionInterpolationMode mode = ofxStableDiffusionInterpolationMode::Smooth) {

	ofxStableDiffusionVideoRequest request;
	request.width = width;
	request.height = height;
	request.frameCount = frameCount;

	request.animationSettings.enableParameterAnimation = true;
	request.animationSettings.parameterKeyframes = keyframes;
	request.animationSettings.parameterInterpolationMode = mode;

	return request;
}

/// Create a video request with seed sequence (incremental seeds per frame)
inline ofxStableDiffusionVideoRequest ofxStableDiffusionCreateSeedSequenceRequest(
	int64_t startSeed,
	int frameCount,
	int64_t seedIncrement = 1,
	int width = 576,
	int height = 1024) {

	ofxStableDiffusionVideoRequest request;
	request.width = width;
	request.height = height;
	request.frameCount = frameCount;
	request.seed = startSeed;

	request.animationSettings.useSeedSequence = true;
	request.animationSettings.seedIncrement = seedIncrement;

	return request;
}

/// Get interpolated prompt for a specific frame
inline std::string ofxStableDiffusionGetFramePrompt(
	const ofxStableDiffusionVideoRequest& request,
	int frameNumber) {

	if (request.animationSettings.enablePromptInterpolation &&
	    !request.animationSettings.promptKeyframes.empty()) {
		return ofxStableDiffusionInterpolatePrompts(
			request.animationSettings.promptKeyframes,
			frameNumber,
			request.animationSettings.promptInterpolationMode
		);
	}

	return "";
}

/// Get interpolated CFG scale for a specific frame
inline float ofxStableDiffusionGetFrameCfgScale(
	const ofxStableDiffusionVideoRequest& request,
	int frameNumber) {

	if (request.animationSettings.enableParameterAnimation &&
	    !request.animationSettings.parameterKeyframes.empty()) {
		float value = ofxStableDiffusionGetInterpolatedParameter(
			request.animationSettings.parameterKeyframes,
			frameNumber,
			request.animationSettings.parameterInterpolationMode,
			[](const ofxStableDiffusionKeyframe& kf) { return kf.cfgScale; }
		);
		if (value >= 0.0f) {
			return value;
		}
	}

	return request.cfgScale;
}

/// Get interpolated strength for a specific frame
inline float ofxStableDiffusionGetFrameStrength(
	const ofxStableDiffusionVideoRequest& request,
	int frameNumber) {

	if (request.animationSettings.enableParameterAnimation &&
	    !request.animationSettings.parameterKeyframes.empty()) {
		float value = ofxStableDiffusionGetInterpolatedParameter(
			request.animationSettings.parameterKeyframes,
			frameNumber,
			request.animationSettings.parameterInterpolationMode,
			[](const ofxStableDiffusionKeyframe& kf) { return kf.strength; }
		);
		if (value >= 0.0f) {
			return value;
		}
	}

	return request.strength;
}

/// Get seed for a specific frame (handles seed sequence)
inline int64_t ofxStableDiffusionGetFrameSeed(
	const ofxStableDiffusionVideoRequest& request,
	int frameNumber) {

	if (request.animationSettings.useSeedSequence && request.seed >= 0) {
		return request.seed + (frameNumber * request.animationSettings.seedIncrement);
	}

	return request.seed;
}

