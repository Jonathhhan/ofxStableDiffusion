#pragma once

#include "../core/ofxStableDiffusionTypes.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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

inline bool ofxStableDiffusionTryMultiplyInt64(int64_t a, int64_t b, int64_t& result) {
	const int64_t maxValue = std::numeric_limits<int64_t>::max();
	const int64_t minValue = std::numeric_limits<int64_t>::min();
	if (a == 0 || b == 0) {
		result = 0;
		return true;
	}
	if (a > 0) {
		if (b > 0) {
			if (a > maxValue / b) {
				return false;
			}
		} else if (b < minValue / a) {
			return false;
		}
	} else if (b > 0) {
		if (a < minValue / b) {
			return false;
		}
	} else if (a < maxValue / b) {
		return false;
	}
	result = a * b;
	return true;
}

inline bool ofxStableDiffusionTryAddInt64(int64_t a, int64_t b, int64_t& result) {
	const int64_t maxValue = std::numeric_limits<int64_t>::max();
	const int64_t minValue = std::numeric_limits<int64_t>::min();
	if (b > 0 && a > maxValue - b) {
		return false;
	}
	if (b < 0 && a < minValue - b) {
		return false;
	}
	result = a + b;
	return true;
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

inline std::string ofxStableDiffusionGetFramePrompt(
	const ofxStableDiffusionVideoRequest& request,
	int frameNumber) {
	if (request.animationSettings.enablePromptInterpolation &&
		!request.animationSettings.promptKeyframes.empty()) {
		return ofxStableDiffusionInterpolatePrompts(
			request.animationSettings.promptKeyframes,
			frameNumber,
			request.animationSettings.promptInterpolationMode);
	}

	const std::string keyframedPrompt = ofxStableDiffusionGetKeyframedString(
		request.animationSettings.parameterKeyframes,
		frameNumber,
		[](const ofxStableDiffusionKeyframe& keyframe) { return keyframe.prompt; });
	return keyframedPrompt.empty() ? request.prompt : keyframedPrompt;
}

inline std::string ofxStableDiffusionGetFrameNegativePrompt(
	const ofxStableDiffusionVideoRequest& request,
	int frameNumber) {
	const std::string keyframedNegativePrompt = ofxStableDiffusionGetKeyframedString(
		request.animationSettings.parameterKeyframes,
		frameNumber,
		[](const ofxStableDiffusionKeyframe& keyframe) { return keyframe.negativePrompt; });
	return keyframedNegativePrompt.empty() ? request.negativePrompt : keyframedNegativePrompt;
}

inline float ofxStableDiffusionGetFrameCfgScale(
	const ofxStableDiffusionVideoRequest& request,
	int frameNumber) {
	if (request.animationSettings.enableParameterAnimation &&
		!request.animationSettings.parameterKeyframes.empty()) {
		const float value = ofxStableDiffusionGetInterpolatedParameter(
			request.animationSettings.parameterKeyframes,
			frameNumber,
			request.animationSettings.parameterInterpolationMode,
			[](const ofxStableDiffusionKeyframe& keyframe) { return keyframe.cfgScale; });
		if (value >= 0.0f) {
			return value;
		}
	}
	return request.cfgScale;
}

inline float ofxStableDiffusionGetFrameStrength(
	const ofxStableDiffusionVideoRequest& request,
	int frameNumber) {
	if (request.animationSettings.enableParameterAnimation &&
		!request.animationSettings.parameterKeyframes.empty()) {
		const float value = ofxStableDiffusionGetInterpolatedParameter(
			request.animationSettings.parameterKeyframes,
			frameNumber,
			request.animationSettings.parameterInterpolationMode,
			[](const ofxStableDiffusionKeyframe& keyframe) { return keyframe.strength; });
		if (value >= 0.0f) {
			return value;
		}
	}
	return request.strength;
}

inline int64_t ofxStableDiffusionGetFrameSeed(
	const ofxStableDiffusionVideoRequest& request,
	int frameNumber) {
	if (request.animationSettings.enableParameterAnimation &&
		!request.animationSettings.parameterKeyframes.empty()) {
		const int64_t keyframedSeed = ofxStableDiffusionGetKeyframedSeed(
			request.animationSettings.parameterKeyframes,
			frameNumber);
		if (keyframedSeed >= 0) {
			return keyframedSeed;
		}
	}

	if (request.animationSettings.useSeedSequence && request.seed >= 0) {
		const int64_t frameOffset = static_cast<int64_t>(frameNumber);
		const int64_t increment = request.animationSettings.seedIncrement;
		if (frameOffset == 0 || increment == 0) {
			return request.seed;
		}
		const int64_t maxValue = std::numeric_limits<int64_t>::max();
		const int64_t minValue = std::numeric_limits<int64_t>::min();
#if defined(__SIZEOF_INT128__)
		const __int128 expandedSeed = static_cast<__int128>(request.seed);
		const __int128 expandedOffset = static_cast<__int128>(frameOffset);
		const __int128 expandedIncrement = static_cast<__int128>(increment);
		const __int128 expandedValue = expandedSeed + (expandedOffset * expandedIncrement);
		if (expandedValue > static_cast<__int128>(maxValue)) {
			return maxValue;
		}
		if (expandedValue < static_cast<__int128>(minValue)) {
			return minValue;
		}
		return static_cast<int64_t>(expandedValue);
#elif defined(__GNUC__) || defined(__clang__)
		int64_t delta = 0;
		if (__builtin_mul_overflow(frameOffset, increment, &delta)) {
			return ((frameOffset < 0) != (increment < 0)) ? minValue : maxValue;
		}
		int64_t expandedValue = 0;
		if (__builtin_add_overflow(request.seed, delta, &expandedValue)) {
			return delta < 0 ? minValue : maxValue;
		}
		return expandedValue;
#else
		int64_t delta = 0;
		if (!ofxStableDiffusionTryMultiplyInt64(frameOffset, increment, delta)) {
			return ((frameOffset < 0) != (increment < 0)) ? minValue : maxValue;
		}
		int64_t expandedValue = 0;
		if (!ofxStableDiffusionTryAddInt64(request.seed, delta, expandedValue)) {
			return delta < 0 ? minValue : maxValue;
		}
		return expandedValue;
#endif
	}

	return request.seed;
}
