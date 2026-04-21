#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum class ofxStableDiffusionInterpolationMode {
	Linear,
	Smooth,
	EaseIn,
	EaseOut,
	EaseInOut
};

struct ofxStableDiffusionPromptKeyframe {
	int frameNumber = 0;
	std::string prompt;
	float weight = 1.0f;

	ofxStableDiffusionPromptKeyframe() = default;
	ofxStableDiffusionPromptKeyframe(int frame, const std::string& value, float keyframeWeight = 1.0f)
		: frameNumber(frame), prompt(value), weight(keyframeWeight) {}
};

struct ofxStableDiffusionKeyframe {
	int frameNumber = 0;
	float cfgScale = -1.0f;
	float strength = -1.0f;
	int64_t seed = -1;
	std::string prompt;
	std::string negativePrompt;

	ofxStableDiffusionKeyframe() = default;
	explicit ofxStableDiffusionKeyframe(int frame) : frameNumber(frame) {}
};

struct ofxStableDiffusionVideoAnimationSettings {
	bool enablePromptInterpolation = false;
	std::vector<ofxStableDiffusionPromptKeyframe> promptKeyframes;
	ofxStableDiffusionInterpolationMode promptInterpolationMode = ofxStableDiffusionInterpolationMode::Smooth;

	bool enableParameterAnimation = false;
	std::vector<ofxStableDiffusionKeyframe> parameterKeyframes;
	ofxStableDiffusionInterpolationMode parameterInterpolationMode = ofxStableDiffusionInterpolationMode::Smooth;

	bool useSeedSequence = false;
	int64_t seedIncrement = 1;
};

inline float ofxStableDiffusionClampUnit(float value) {
	return std::max(0.0f, std::min(1.0f, value));
}

inline float ofxStableDiffusionApplyInterpolation(float t, ofxStableDiffusionInterpolationMode mode) {
	const float clamped = ofxStableDiffusionClampUnit(t);
	constexpr float pi = 3.14159265358979323846f;
	switch (mode) {
	case ofxStableDiffusionInterpolationMode::Linear:
		return clamped;
	case ofxStableDiffusionInterpolationMode::Smooth:
		return (1.0f - std::cos(clamped * pi)) * 0.5f;
	case ofxStableDiffusionInterpolationMode::EaseIn:
		return clamped * clamped;
	case ofxStableDiffusionInterpolationMode::EaseOut:
		return clamped * (2.0f - clamped);
	case ofxStableDiffusionInterpolationMode::EaseInOut:
		if (clamped < 0.5f) {
			return 4.0f * clamped * clamped * clamped;
		}
		return 0.5f * std::pow((2.0f * clamped) - 2.0f, 3.0f) + 1.0f;
	default:
		return clamped;
	}
}

inline float ofxStableDiffusionLerp(float a, float b, float t) {
	return a + ((b - a) * t);
}

template<typename T>
inline bool ofxStableDiffusionFindSurroundingKeyframes(
	const std::vector<T>& keyframes,
	int frameNumber,
	const T*& prevKeyframe,
	const T*& nextKeyframe) {
	prevKeyframe = nullptr;
	nextKeyframe = nullptr;

	for (const auto& keyframe : keyframes) {
		if (keyframe.frameNumber <= frameNumber &&
			(prevKeyframe == nullptr || keyframe.frameNumber > prevKeyframe->frameNumber)) {
			prevKeyframe = &keyframe;
		}
		if (keyframe.frameNumber >= frameNumber &&
			(nextKeyframe == nullptr || keyframe.frameNumber < nextKeyframe->frameNumber)) {
			nextKeyframe = &keyframe;
		}
	}

	if (prevKeyframe == nullptr && !keyframes.empty()) {
		prevKeyframe = &*std::min_element(
			keyframes.begin(),
			keyframes.end(),
			[](const T& left, const T& right) { return left.frameNumber < right.frameNumber; });
	}
	if (nextKeyframe == nullptr && !keyframes.empty()) {
		nextKeyframe = &*std::max_element(
			keyframes.begin(),
			keyframes.end(),
			[](const T& left, const T& right) { return left.frameNumber < right.frameNumber; });
	}

	return prevKeyframe != nullptr && nextKeyframe != nullptr;
}

inline float ofxStableDiffusionInterpolateParameter(
	float prevValue,
	float nextValue,
	int prevFrame,
	int nextFrame,
	int currentFrame,
	ofxStableDiffusionInterpolationMode mode) {
	if (prevFrame == nextFrame) {
		return prevValue;
	}

	const float t = static_cast<float>(currentFrame - prevFrame) /
		static_cast<float>(nextFrame - prevFrame);
	return ofxStableDiffusionLerp(
		prevValue,
		nextValue,
		ofxStableDiffusionApplyInterpolation(t, mode));
}

inline std::string ofxStableDiffusionInterpolatePrompts(
	const std::vector<ofxStableDiffusionPromptKeyframe>& keyframes,
	int currentFrame,
	ofxStableDiffusionInterpolationMode mode) {
	if (keyframes.empty()) {
		return "";
	}

	const ofxStableDiffusionPromptKeyframe* prev = nullptr;
	const ofxStableDiffusionPromptKeyframe* next = nullptr;
	if (!ofxStableDiffusionFindSurroundingKeyframes(keyframes, currentFrame, prev, next)) {
		return "";
	}

	if (prev == next || prev->frameNumber == currentFrame) {
		return prev->prompt;
	}
	if (next->frameNumber == currentFrame) {
		return next->prompt;
	}
	if (prev->prompt == next->prompt) {
		return prev->prompt;
	}

	const float rawT = static_cast<float>(currentFrame - prev->frameNumber) /
		static_cast<float>(next->frameNumber - prev->frameNumber);
	const float t = ofxStableDiffusionApplyInterpolation(rawT, mode);
	const float weight1 = (1.0f - t) * prev->weight;
	const float weight2 = t * next->weight;

	return "(" + prev->prompt + ":" + std::to_string(weight1) + ") AND (" +
		next->prompt + ":" + std::to_string(weight2) + ")";
}

inline float ofxStableDiffusionGetInterpolatedParameter(
	const std::vector<ofxStableDiffusionKeyframe>& keyframes,
	int currentFrame,
	ofxStableDiffusionInterpolationMode mode,
	const std::function<float(const ofxStableDiffusionKeyframe&)>& getter) {
	if (keyframes.empty()) {
		return -1.0f;
	}

	const ofxStableDiffusionKeyframe* prev = nullptr;
	const ofxStableDiffusionKeyframe* next = nullptr;
	if (!ofxStableDiffusionFindSurroundingKeyframes(keyframes, currentFrame, prev, next)) {
		return -1.0f;
	}

	const float prevValue = getter(*prev);
	const float nextValue = getter(*next);
	if (prevValue < 0.0f && nextValue < 0.0f) {
		return -1.0f;
	}
	if (prevValue < 0.0f) {
		return nextValue;
	}
	if (nextValue < 0.0f) {
		return prevValue;
	}
	return ofxStableDiffusionInterpolateParameter(
		prevValue,
		nextValue,
		prev->frameNumber,
		next->frameNumber,
		currentFrame,
		mode);
}

template<typename Getter>
inline std::string ofxStableDiffusionGetKeyframedString(
	const std::vector<ofxStableDiffusionKeyframe>& keyframes,
	int currentFrame,
	Getter getter) {
	const ofxStableDiffusionKeyframe* bestPrev = nullptr;
	const ofxStableDiffusionKeyframe* bestNext = nullptr;
	for (const auto& keyframe : keyframes) {
		const std::string value = getter(keyframe);
		if (value.empty()) {
			continue;
		}
		if (keyframe.frameNumber <= currentFrame &&
			(bestPrev == nullptr || keyframe.frameNumber > bestPrev->frameNumber)) {
			bestPrev = &keyframe;
		}
		if (keyframe.frameNumber >= currentFrame &&
			(bestNext == nullptr || keyframe.frameNumber < bestNext->frameNumber)) {
			bestNext = &keyframe;
		}
	}

	if (bestPrev != nullptr) {
		return getter(*bestPrev);
	}
	if (bestNext != nullptr) {
		return getter(*bestNext);
	}
	return "";
}

inline int64_t ofxStableDiffusionGetKeyframedSeed(
	const std::vector<ofxStableDiffusionKeyframe>& keyframes,
	int currentFrame) {
	const ofxStableDiffusionKeyframe* bestPrev = nullptr;
	const ofxStableDiffusionKeyframe* bestNext = nullptr;
	for (const auto& keyframe : keyframes) {
		if (keyframe.seed < 0) {
			continue;
		}
		if (keyframe.frameNumber <= currentFrame &&
			(bestPrev == nullptr || keyframe.frameNumber > bestPrev->frameNumber)) {
			bestPrev = &keyframe;
		}
		if (keyframe.frameNumber >= currentFrame &&
			(bestNext == nullptr || keyframe.frameNumber < bestNext->frameNumber)) {
			bestNext = &keyframe;
		}
	}

	if (bestPrev != nullptr) {
		return bestPrev->seed;
	}
	if (bestNext != nullptr) {
		return bestNext->seed;
	}
	return -1;
}
