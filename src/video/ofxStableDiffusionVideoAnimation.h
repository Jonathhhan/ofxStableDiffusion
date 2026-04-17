#pragma once

#include "ofMain.h"
#include "../core/ofxStableDiffusionEnums.h"
#include <string>
#include <vector>
#include <map>

/// Interpolation mode for animation parameters
enum class ofxStableDiffusionInterpolationMode {
	Linear,         // Linear interpolation
	Smooth,         // Smooth (cosine) interpolation
	EaseIn,         // Ease in (accelerating)
	EaseOut,        // Ease out (decelerating)
	EaseInOut       // Ease in-out (S-curve)
};

/// Prompt keyframe for prompt interpolation
struct ofxStableDiffusionPromptKeyframe {
	int frameNumber = 0;
	std::string prompt;
	float weight = 1.0f;  // Blend weight at this keyframe

	ofxStableDiffusionPromptKeyframe() = default;
	ofxStableDiffusionPromptKeyframe(int frame, const std::string& p, float w = 1.0f)
		: frameNumber(frame), prompt(p), weight(w) {}
};

/// General parameter keyframe for animation
struct ofxStableDiffusionKeyframe {
	int frameNumber = 0;

	// Optional parameters (use -1 or empty to indicate not set)
	float cfgScale = -1.0f;
	float strength = -1.0f;
	int64_t seed = -1;
	std::string prompt;
	std::string negativePrompt;

	ofxStableDiffusionKeyframe() = default;
	explicit ofxStableDiffusionKeyframe(int frame) : frameNumber(frame) {}
};

/// Video animation settings
struct ofxStableDiffusionVideoAnimationSettings {
	// Prompt interpolation
	bool enablePromptInterpolation = false;
	std::vector<ofxStableDiffusionPromptKeyframe> promptKeyframes;
	ofxStableDiffusionInterpolationMode promptInterpolationMode = ofxStableDiffusionInterpolationMode::Smooth;

	// Parameter animation
	bool enableParameterAnimation = false;
	std::vector<ofxStableDiffusionKeyframe> parameterKeyframes;
	ofxStableDiffusionInterpolationMode parameterInterpolationMode = ofxStableDiffusionInterpolationMode::Smooth;

	// Seed variation
	bool useSeedSequence = false;
	int64_t seedIncrement = 1;  // Increment seed by this value each frame

	// Export settings
	bool embedMetadata = true;
	bool exportParametersJson = true;
	std::string outputDirectory;
};

//--------------------------------------------------------------
// Interpolation Helper Functions
//--------------------------------------------------------------

/// Apply interpolation factor based on mode
inline float ofxStableDiffusionApplyInterpolation(float t, ofxStableDiffusionInterpolationMode mode) {
	switch (mode) {
		case ofxStableDiffusionInterpolationMode::Linear:
			return t;

		case ofxStableDiffusionInterpolationMode::Smooth:
			// Cosine interpolation
			return (1.0f - std::cos(t * PI)) * 0.5f;

		case ofxStableDiffusionInterpolationMode::EaseIn:
			// Quadratic ease in
			return t * t;

		case ofxStableDiffusionInterpolationMode::EaseOut:
			// Quadratic ease out
			return t * (2.0f - t);

		case ofxStableDiffusionInterpolationMode::EaseInOut:
			// Cubic ease in-out
			if (t < 0.5f) {
				return 4.0f * t * t * t;
			} else {
				float f = (2.0f * t) - 2.0f;
				return 0.5f * f * f * f + 1.0f;
			}

		default:
			return t;
	}
}

/// Linear interpolation between two values
inline float ofxStableDiffusionLerp(float a, float b, float t) {
	return a + (b - a) * t;
}

/// Find keyframes surrounding a given frame
template<typename T>
inline bool ofxStableDiffusionFindSurroundingKeyframes(
	const std::vector<T>& keyframes,
	int frameNumber,
	const T*& prevKeyframe,
	const T*& nextKeyframe) {

	if (keyframes.empty()) {
		prevKeyframe = nullptr;
		nextKeyframe = nullptr;
		return false;
	}

	// Find the keyframes before and after the current frame
	prevKeyframe = nullptr;
	nextKeyframe = nullptr;

	for (size_t i = 0; i < keyframes.size(); ++i) {
		if (keyframes[i].frameNumber <= frameNumber) {
			prevKeyframe = &keyframes[i];
		}
		if (keyframes[i].frameNumber >= frameNumber && nextKeyframe == nullptr) {
			nextKeyframe = &keyframes[i];
			break;
		}
	}

	// If no next keyframe, use the last one
	if (nextKeyframe == nullptr && !keyframes.empty()) {
		nextKeyframe = &keyframes.back();
	}

	// If no prev keyframe, use the first one
	if (prevKeyframe == nullptr && !keyframes.empty()) {
		prevKeyframe = &keyframes.front();
	}

	return prevKeyframe != nullptr && nextKeyframe != nullptr;
}

/// Interpolate float parameter between keyframes
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

	float t = static_cast<float>(currentFrame - prevFrame) / (nextFrame - prevFrame);
	t = ofClamp(t, 0.0f, 1.0f);
	t = ofxStableDiffusionApplyInterpolation(t, mode);

	return ofxStableDiffusionLerp(prevValue, nextValue, t);
}

/// Interpolate prompts between keyframes (returns blended prompt string)
inline std::string ofxStableDiffusionInterpolatePrompts(
	const std::vector<ofxStableDiffusionPromptKeyframe>& keyframes,
	int currentFrame,
	ofxStableDiffusionInterpolationMode mode) {

	if (keyframes.empty()) {
		return "";
	}

	// Find surrounding keyframes
	const ofxStableDiffusionPromptKeyframe* prev = nullptr;
	const ofxStableDiffusionPromptKeyframe* next = nullptr;

	if (!ofxStableDiffusionFindSurroundingKeyframes(keyframes, currentFrame, prev, next)) {
		return "";
	}

	// If we're exactly on a keyframe or both are the same, return that prompt
	if (prev == next || prev->frameNumber == currentFrame) {
		return prev->prompt;
	}
	if (next->frameNumber == currentFrame) {
		return next->prompt;
	}

	// Calculate interpolation factor
	float t = static_cast<float>(currentFrame - prev->frameNumber) /
	          (next->frameNumber - prev->frameNumber);
	t = ofClamp(t, 0.0f, 1.0f);
	t = ofxStableDiffusionApplyInterpolation(t, mode);

	// For prompt interpolation, we blend by using both prompts with weights
	// Format: "(prompt1:weight1) AND (prompt2:weight2)"
	float weight1 = 1.0f - t;
	float weight2 = t;

	// If prompts are the same, just return one
	if (prev->prompt == next->prompt) {
		return prev->prompt;
	}

	// Build blended prompt with weights
	std::string blended = "(";
	blended += prev->prompt;
	blended += ":";
	blended += std::to_string(weight1);
	blended += ") AND (";
	blended += next->prompt;
	blended += ":";
	blended += std::to_string(weight2);
	blended += ")";

	return blended;
}

/// Get interpolated parameter value for a given frame
inline float ofxStableDiffusionGetInterpolatedParameter(
	const std::vector<ofxStableDiffusionKeyframe>& keyframes,
	int currentFrame,
	ofxStableDiffusionInterpolationMode mode,
	std::function<float(const ofxStableDiffusionKeyframe&)> paramGetter) {

	if (keyframes.empty()) {
		return -1.0f;
	}

	const ofxStableDiffusionKeyframe* prev = nullptr;
	const ofxStableDiffusionKeyframe* next = nullptr;

	if (!ofxStableDiffusionFindSurroundingKeyframes(keyframes, currentFrame, prev, next)) {
		return -1.0f;
	}

	float prevValue = paramGetter(*prev);
	float nextValue = paramGetter(*next);

	// If parameter not set in keyframes, return -1
	if (prevValue < 0.0f && nextValue < 0.0f) {
		return -1.0f;
	}

	// If one is not set, use the other
	if (prevValue < 0.0f) return nextValue;
	if (nextValue < 0.0f) return prevValue;

	return ofxStableDiffusionInterpolateParameter(
		prevValue, nextValue,
		prev->frameNumber, next->frameNumber,
		currentFrame, mode
	);
}

//--------------------------------------------------------------
// Export Helper Functions
//--------------------------------------------------------------

/// Generate JSON metadata for a video frame
inline ofJson ofxStableDiffusionGenerateFrameMetadata(
	int frameNumber,
	const std::string& prompt,
	const std::string& negativePrompt,
	float cfgScale,
	float strength,
	int64_t seed,
	int width,
	int height,
	const std::string& modelPath) {

	ofJson metadata;
	metadata["frame"] = frameNumber;
	metadata["prompt"] = prompt;
	metadata["negative_prompt"] = negativePrompt;
	metadata["cfg_scale"] = cfgScale;
	metadata["strength"] = strength;
	metadata["seed"] = seed;
	metadata["width"] = width;
	metadata["height"] = height;
	metadata["model"] = modelPath;
	metadata["timestamp"] = ofGetTimestampString();

	return metadata;
}

/// Export video generation parameters to JSON file
inline bool ofxStableDiffusionExportVideoParametersJson(
	const std::string& filepath,
	const ofxStableDiffusionVideoAnimationSettings& settings,
	const std::vector<ofJson>& frameMetadata) {

	ofJson root;
	root["version"] = "1.0";
	root["generated_at"] = ofGetTimestampString();

	// Animation settings
	ofJson animSettings;
	animSettings["prompt_interpolation_enabled"] = settings.enablePromptInterpolation;
	animSettings["parameter_animation_enabled"] = settings.enableParameterAnimation;
	animSettings["use_seed_sequence"] = settings.useSeedSequence;
	animSettings["seed_increment"] = settings.seedIncrement;
	root["animation_settings"] = animSettings;

	// Prompt keyframes
	if (!settings.promptKeyframes.empty()) {
		ofJson promptKfs = ofJson::array();
		for (const auto& kf : settings.promptKeyframes) {
			ofJson kfJson;
			kfJson["frame"] = kf.frameNumber;
			kfJson["prompt"] = kf.prompt;
			kfJson["weight"] = kf.weight;
			promptKfs.push_back(kfJson);
		}
		root["prompt_keyframes"] = promptKfs;
	}

	// Parameter keyframes
	if (!settings.parameterKeyframes.empty()) {
		ofJson paramKfs = ofJson::array();
		for (const auto& kf : settings.parameterKeyframes) {
			ofJson kfJson;
			kfJson["frame"] = kf.frameNumber;
			if (kf.cfgScale >= 0.0f) kfJson["cfg_scale"] = kf.cfgScale;
			if (kf.strength >= 0.0f) kfJson["strength"] = kf.strength;
			if (kf.seed >= 0) kfJson["seed"] = kf.seed;
			if (!kf.prompt.empty()) kfJson["prompt"] = kf.prompt;
			if (!kf.negativePrompt.empty()) kfJson["negative_prompt"] = kf.negativePrompt;
			paramKfs.push_back(kfJson);
		}
		root["parameter_keyframes"] = paramKfs;
	}

	// Frame metadata
	root["frames"] = frameMetadata;

	return ofSavePrettyJson(filepath, root);
}
