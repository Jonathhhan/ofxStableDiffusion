#pragma once

#include "../core/ofxStableDiffusionTypes.h"

#include <algorithm>
#include <string>
#include <vector>

enum class ofxStableDiffusionVideoWorkflowPreset {
	FastPreview = 0,
	LowVram,
	Balanced,
	Quality,
	BatchStoryboard,
	// Quick preview modes
	QuickPreview_8,   // Every 8th frame at 256x
	QuickPreview_4,   // Every 4th frame at 384x
	QuickPreview_2,   // Every 2nd frame at 512x
	// Storyboard modes
	Storyboard_6,     // 6 evenly-spaced keyframes
	Storyboard_12,    // 12 evenly-spaced keyframes
	// High quality modes
	HighQuality_24fps,
	HighQuality_30fps
};

struct ofxStableDiffusionVideoPromptValidation {
	bool ok = true;
	std::vector<std::string> errors;
	std::vector<std::string> warnings;
};

struct ofxStableDiffusionVideoProgressInfo {
	int currentFrame = 0;
	int totalFrames = 0;
	int currentStep = 0;
	int totalSteps = 0;
	float elapsedSeconds = 0.0f;
	float estimatedRemainingSeconds = 0.0f;
	float percentComplete = 0.0f;

	std::string formatETA() const {
		if (estimatedRemainingSeconds <= 0.0f) {
			return "calculating...";
		}
		const int mins = static_cast<int>(estimatedRemainingSeconds) / 60;
		const int secs = static_cast<int>(estimatedRemainingSeconds) % 60;
		if (mins > 0) {
			return std::to_string(mins) + "m " + std::to_string(secs) + "s";
		}
		return std::to_string(secs) + "s";
	}

	std::string formatProgress() const {
		return "Frame " + std::to_string(currentFrame) + "/" +
		       std::to_string(totalFrames) + " (" +
		       std::to_string(static_cast<int>(percentComplete * 100)) + "%)";
	}
};

inline ofxStableDiffusionVideoProgressInfo ofxStableDiffusionCalculateVideoProgress(
	int currentStep,
	int stepsPerFrame,
	int currentFrame,
	int totalFrames,
	float elapsedSeconds) {
	ofxStableDiffusionVideoProgressInfo info;
	info.currentFrame = currentFrame;
	info.totalFrames = totalFrames;
	info.currentStep = currentStep;
	info.totalSteps = totalFrames * stepsPerFrame;
	info.elapsedSeconds = elapsedSeconds;

	if (currentStep > 0 && info.totalSteps > 0) {
		info.percentComplete = static_cast<float>(currentStep) / static_cast<float>(info.totalSteps);
		if (elapsedSeconds > 0.0f && info.percentComplete > 0.01f) {
			const float totalEstimatedTime = elapsedSeconds / info.percentComplete;
			info.estimatedRemainingSeconds = totalEstimatedTime - elapsedSeconds;
		}
	}

	return info;
}


inline const char * ofxStableDiffusionVideoWorkflowPresetLabel(
	ofxStableDiffusionVideoWorkflowPreset preset) {
	switch (preset) {
	case ofxStableDiffusionVideoWorkflowPreset::FastPreview: return "FastPreview";
	case ofxStableDiffusionVideoWorkflowPreset::LowVram: return "LowVram";
	case ofxStableDiffusionVideoWorkflowPreset::Balanced: return "Balanced";
	case ofxStableDiffusionVideoWorkflowPreset::Quality: return "Quality";
	case ofxStableDiffusionVideoWorkflowPreset::BatchStoryboard: return "BatchStoryboard";
	case ofxStableDiffusionVideoWorkflowPreset::QuickPreview_8: return "QuickPreview_8";
	case ofxStableDiffusionVideoWorkflowPreset::QuickPreview_4: return "QuickPreview_4";
	case ofxStableDiffusionVideoWorkflowPreset::QuickPreview_2: return "QuickPreview_2";
	case ofxStableDiffusionVideoWorkflowPreset::Storyboard_6: return "Storyboard_6";
	case ofxStableDiffusionVideoWorkflowPreset::Storyboard_12: return "Storyboard_12";
	case ofxStableDiffusionVideoWorkflowPreset::HighQuality_24fps: return "HighQuality_24fps";
	case ofxStableDiffusionVideoWorkflowPreset::HighQuality_30fps: return "HighQuality_30fps";
	default:
		return "Balanced";
	}
}

inline void ofxStableDiffusionApplyVideoWorkflowPreset(
	ofxStableDiffusionVideoRequest * request,
	ofxStableDiffusionVideoWorkflowPreset preset) {
	if (!request) {
		return;
	}

	switch (preset) {
	case ofxStableDiffusionVideoWorkflowPreset::FastPreview:
		request->frameCount = 8;
		request->fps = 8;
		request->sampleSteps = 12;
		request->cfgScale = 5.0f;
		request->strength = 0.55f;
		break;
	case ofxStableDiffusionVideoWorkflowPreset::LowVram:
		request->frameCount = 10;
		request->fps = 8;
		request->sampleSteps = 14;
		request->cfgScale = 5.5f;
		request->strength = 0.6f;
		request->width = std::min(request->width, 512);
		request->height = std::min(request->height, 768);
		break;
	case ofxStableDiffusionVideoWorkflowPreset::Balanced:
		request->frameCount = 12;
		request->fps = 10;
		request->sampleSteps = 20;
		request->cfgScale = 6.5f;
		request->strength = 0.65f;
		break;
	case ofxStableDiffusionVideoWorkflowPreset::Quality:
		request->frameCount = 16;
		request->fps = 12;
		request->sampleSteps = 28;
		request->cfgScale = 7.0f;
		request->strength = 0.7f;
		break;
	case ofxStableDiffusionVideoWorkflowPreset::BatchStoryboard:
		request->frameCount = 6;
		request->fps = 6;
		request->sampleSteps = 16;
		request->cfgScale = 6.0f;
		request->strength = 0.6f;
		break;
	// Quick preview modes - reduce quality for fast iteration
	case ofxStableDiffusionVideoWorkflowPreset::QuickPreview_8:
		request->frameCount = std::max(8, request->frameCount / 8);
		request->fps = 4;
		request->sampleSteps = 8;
		request->cfgScale = 4.0f;
		request->strength = 0.5f;
		request->width = 256;
		request->height = 384;
		break;
	case ofxStableDiffusionVideoWorkflowPreset::QuickPreview_4:
		request->frameCount = std::max(4, request->frameCount / 4);
		request->fps = 6;
		request->sampleSteps = 10;
		request->cfgScale = 4.5f;
		request->strength = 0.5f;
		request->width = 384;
		request->height = 576;
		break;
	case ofxStableDiffusionVideoWorkflowPreset::QuickPreview_2:
		request->frameCount = std::max(4, request->frameCount / 2);
		request->fps = 8;
		request->sampleSteps = 12;
		request->cfgScale = 5.0f;
		request->strength = 0.55f;
		request->width = 512;
		request->height = 768;
		break;
	// Storyboard modes - generate keyframes only
	case ofxStableDiffusionVideoWorkflowPreset::Storyboard_6:
		request->frameCount = 6;
		request->fps = 1;  // Static keyframes
		request->sampleSteps = 20;
		request->cfgScale = 6.5f;
		request->strength = 0.6f;
		break;
	case ofxStableDiffusionVideoWorkflowPreset::Storyboard_12:
		request->frameCount = 12;
		request->fps = 2;  // Static keyframes
		request->sampleSteps = 20;
		request->cfgScale = 6.5f;
		request->strength = 0.6f;
		break;
	// High quality modes
	case ofxStableDiffusionVideoWorkflowPreset::HighQuality_24fps:
		request->frameCount = std::min(48, request->frameCount);
		request->fps = 24;
		request->sampleSteps = 35;
		request->cfgScale = 7.5f;
		request->strength = 0.75f;
		break;
	case ofxStableDiffusionVideoWorkflowPreset::HighQuality_30fps:
		request->frameCount = std::min(60, request->frameCount);
		request->fps = 30;
		request->sampleSteps = 40;
		request->cfgScale = 8.0f;
		request->strength = 0.8f;
		break;
	}
}

inline ofxStableDiffusionVideoPromptValidation ofxStableDiffusionValidateVideoPromptWorkflow(
	const ofxStableDiffusionVideoRequest & request) {
	ofxStableDiffusionVideoPromptValidation validation;
	const std::string prompt = ofTrim(request.prompt);
	if (prompt.empty()) {
		validation.ok = false;
		validation.errors.push_back("Video prompt is empty.");
	}
	if (request.initImage.data == nullptr) {
		validation.ok = false;
		validation.errors.push_back("Video generation requires an initial image.");
	}
	if (request.frameCount <= 0) {
		validation.ok = false;
		validation.errors.push_back("Frame count must be greater than zero.");
	}
	if (request.frameCount > 300) {
		validation.ok = false;
		validation.errors.push_back("Frame count exceeds maximum of 300 frames.");
	}
	if (request.fps <= 0) {
		validation.ok = false;
		validation.errors.push_back("FPS must be greater than zero.");
	}
	if (request.width <= 0 || request.height <= 0) {
		validation.ok = false;
		validation.errors.push_back("Output width and height must be greater than zero.");
	}
	// Dimension validation with auto-correction suggestions
	if (request.width % 64 != 0 || request.height % 64 != 0) {
		validation.ok = false;
		const int suggestedWidth = ((request.width + 31) / 64) * 64;
		const int suggestedHeight = ((request.height + 31) / 64) * 64;
		validation.errors.push_back(
			"Width and height must be multiples of 64. Suggestion: " +
			std::to_string(suggestedWidth) + "x" + std::to_string(suggestedHeight));
	}
	// Temporal coherence validation
	if (request.animationSettings.temporalCoherence < 0.0f ||
	    request.animationSettings.temporalCoherence > 1.0f) {
		validation.warnings.push_back(
			"Temporal coherence should be between 0.0 and 1.0 (currently " +
			std::to_string(request.animationSettings.temporalCoherence) + ").");
	}
	// Frame count optimization suggestions
	if (request.frameCount > 100 && request.sampleSteps > 25) {
		validation.warnings.push_back(
			"Large frame count (" + std::to_string(request.frameCount) +
			") with high sample steps (" + std::to_string(request.sampleSteps) +
			") will take significant time. Consider reducing steps or using a preset.");
	}
	if (prompt.find("dialog") == std::string::npos &&
		prompt.find("speak") == std::string::npos &&
		prompt.find("sing") == std::string::npos &&
		prompt.find("voice") == std::string::npos) {
		validation.warnings.push_back(
			"Prompt does not mention speech, singing, or dialogue. Audio/video sync workflows may look better with explicit voice intent.");
	}
	if (ofTrim(request.negativePrompt).empty()) {
		validation.warnings.push_back(
			"Negative prompt is empty. Video workflows usually benefit from motion and artifact constraints.");
	}
	if (request.frameCount < 8) {
		validation.warnings.push_back(
			"Very short frame counts can make story or lip-sync continuity harder to judge.");
	}
	if (request.fps > 24) {
		validation.warnings.push_back(
			"High FPS increases render cost quickly. Consider 6-12 FPS for exploration presets.");
	}
	return validation;
}

inline ofJson ofxStableDiffusionBuildVideoRenderManifest(
	const ofxStableDiffusionVideoRequest & request,
	const ofxStableDiffusionVideoClip & clip,
	ofxStableDiffusionVideoWorkflowPreset preset,
	const std::string & outputDirectory = {},
	const std::string & metadataPath = {}) {
	ofJson root;
	root["preset"] = ofxStableDiffusionVideoWorkflowPresetLabel(preset);
	root["prompt"] = request.prompt;
	root["negative_prompt"] = request.negativePrompt;
	root["width"] = request.width;
	root["height"] = request.height;
	root["frame_count"] = request.frameCount;
	root["fps"] = request.fps;
	root["cfg_scale"] = request.cfgScale;
	root["sample_steps"] = request.sampleSteps;
	root["strength"] = request.strength;
	root["seed"] = request.seed;
	root["mode"] = ofxStableDiffusionVideoModeLabel(request.mode);
	root["has_end_image"] = (request.endImage.data != nullptr);
	root["has_animation"] = request.hasAnimation();
	root["clip_duration_seconds"] = clip.durationSeconds();
	root["rendered_frame_count"] = static_cast<int>(clip.frames.size());
	if (!outputDirectory.empty()) {
		root["output_directory"] = outputDirectory;
	}
	if (!metadataPath.empty()) {
		root["metadata_path"] = metadataPath;
	}

	ofJson loras = ofJson::array();
	for (const auto & lora : request.loras) {
		if (!lora.isValid()) {
			continue;
		}
		loras.push_back({
			{"path", lora.path},
			{"strength", lora.strength},
			{"high_noise", lora.isHighNoise}
		});
	}
	root["loras"] = std::move(loras);

	const auto validation = ofxStableDiffusionValidateVideoPromptWorkflow(request);
	root["validation"]["ok"] = validation.ok;
	root["validation"]["errors"] = validation.errors;
	root["validation"]["warnings"] = validation.warnings;
	return root;
}

inline bool ofxStableDiffusionSaveVideoRenderManifest(
	const std::string & path,
	const ofxStableDiffusionVideoRequest & request,
	const ofxStableDiffusionVideoClip & clip,
	ofxStableDiffusionVideoWorkflowPreset preset,
	const std::string & outputDirectory = {},
	const std::string & metadataPath = {}) {
	return ofSavePrettyJson(
		path,
		ofxStableDiffusionBuildVideoRenderManifest(
			request,
			clip,
			preset,
			outputDirectory,
			metadataPath));
}

// Dry-run estimation structure
struct ofxStableDiffusionVideoDryRunEstimate {
	int totalFrames = 0;
	int totalSteps = 0;
	float estimatedMinutes = 0.0f;
	float estimatedMemoryMB = 0.0f;
	std::string recommendation;
	bool feasible = true;
	std::vector<std::string> warnings;
};

// Estimate render time and resources for dry-run validation
inline ofxStableDiffusionVideoDryRunEstimate ofxStableDiffusionEstimateVideoRender(
	const ofxStableDiffusionVideoRequest & request,
	float avgSecondsPerStep = 0.5f) {

	ofxStableDiffusionVideoDryRunEstimate estimate;
	estimate.totalFrames = request.frameCount;
	estimate.totalSteps = request.frameCount * request.sampleSteps;

	// Estimate render time (conservative)
	estimate.estimatedMinutes = (estimate.totalSteps * avgSecondsPerStep) / 60.0f;

	// Estimate memory usage (rough approximation)
	const float pixelCount = request.width * request.height;
	const float baseMemoryMB = (pixelCount * 3 * 4) / (1024.0f * 1024.0f); // RGBA float
	estimate.estimatedMemoryMB = baseMemoryMB * 1.5f * request.frameCount; // With overhead

	// Generate recommendation
	if (estimate.estimatedMinutes < 2.0f) {
		estimate.recommendation = "Fast render - should complete quickly";
	} else if (estimate.estimatedMinutes < 10.0f) {
		estimate.recommendation = "Moderate render time - good for iteration";
	} else if (estimate.estimatedMinutes < 30.0f) {
		estimate.recommendation = "Long render - consider using a preview preset first";
	} else {
		estimate.recommendation = "Very long render - strongly recommend preview mode";
		estimate.warnings.push_back("Estimated time exceeds 30 minutes");
	}

	// Check feasibility
	if (estimate.estimatedMemoryMB > 8000.0f) {
		estimate.feasible = false;
		estimate.warnings.push_back("High memory usage may cause OOM errors");
	}

	if (request.frameCount > 200 && request.sampleSteps > 30) {
		estimate.warnings.push_back("High frame count with high steps - consider reducing");
	}

	return estimate;
}

// Preset composition - combine multiple presets
inline ofxStableDiffusionVideoRequest ofxStableDiffusionComposePresets(
	const ofxStableDiffusionVideoRequest & baseRequest,
	const std::vector<ofxStableDiffusionVideoWorkflowPreset> & presets) {

	ofxStableDiffusionVideoRequest composed = baseRequest;

	// Apply each preset in sequence, allowing later presets to override
	for (const auto & preset : presets) {
		ofxStableDiffusionApplyVideoWorkflowPreset(&composed, preset);
	}

	return composed;
}

// Video template system - reusable animation patterns
struct ofxStableDiffusionVideoTemplate {
	std::string name;
	std::string description;
	ofxStableDiffusionVideoRequest baseRequest;
	std::vector<ofxStableDiffusionPromptKeyframe> promptKeyframes;
	std::vector<ofxStableDiffusionKeyframe> parameterKeyframes;
};

// Create a simple fade template
inline ofxStableDiffusionVideoTemplate ofxStableDiffusionCreateFadeTemplate(
	const std::string & startPrompt,
	const std::string & endPrompt,
	int frameCount = 24) {

	ofxStableDiffusionVideoTemplate tmpl;
	tmpl.name = "Fade";
	tmpl.description = "Smooth fade between two prompts";

	tmpl.baseRequest.frameCount = frameCount;
	tmpl.baseRequest.fps = 12;
	tmpl.baseRequest.sampleSteps = 20;

	tmpl.promptKeyframes.push_back({0, startPrompt, 1.0f});
	tmpl.promptKeyframes.push_back({frameCount - 1, endPrompt, 1.0f});

	tmpl.baseRequest.animationSettings.enablePromptInterpolation = true;
	tmpl.baseRequest.animationSettings.promptKeyframes = tmpl.promptKeyframes;
	tmpl.baseRequest.animationSettings.promptInterpolationMode =
		ofxStableDiffusionInterpolationMode::Smooth;

	return tmpl;
}

// Create a pulse template (varying strength)
inline ofxStableDiffusionVideoTemplate ofxStableDiffusionCreatePulseTemplate(
	int frameCount = 24,
	float minStrength = 0.4f,
	float maxStrength = 0.8f) {

	ofxStableDiffusionVideoTemplate tmpl;
	tmpl.name = "Pulse";
	tmpl.description = "Pulsing strength variation";

	tmpl.baseRequest.frameCount = frameCount;
	tmpl.baseRequest.fps = 12;
	tmpl.baseRequest.sampleSteps = 20;

	// Create pulse pattern
	ofxStableDiffusionKeyframe keyframe0(0);
	keyframe0.strength = minStrength;
	tmpl.parameterKeyframes.push_back(keyframe0);

	ofxStableDiffusionKeyframe keyframeMid(frameCount / 2);
	keyframeMid.strength = maxStrength;
	tmpl.parameterKeyframes.push_back(keyframeMid);

	ofxStableDiffusionKeyframe keyframeLast(frameCount - 1);
	keyframeLast.strength = minStrength;
	tmpl.parameterKeyframes.push_back(keyframeLast);

	tmpl.baseRequest.animationSettings.enableParameterAnimation = true;
	tmpl.baseRequest.animationSettings.parameterKeyframes = tmpl.parameterKeyframes;
	tmpl.baseRequest.animationSettings.parameterInterpolationMode =
		ofxStableDiffusionInterpolationMode::Smooth;

	return tmpl;
}

// Apply a template to a request
inline ofxStableDiffusionVideoRequest ofxStableDiffusionApplyTemplate(
	const ofxStableDiffusionVideoRequest & request,
	const ofxStableDiffusionVideoTemplate & tmpl) {

	ofxStableDiffusionVideoRequest result = request;

	// Merge template settings
	result.frameCount = tmpl.baseRequest.frameCount;
	result.fps = tmpl.baseRequest.fps;
	result.sampleSteps = tmpl.baseRequest.sampleSteps;

	// Merge animation settings
	if (!tmpl.promptKeyframes.empty()) {
		result.animationSettings.enablePromptInterpolation = true;
		result.animationSettings.promptKeyframes = tmpl.promptKeyframes;
		result.animationSettings.promptInterpolationMode =
			tmpl.baseRequest.animationSettings.promptInterpolationMode;
	}

	if (!tmpl.parameterKeyframes.empty()) {
		result.animationSettings.enableParameterAnimation = true;
		result.animationSettings.parameterKeyframes = tmpl.parameterKeyframes;
		result.animationSettings.parameterInterpolationMode =
			tmpl.baseRequest.animationSettings.parameterInterpolationMode;
	}

	return result;
}
