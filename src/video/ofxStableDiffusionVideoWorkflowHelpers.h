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
	BatchStoryboard
};

struct ofxStableDiffusionVideoPromptValidation {
	bool ok = true;
	std::vector<std::string> errors;
	std::vector<std::string> warnings;
};

inline const char * ofxStableDiffusionVideoWorkflowPresetLabel(
	ofxStableDiffusionVideoWorkflowPreset preset) {
	switch (preset) {
	case ofxStableDiffusionVideoWorkflowPreset::FastPreview: return "FastPreview";
	case ofxStableDiffusionVideoWorkflowPreset::LowVram: return "LowVram";
	case ofxStableDiffusionVideoWorkflowPreset::Balanced: return "Balanced";
	case ofxStableDiffusionVideoWorkflowPreset::Quality: return "Quality";
	case ofxStableDiffusionVideoWorkflowPreset::BatchStoryboard: return "BatchStoryboard";
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
	if (request.fps <= 0) {
		validation.ok = false;
		validation.errors.push_back("FPS must be greater than zero.");
	}
	if (request.width <= 0 || request.height <= 0) {
		validation.ok = false;
		validation.errors.push_back("Output width and height must be greater than zero.");
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
