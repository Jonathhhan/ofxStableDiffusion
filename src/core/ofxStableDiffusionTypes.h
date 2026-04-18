#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionEnums.h"
#include "ofxStableDiffusionImageHelpers.h"
#include "ofxStableDiffusionRankingHelpers.h"
#include "../../libs/stable-diffusion/include/stable-diffusion.h"
#include "../video/ofxStableDiffusionVideoAnimation.h"

#include <cstdint>
#include <string>
#include <vector>

struct ofxStableDiffusionError {
	ofxStableDiffusionErrorCode code = ofxStableDiffusionErrorCode::None;
	std::string message;
	std::string suggestion;
	uint64_t timestampMicros = 0;

	bool hasError() const {
		return code != ofxStableDiffusionErrorCode::None;
	}
};

struct ofxStableDiffusionContextSettings {
	// Primary model path (single-file models: SD1.x, SDXL, SD3, FLUX unified gguf)
	std::string modelPath;
	// Split model paths – used when the model is distributed across separate files
	// (e.g. FLUX: set diffusionModelPath + clipLPath + t5xxlPath instead of modelPath)
	std::string diffusionModelPath;
	std::string clipLPath;
	std::string clipGPath;
	std::string t5xxlPath;
	// VAE / accelerators
	std::string vaePath;
	std::string taesdPath;
	// ControlNet
	std::string controlNetPath;
	// LoRA search directory (legacy; LoRAs are now applied per-generation via loras field)
	std::string loraModelDir;
	// Text-embedding / photo-maker paths
	std::string embedDir;
	std::string stackedIdEmbedDir;
	// Context flags
	bool vaeDecodeOnly = false;
	bool vaeTiling = false;
	bool freeParamsImmediately = false;
	int nThreads = -1;
	sd_type_t weightType = SD_TYPE_F16;
	rng_type_t rngType = STD_DEFAULT_RNG;
	scheduler_t schedule = SCHEDULER_COUNT;
	// Model behaviour
	prediction_t prediction = EPS_PRED;
	lora_apply_mode_t loraApplyMode = LORA_APPLY_AUTO;
	// CPU offload
	bool keepClipOnCpu = false;
	bool keepControlNetCpu = false;
	bool keepVaeOnCpu = false;
	bool offloadParamsToCpu = false;
	// Performance
	bool flashAttn = false;
	bool enableMmap = true;
};

struct ofxStableDiffusionLora {
	std::string path;
	float strength = 1.0f;
	bool isHighNoise = false;

	bool isValid() const {
		return !path.empty();
	}
};

struct ofxStableDiffusionUpscalerSettings {
	std::string modelPath;
	int nThreads = -1;
	sd_type_t weightType = SD_TYPE_F16;
	int multiplier = 4;
	bool enabled = false;
};

struct ofxStableDiffusionImageRequest {
	ofxStableDiffusionImageMode mode = ofxStableDiffusionImageMode::TextToImage;
	ofxStableDiffusionImageSelectionMode selectionMode =
		ofxStableDiffusionImageSelectionMode::KeepOrder;
	sd_image_t initImage{0, 0, 0, nullptr};
	sd_image_t maskImage{0, 0, 0, nullptr};  // For inpainting (white=inpaint, black=keep)
	std::string prompt;
	std::string instruction;
	std::string negativePrompt;
	int clipSkip = -1;
	float cfgScale = 7.0f;
	int width = 512;
	int height = 512;
	sample_method_t sampleMethod = EULER_A_SAMPLE_METHOD;
	int sampleSteps = 20;
	float strength = 0.5f;
	int64_t seed = -1;
	int batchCount = 1;
	sd_image_t * controlCond = nullptr;
	float controlStrength = 0.9f;
	float styleStrength = 20.0f;
	float maskBlur = 4.0f;  // Blur radius for mask edge softening
	bool normalizeInput = true;
	std::string inputIdImagesPath;
	std::vector<ofxStableDiffusionLora> loras;
};

struct ofxStableDiffusionVideoRequest {
	sd_image_t initImage{0, 0, 0, nullptr};
	// Optional end frame for video morphing (leave zeroed to disable)
	sd_image_t endImage{0, 0, 0, nullptr};
	std::string prompt;
	std::string negativePrompt;
	int clipSkip = -1;
	int width = 576;
	int height = 1024;
	int frameCount = 6;
	int fps = 6;
	float cfgScale = 7.0f;
	sample_method_t sampleMethod = EULER_A_SAMPLE_METHOD;
	int sampleSteps = 20;
	float strength = 0.5f;
	int64_t seed = -1;
	// VACE control strength (0 = disabled, 1 = full)
	float vaceStrength = 1.0f;
	ofxStableDiffusionVideoMode mode = ofxStableDiffusionVideoMode::Standard;
	std::vector<ofxStableDiffusionLora> loras;

	// Animation settings (optional)
	ofxStableDiffusionVideoAnimationSettings animationSettings;

	// Helper to check if animation is enabled
	bool hasAnimation() const {
		return animationSettings.enablePromptInterpolation ||
		       animationSettings.enableParameterAnimation ||
		       animationSettings.useSeedSequence;
	}
};

struct ofxStableDiffusionImageFrame {
	ofPixels pixels;
	int index = 0;
	int sourceIndex = 0;
	int seed = -1;
	bool isSelected = false;
	ofxStableDiffusionImageScore score;

	bool isAllocated() const {
		return pixels.isAllocated();
	}

	int width() const {
		return static_cast<int>(pixels.getWidth());
	}

	int height() const {
		return static_cast<int>(pixels.getHeight());
	}

	int channels() const {
		return static_cast<int>(pixels.getNumChannels());
	}
};

struct ofxStableDiffusionVideoClip {
	std::vector<ofxStableDiffusionImageFrame> frames;
	int fps = 6;
	int sourceFrameCount = 0;
	ofxStableDiffusionVideoMode mode = ofxStableDiffusionVideoMode::Standard;

	bool empty() const;
	std::size_t size() const;
	float durationSeconds() const;
	int frameIndexForTime(float seconds) const;
	const ofxStableDiffusionImageFrame * frameForTime(float seconds) const;
	bool saveFrameSequence(
		const std::string & directory,
		const std::string & prefix = "frame") const;
};

struct ofxStableDiffusionResult {
	bool success = false;
	ofxStableDiffusionTask task = ofxStableDiffusionTask::None;
	ofxStableDiffusionImageMode imageMode = ofxStableDiffusionImageMode::TextToImage;
	ofxStableDiffusionImageSelectionMode selectionMode =
		ofxStableDiffusionImageSelectionMode::KeepOrder;
	float elapsedMs = 0.0f;
	bool rankingApplied = false;
	int selectedImageIndex = -1;
	int64_t actualSeedUsed = -1;
	std::string error;
	std::vector<ofxStableDiffusionImageFrame> images;
	ofxStableDiffusionVideoClip video;

	bool hasImages() const {
		return !images.empty();
	}

	bool hasVideo() const {
		return !video.frames.empty();
	}
};

const char * ofxStableDiffusionTaskLabel(ofxStableDiffusionTask task);
const char * ofxStableDiffusionImageModeLabel(ofxStableDiffusionImageMode mode);
const char * ofxStableDiffusionImageSelectionModeLabel(
	ofxStableDiffusionImageSelectionMode mode);
const char * ofxStableDiffusionVideoModeLabel(ofxStableDiffusionVideoMode mode);
const char * ofxStableDiffusionErrorCodeLabel(ofxStableDiffusionErrorCode code);
std::string ofxStableDiffusionErrorCodeSuggestion(ofxStableDiffusionErrorCode code);
std::vector<ofxStableDiffusionImageFrame> ofxStableDiffusionBuildVideoFrames(
	const std::vector<ofxStableDiffusionImageFrame> & sourceFrames,
	ofxStableDiffusionVideoMode mode);

/// Hash a string to a deterministic seed value for reproducibility.
int64_t ofxStableDiffusionHashStringToSeed(const std::string& text);
