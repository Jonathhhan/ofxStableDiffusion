#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionEnums.h"
#include "ofxStableDiffusionImageHelpers.h"
#include "ofxStableDiffusionRankingHelpers.h"
#include "../video/ofxStableDiffusionVideoAnimation.h"
#include "ofxStableDiffusionNativeApi.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

inline const char* ofxStableDiffusionCacheModeName(sd_cache_mode_t mode) {
	switch (mode) {
	case SD_CACHE_DISABLED: return "disabled";
	case SD_CACHE_EASYCACHE: return "easycache";
	case SD_CACHE_UCACHE: return "ucache";
	case SD_CACHE_DBCACHE: return "dbcache";
	case SD_CACHE_TAYLORSEER: return "taylorseer";
	case SD_CACHE_CACHE_DIT: return "cache-dit";
	case SD_CACHE_SPECTRUM: return "spectrum";
	default:
		return "disabled";
	}
}

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
	bool freeParamsImmediately = true;
	int nThreads = -1;
	sd_type_t weightType = SD_TYPE_F16;
	rng_type_t rngType = CUDA_RNG;
	scheduler_t schedule = SCHEDULER_COUNT;
	// Model behaviour
	prediction_t prediction = PREDICTION_COUNT;
	lora_apply_mode_t loraApplyMode = LORA_APPLY_AUTO;
	// CPU offload
	bool keepClipOnCpu = false;
	bool keepControlNetCpu = false;
	bool keepVaeOnCpu = false;
	bool offloadParamsToCpu = false;
	// Performance
	bool flashAttn = false;
	bool diffusionFlashAttn = false;
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

struct ofxStableDiffusionControlNet {
	sd_image_t conditionImage{0, 0, 0, nullptr};
	float strength = 0.9f;
	std::string type;  // Optional type hint: "canny", "depth", "pose", etc.

	bool isValid() const {
		return conditionImage.data != nullptr;
	}
};

struct ofxStableDiffusionUpscalerSettings {
	std::string modelPath;
	int nThreads = -1;
	sd_type_t weightType = SD_TYPE_F16;
	int multiplier = 4;
	bool enabled = false;
};

enum class ofxStableDiffusionModelFamily {
	Unknown = 0,
	SD1,
	SD2,
	SDXL,
	SD3,
	FLUX,
	FLUXFill,
	FLUXControl,
	FLUX2,
	WAN,
	WANI2V,
	WANTI2V,
	WANFLF2V,
	WANVACE
};

struct ofxStableDiffusionCapabilities {
	bool contextConfigured = false;
	bool runtimeResolved = false;
	ofxStableDiffusionModelFamily modelFamily = ofxStableDiffusionModelFamily::Unknown;
	bool textToImage = false;
	bool imageToImage = false;
	bool instructImage = false;
	bool variation = false;
	bool restyle = false;
	bool inpainting = false;
	bool imageToVideo = false;
	bool videoRequiresInputImage = true;
	bool videoEndFrame = false;
	bool videoAnimation = false;
	bool videoMetadataExport = true;
	bool lora = false;
	bool embeddings = false;
	bool controlNet = false;
	bool photoMaker = false;
	bool controlNetConfigured = false;
	bool photoMakerConfigured = false;
	bool nativeControlModel = false;
	bool upscaling = false;
	bool splitModelPaths = false;
	bool mmap = true;
	bool flashAttention = false;

	bool supportsImageMode(ofxStableDiffusionImageMode mode) const {
		switch (mode) {
		case ofxStableDiffusionImageMode::TextToImage: return textToImage;
		case ofxStableDiffusionImageMode::ImageToImage: return imageToImage;
		case ofxStableDiffusionImageMode::InstructImage: return instructImage;
		case ofxStableDiffusionImageMode::Variation: return variation;
		case ofxStableDiffusionImageMode::Restyle: return restyle;
		case ofxStableDiffusionImageMode::Inpainting: return inpainting;
		default:
			return false;
		}
	}
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
	float cfgScale = std::numeric_limits<float>::infinity();
	int width = 512;
	int height = 512;
	sample_method_t sampleMethod = SAMPLE_METHOD_COUNT;
	int sampleSteps = -1;
	float flowShift = std::numeric_limits<float>::infinity();
	float strength = std::numeric_limits<float>::infinity();
	int64_t seed = -1;
	int batchCount = 1;
	// Legacy single ControlNet (deprecated, use controlNets vector instead)
	sd_image_t * controlCond = nullptr;
	float controlStrength = 0.9f;
	// Multi-ControlNet support
	std::vector<ofxStableDiffusionControlNet> controlNets;
	float styleStrength = 20.0f;
	bool normalizeInput = true;
	std::string inputIdImagesPath;
	std::vector<ofxStableDiffusionLora> loras;
};

struct ofxStableDiffusionVideoRequest {
	sd_image_t initImage{0, 0, 0, nullptr};
	// Optional end frame for video morphing (leave zeroed to disable)
	sd_image_t endImage{0, 0, 0, nullptr};
	// Optional per-frame controls for native VACE / guided video generation.
	std::vector<sd_image_t> controlFrames;
	std::string prompt;
	std::string negativePrompt;
	int clipSkip = -1;
	int width = 576;
	int height = 1024;
	int frameCount = 6;
	int fps = 6;
	float cfgScale = std::numeric_limits<float>::infinity();
	float guidance = std::numeric_limits<float>::infinity();
	sample_method_t sampleMethod = SAMPLE_METHOD_COUNT;
	int sampleSteps = -1;
	float eta = std::numeric_limits<float>::infinity();
	float flowShift = std::numeric_limits<float>::infinity();
	bool useHighNoiseOverrides = false;
	float highNoiseCfgScale = std::numeric_limits<float>::infinity();
	float highNoiseGuidance = std::numeric_limits<float>::infinity();
	sample_method_t highNoiseSampleMethod = SAMPLE_METHOD_COUNT;
	int highNoiseSampleSteps = -1;
	float highNoiseEta = std::numeric_limits<float>::infinity();
	float highNoiseFlowShift = std::numeric_limits<float>::infinity();
	float moeBoundary = std::numeric_limits<float>::infinity();
	float strength = std::numeric_limits<float>::infinity();
	int64_t seed = -1;
	// VACE control strength (0 = disabled, 1 = full)
	float vaceStrength = std::numeric_limits<float>::infinity();
	ofxStableDiffusionVideoMode mode = ofxStableDiffusionVideoMode::Standard;
	sd_cache_params_t cache{};
	std::vector<ofxStableDiffusionLora> loras;
	ofxStableDiffusionVideoAnimationSettings animationSettings;

	ofxStableDiffusionVideoRequest() {
		sd_cache_params_init(&cache);
	}

	bool hasAnimation() const {
		return animationSettings.enablePromptInterpolation ||
			animationSettings.enableParameterAnimation ||
			animationSettings.useSeedSequence;
	}

	bool hasControlFrames() const {
		return !controlFrames.empty();
	}
};

struct ofxStableDiffusionGenerationParameters {
	std::string prompt;
	std::string negativePrompt;
	float cfgScale = -1.0f;
	float strength = -1.0f;
};

struct ofxStableDiffusionImageFrame {
	ofPixels pixels;
	int index = 0;
	int sourceIndex = 0;
	int64_t seed = -1;
	ofxStableDiffusionGenerationParameters generation;
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
	std::vector<int64_t> seeds() const;
	bool saveFrameSequence(
		const std::string & directory,
		const std::string & prefix = "frame") const;
	bool saveMetadataJson(const std::string & path) const;
	bool saveFrameSequenceWithMetadata(
		const std::string & directory,
		const std::string & prefix = "frame",
		const std::string & metadataFilename = "metadata.json") const;
	bool saveWebm(const std::string & path, int quality = 90) const;
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
