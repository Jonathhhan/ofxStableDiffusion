#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionEnums.h"
#include "ofxStableDiffusionImageHelpers.h"
#include "ofxStableDiffusionRankingHelpers.h"
#include "../../libs/stable-diffusion/include/stable-diffusion.h"

#include <cstdint>
#include <string>
#include <vector>

struct ofxStableDiffusionContextSettings {
	std::string modelPath;
	std::string vaePath;
	std::string taesdPath;
	std::string controlNetPath;
	std::string loraModelDir;
	std::string embedDir;
	std::string stackedIdEmbedDir;
	bool vaeDecodeOnly = false;
	bool vaeTiling = false;
	bool freeParamsImmediately = false;
	int nThreads = -1;
	sd_type_t weightType = SD_TYPE_F16;
	rng_type_t rngType = STD_DEFAULT_RNG;
	scheduler_t schedule = SCHEDULER_COUNT;
	bool keepClipOnCpu = false;
	bool keepControlNetCpu = false;
	bool keepVaeOnCpu = false;
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
	bool normalizeInput = true;
	std::string inputIdImagesPath;
};

struct ofxStableDiffusionVideoRequest {
	sd_image_t initImage{0, 0, 0, nullptr};
	int width = 576;
	int height = 1024;
	int frameCount = 6;
	int motionBucketId = 127;
	int fps = 6;
	float augmentationLevel = 0.0f;
	float minCfg = 1.0f;
	float cfgScale = 7.0f;
	sample_method_t sampleMethod = EULER_A_SAMPLE_METHOD;
	int sampleSteps = 20;
	float strength = 0.5f;
	int64_t seed = -1;
	ofxStableDiffusionVideoMode mode = ofxStableDiffusionVideoMode::Standard;
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
std::vector<ofxStableDiffusionImageFrame> ofxStableDiffusionBuildVideoFrames(
	const std::vector<ofxStableDiffusionImageFrame> & sourceFrames,
	ofxStableDiffusionVideoMode mode);
