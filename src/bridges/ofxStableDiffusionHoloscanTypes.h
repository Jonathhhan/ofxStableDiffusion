#pragma once

#include "ofMain.h"
#include "core/ofxStableDiffusionTypes.h"

#include <cstdint>
#include <memory>
#include <string>

struct ofxStableDiffusionHoloscanSettings {
	bool enabled = false;
	bool useEventScheduler = true;
	int workerThreads = 2;
	bool enableRerankStage = false;
	bool enableUpscaleStage = false;
	bool pinWorkers = false;
};

struct ofxStableDiffusionHoloscanFramePacket {
	uint64_t frameIndex = 0;
	double timestampSeconds = 0.0;
	std::shared_ptr<ofPixels> pixels;
	std::string sourceLabel;

	bool isValid() const {
		return pixels != nullptr && pixels->isAllocated();
	}
};

struct ofxStableDiffusionHoloscanConditioningPacket {
	uint64_t frameIndex = 0;
	double timestampSeconds = 0.0;
	std::string prompt;
	std::string negativePrompt;
	std::shared_ptr<ofPixels> initImage;
	float strength = 0.35f;

	bool isValid() const {
		return initImage != nullptr && initImage->isAllocated() && !prompt.empty();
	}
};

struct ofxStableDiffusionHoloscanImagePacket {
	uint64_t frameIndex = 0;
	double timestampSeconds = 0.0;
	ofxStableDiffusionImageFrame imageFrame;
};

struct ofxStableDiffusionHoloscanPreviewFrame {
	bool valid = false;
	uint64_t frameIndex = 0;
	double timestampSeconds = 0.0;
	ofPixels pixels;
};
