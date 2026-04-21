#pragma once

#include "bridges/ofxStableDiffusionHoloscanTypes.h"

#include <memory>
#include <string>
#include <vector>

class ofxStableDiffusion;

class ofxStableDiffusionHoloscanBridge {
public:
	ofxStableDiffusionHoloscanBridge();
	~ofxStableDiffusionHoloscanBridge();

	bool setup(
		ofxStableDiffusion* diffusion,
		const ofxStableDiffusionHoloscanSettings& settings = {});
	void shutdown();

	bool startImagePipeline();
	void stop();
	void update();

	void submitFrame(
		const ofPixels& pixels,
		double timestampSeconds,
		const std::string& sourceLabel = "frame");

	void submitPrompt(
		const std::string& prompt,
		const std::string& negativePrompt = "");

	bool hasPreviewFrame() const;
	const ofTexture& getPreviewTexture() const;
	ofxStableDiffusionHoloscanPreviewFrame getPreviewFrameCopy() const;
	std::vector<ofxStableDiffusionImageFrame> consumeFinishedImages();

	bool isConfigured() const;
	bool isRunning() const;
	bool isHoloscanAvailable() const;
	std::string getLastError() const;
	const ofxStableDiffusionHoloscanSettings& getSettings() const;

private:
	struct Impl;
	std::unique_ptr<Impl> impl_;
};
