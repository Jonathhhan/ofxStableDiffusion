#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionTypes.h"
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <string>

class ofxStableDiffusion;

enum class ofxStableDiffusionRealtimeVideoQuality {
	Preview,
	Refine
};

struct ofxStableDiffusionRealtimeVideoSettings {
	int previewWidth = 512;
	int previewHeight = 512;
	int previewSteps = 4;
	int refineSteps = 16;
	float previewStrength = 0.68f;
	float refineStrength = 0.35f;
	float cfgScale = 1.5f;
	sample_method_t sampleMethod = EULER_A_SAMPLE_METHOD;
	int64_t seed = -1;
	bool lockSeed = false;
	bool usePreviousFrameFeedback = true;
	bool coalescePromptUpdates = true;
	bool dropIfBusy = false;
	bool enableRefineOnIdle = true;
	uint64_t refineAfterStableMs = 700;
};

struct ofxStableDiffusionRealtimeVideoRequest {
	std::string prompt;
	std::string negativePrompt;
	int width = 0;
	int height = 0;
	int previewSteps = -1;
	int refineSteps = -1;
	float cfgScale = std::numeric_limits<float>::infinity();
	float previewStrength = std::numeric_limits<float>::infinity();
	float refineStrength = std::numeric_limits<float>::infinity();
	sample_method_t sampleMethod = SAMPLE_METHOD_COUNT;
	int64_t seed = std::numeric_limits<int64_t>::min();
	bool lockSeed = false;
};

struct ofxStableDiffusionRealtimeVideoStats {
	bool isActive = false;
	int promptUpdates = 0;
	int coalescedUpdates = 0;
	int droppedUpdates = 0;
	int framesGenerated = 0;
	int previewFrames = 0;
	int refineFrames = 0;
	float averageLatencyMs = 0.0f;
	float lastLatencyMs = 0.0f;
	ofxStableDiffusionRealtimeVideoQuality lastQuality =
		ofxStableDiffusionRealtimeVideoQuality::Preview;
};

struct ofxStableDiffusionRealtimeVideoFrame {
	ofxStableDiffusionResult result;
	ofxStableDiffusionImageFrame frame;
	ofxStableDiffusionRealtimeVideoQuality quality =
		ofxStableDiffusionRealtimeVideoQuality::Preview;
	std::string prompt;
	float latencyMs = 0.0f;
	int frameIndex = -1;
};

using ofxSdRealtimeVideoFrameCallback =
	std::function<void(const ofxStableDiffusionRealtimeVideoFrame&)>;
using ofxSdRealtimeVideoLatencyCallback =
	std::function<void(float latencyMs, ofxStableDiffusionRealtimeVideoQuality quality)>;

inline const char * ofxStableDiffusionRealtimeVideoQualityLabel(
	ofxStableDiffusionRealtimeVideoQuality quality) {
	switch (quality) {
	case ofxStableDiffusionRealtimeVideoQuality::Preview: return "preview";
	case ofxStableDiffusionRealtimeVideoQuality::Refine: return "refine";
	}
	return "preview";
}

inline bool ofxStableDiffusionFrameHasPixels(
	const ofxStableDiffusionImageFrame * frame) {
	return frame != nullptr && frame->pixels.isAllocated() &&
		frame->pixels.getData() != nullptr;
}

inline sd_image_t ofxStableDiffusionFrameToSdImage(
	const ofxStableDiffusionImageFrame & frame) {
	return sd_image_t{
		static_cast<uint32_t>(frame.pixels.getWidth()),
		static_cast<uint32_t>(frame.pixels.getHeight()),
		static_cast<uint32_t>(frame.pixels.getNumChannels()),
		const_cast<unsigned char *>(frame.pixels.getData())
	};
}

inline ofxStableDiffusionImageRequest ofxStableDiffusionBuildRealtimeVideoImageRequest(
	const ofxStableDiffusionRealtimeVideoSettings & settings,
	const ofxStableDiffusionRealtimeVideoRequest & request,
	ofxStableDiffusionRealtimeVideoQuality quality,
	const ofxStableDiffusionImageFrame * feedbackFrame = nullptr) {
	const bool isRefine = quality == ofxStableDiffusionRealtimeVideoQuality::Refine;
	const bool hasFeedback =
		settings.usePreviousFrameFeedback && ofxStableDiffusionFrameHasPixels(feedbackFrame);

	ofxStableDiffusionImageRequest imageRequest;
	imageRequest.mode = hasFeedback
		? ofxStableDiffusionImageMode::ImageToImage
		: ofxStableDiffusionImageMode::TextToImage;
	imageRequest.prompt = request.prompt;
	imageRequest.negativePrompt = request.negativePrompt;
	imageRequest.width = request.width > 0 ? request.width : settings.previewWidth;
	imageRequest.height = request.height > 0 ? request.height : settings.previewHeight;
	imageRequest.sampleMethod = request.sampleMethod != SAMPLE_METHOD_COUNT
		? request.sampleMethod
		: settings.sampleMethod;
	imageRequest.sampleSteps = isRefine
		? (request.refineSteps > 0 ? request.refineSteps : settings.refineSteps)
		: (request.previewSteps > 0 ? request.previewSteps : settings.previewSteps);
	imageRequest.cfgScale = std::isfinite(request.cfgScale)
		? request.cfgScale
		: settings.cfgScale;
	imageRequest.strength = isRefine
		? (std::isfinite(request.refineStrength) ? request.refineStrength : settings.refineStrength)
		: (std::isfinite(request.previewStrength) ? request.previewStrength : settings.previewStrength);
	const bool lockSeed = request.lockSeed || settings.lockSeed;
	if (request.seed != std::numeric_limits<int64_t>::min()) {
		imageRequest.seed = request.seed;
	} else {
		imageRequest.seed = lockSeed ? settings.seed : -1;
	}
	imageRequest.batchCount = 1;
	if (hasFeedback) {
		imageRequest.initImage = ofxStableDiffusionFrameToSdImage(*feedbackFrame);
	}
	return imageRequest;
}

class ofxStableDiffusionRealtimeVideoSession {
public:
	bool start(const ofxStableDiffusionRealtimeVideoSettings & settings);
	bool start(const ofxStableDiffusionRealtimeVideoSettings & settings, ofxStableDiffusion & sd);
	void stop();
	bool isActive() const;

	void setGenerator(ofxStableDiffusion * sd);
	bool submit(const ofxStableDiffusionRealtimeVideoRequest & request);
	void updatePrompt(const std::string & prompt);
	void updateNegativePrompt(const std::string & negativePrompt);
	void update();

	bool isGenerating() const;
	bool hasPendingRequest() const;
	void clearFeedbackFrame();

	ofxStableDiffusionRealtimeVideoStats getStats() const;
	ofxStableDiffusionRealtimeVideoFrame getLastFrame() const;
	ofxStableDiffusionRealtimeVideoSettings getSettings() const;

	void setFrameCallback(ofxSdRealtimeVideoFrameCallback callback);
	void setLatencyCallback(ofxSdRealtimeVideoLatencyCallback callback);

private:
	bool shouldStartRefine(uint64_t nowMicros) const;
	void processNext();
	void updateStats(float latencyMs, ofxStableDiffusionRealtimeVideoQuality quality);

	mutable std::mutex mutex_;
	ofxStableDiffusion * generator_ = nullptr;
	ofxStableDiffusionRealtimeVideoSettings settings_;
	ofxStableDiffusionRealtimeVideoRequest pendingRequest_;
	ofxStableDiffusionRealtimeVideoRequest activeRequest_;
	ofxStableDiffusionRealtimeVideoStats stats_;
	ofxStableDiffusionRealtimeVideoFrame lastFrame_;
	ofxSdRealtimeVideoFrameCallback frameCallback_;
	ofxSdRealtimeVideoLatencyCallback latencyCallback_;
	bool active_ = false;
	bool generationInFlight_ = false;
	bool hasPendingRequest_ = false;
	bool hasFeedbackFrame_ = false;
	bool refinedActiveRequest_ = false;
	uint64_t generationStartMicros_ = 0;
	uint64_t lastPromptChangeMicros_ = 0;
	ofxStableDiffusionRealtimeVideoQuality activeQuality_ =
		ofxStableDiffusionRealtimeVideoQuality::Preview;
};
