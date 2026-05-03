#pragma once

#include "ofxStableDiffusionParameterTuningHelpers.h"
#include "ofxStableDiffusionQueue.h"
#include "ofxStableDiffusionRealtimeSession.h"
#include "ofxStableDiffusionRealtimeVideoSession.h"

#include <cmath>
#include <limits>
#include <mutex>
#include <string>

class ofxStableDiffusion;

struct ofxStableDiffusionCreativeWorkflowSettings {
	ofxStableDiffusionRealtimeSettings imagePreviewSettings;
	ofxStableDiffusionRealtimeVideoSettings videoPreviewSettings;
	ofxStableDiffusionPriority queuedRenderPriority =
		ofxStableDiffusionPriority::High;
	int renderSampleSteps = -1;
	float renderCfgScale = std::numeric_limits<float>::infinity();
	bool autoApplyModelDefaults = true;
	bool pausePreviewWhileQueueDrains = false;
	std::string queueTag = "creative-workflow";
};

struct ofxStableDiffusionCreativeWorkflowSnapshot {
	ofxStableDiffusionContextSettings contextSettings;
	ofxStableDiffusionRealtimeRequest lastImagePreviewRequest;
	ofxStableDiffusionRealtimeVideoRequest lastVideoPreviewRequest;
	bool imagePreviewActive = false;
	bool videoPreviewActive = false;
	int queuedImageRequests = 0;
	int queuedVideoRequests = 0;
};

inline ofxStableDiffusionImageRequest ofxStableDiffusionBuildCreativeRenderRequest(
	const ofxStableDiffusionRealtimeRequest& previewRequest,
	const ofxStableDiffusionContextSettings& contextSettings,
	const ofxStableDiffusionCreativeWorkflowSettings& workflowSettings,
	const ofxStableDiffusionCapabilities* capabilities = nullptr) {
	ofxStableDiffusionImageRequest request;
	request.mode = ofxStableDiffusionImageMode::TextToImage;
	request.prompt = previewRequest.prompt;
	request.negativePrompt = previewRequest.negativePrompt;
	request.width = previewRequest.width;
	request.height = previewRequest.height;
	request.seed = previewRequest.seed;
	request.sampleMethod = previewRequest.sampleMethod;
	request.cfgScale = previewRequest.cfgScale;
	request.sampleSteps = previewRequest.sampleSteps;
	request.strength = previewRequest.strength;
	request.batchCount = 1;

	if (workflowSettings.autoApplyModelDefaults) {
		ofxStableDiffusionParameterTuningHelpers::applyRecommendedImageRequest(
			contextSettings,
			request,
			capabilities,
			false);
	}

	if (workflowSettings.renderSampleSteps > 0) {
		request.sampleSteps = workflowSettings.renderSampleSteps;
	}
	if (std::isfinite(workflowSettings.renderCfgScale)) {
		request.cfgScale = workflowSettings.renderCfgScale;
	}

	const auto profile = ofxStableDiffusionParameterTuningHelpers::resolveImageProfile(
		contextSettings,
		request.mode);
	ofxStableDiffusionParameterTuningHelpers::clampImageParametersToProfile(
		profile,
		request.cfgScale,
		request.sampleSteps,
		request.strength,
		request.clipSkip);
	return request;
}

class ofxStableDiffusionCreativeWorkflow {
public:
	ofxStableDiffusionCreativeWorkflow();

	void setGenerator(ofxStableDiffusion* generator);
	ofxStableDiffusion* getGenerator() const;

	bool start(
		const ofxStableDiffusionCreativeWorkflowSettings& settings,
		ofxStableDiffusion& generator);
	void stop();
	bool isActive() const;

	bool submitImagePreview(const ofxStableDiffusionRealtimeRequest& request);
	bool submitVideoPreview(const ofxStableDiffusionRealtimeVideoRequest& request);

	int queueImageRender(
		const ofxStableDiffusionImageRequest& request,
		ofxStableDiffusionPriority priority = ofxStableDiffusionPriority::High,
		const std::string& tag = "");
	int queueImageRenderFromPreview();
	int queueVideoRender(
		const ofxStableDiffusionVideoRequest& request,
		ofxStableDiffusionPriority priority = ofxStableDiffusionPriority::High,
		const std::string& tag = "");

	void update();

	ofxStableDiffusionQueue& getQueue();
	const ofxStableDiffusionQueue& getQueue() const;
	ofxStableDiffusionRealtimeSession& getImagePreviewSession();
	const ofxStableDiffusionRealtimeSession& getImagePreviewSession() const;
	ofxStableDiffusionRealtimeVideoSession& getVideoPreviewSession();
	const ofxStableDiffusionRealtimeVideoSession& getVideoPreviewSession() const;

	ofxStableDiffusionCreativeWorkflowSnapshot getSnapshot() const;
	bool saveSession(const std::string& path) const;
	bool loadSession(const std::string& path);

private:
	void processQueue();

	mutable std::mutex mutex_;
	ofxStableDiffusion* generator_ = nullptr;
	ofxStableDiffusionCreativeWorkflowSettings settings_;
	ofxStableDiffusionQueue queue_;
	ofxStableDiffusionRealtimeSession imagePreview_;
	ofxStableDiffusionRealtimeVideoSession videoPreview_;
	ofxStableDiffusionRealtimeRequest lastImagePreviewRequest_;
	ofxStableDiffusionRealtimeVideoRequest lastVideoPreviewRequest_;
	int activeQueueRequestId_ = -1;
	bool active_ = false;
	bool queueGenerationInFlight_ = false;
};
