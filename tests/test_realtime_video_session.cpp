#include "core/ofxStableDiffusionRealtimeVideoSession.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

static bool expect(bool condition, const std::string& message) {
	if (!condition) {
		std::cerr << "FAIL: " << message << std::endl;
	}
	return condition;
}

static ofxStableDiffusionImageFrame makeFeedbackFrame() {
	std::vector<unsigned char> pixels = {
		255, 0, 0,
		0, 255, 0,
		0, 0, 255,
		255, 255, 255
	};
	ofxStableDiffusionImageFrame frame;
	frame.index = 4;
	frame.pixels.setFromPixels(pixels.data(), 2, 2, OF_IMAGE_COLOR);
	return frame;
}

static bool testPreviewWithoutFeedback() {
	bool ok = true;
	ofxStableDiffusionRealtimeVideoSettings settings;
	settings.previewWidth = 640;
	settings.previewHeight = 384;
	settings.previewSteps = 3;
	settings.refineSteps = 11;
	settings.cfgScale = 1.25f;
	settings.seed = 1234;
	settings.lockSeed = true;

	ofxStableDiffusionRealtimeVideoRequest request;
	request.prompt = "a live painted city";

	const auto imageRequest = ofxStableDiffusionBuildRealtimeVideoImageRequest(
		settings,
		request,
		ofxStableDiffusionRealtimeVideoQuality::Preview,
		nullptr);

	ok &= expect(imageRequest.mode == ofxStableDiffusionImageMode::TextToImage,
		"preview without feedback uses text-to-image");
	ok &= expect(imageRequest.width == 640 && imageRequest.height == 384,
		"preview dimensions come from settings");
	ok &= expect(imageRequest.sampleSteps == 3,
		"preview uses preview steps");
	ok &= expect(imageRequest.cfgScale == 1.25f,
		"cfg scale comes from settings");
	ok &= expect(imageRequest.seed == 1234,
		"locked seed comes from settings");
	ok &= expect(imageRequest.initImage.data == nullptr,
		"no feedback image attached");
	return ok;
}

static bool testFeedbackTurnsPreviewIntoImg2Img() {
	bool ok = true;
	ofxStableDiffusionRealtimeVideoSettings settings;
	settings.previewSteps = 5;
	settings.previewStrength = 0.72f;
	settings.usePreviousFrameFeedback = true;

	ofxStableDiffusionRealtimeVideoRequest request;
	request.prompt = "make the camera orbit";
	request.width = 768;
	request.height = 448;
	request.cfgScale = 2.0f;
	request.seed = 99;

	const auto feedback = makeFeedbackFrame();
	const auto imageRequest = ofxStableDiffusionBuildRealtimeVideoImageRequest(
		settings,
		request,
		ofxStableDiffusionRealtimeVideoQuality::Preview,
		&feedback);

	ok &= expect(imageRequest.mode == ofxStableDiffusionImageMode::ImageToImage,
		"feedback preview uses image-to-image");
	ok &= expect(imageRequest.initImage.data == feedback.pixels.getData(),
		"feedback image points at previous frame pixels");
	ok &= expect(imageRequest.initImage.width == 2 && imageRequest.initImage.height == 2,
		"feedback image dimensions preserved");
	ok &= expect(imageRequest.width == 768 && imageRequest.height == 448,
		"request dimensions override settings");
	ok &= expect(imageRequest.sampleSteps == 5,
		"preview feedback uses preview steps");
	ok &= expect(imageRequest.strength == 0.72f,
		"preview feedback uses preview strength");
	ok &= expect(imageRequest.seed == 99,
		"request seed overrides settings");
	return ok;
}

static bool testRefineUsesRefineBudget() {
	bool ok = true;
	ofxStableDiffusionRealtimeVideoSettings settings;
	settings.refineSteps = 18;
	settings.refineStrength = 0.28f;

	ofxStableDiffusionRealtimeVideoRequest request;
	request.prompt = "settle the details";
	request.refineSteps = 24;
	request.refineStrength = 0.22f;
	request.previewSteps = 4;
	request.previewStrength = 0.80f;
	request.cfgScale = std::numeric_limits<float>::infinity();

	const auto feedback = makeFeedbackFrame();
	const auto imageRequest = ofxStableDiffusionBuildRealtimeVideoImageRequest(
		settings,
		request,
		ofxStableDiffusionRealtimeVideoQuality::Refine,
		&feedback);

	ok &= expect(imageRequest.mode == ofxStableDiffusionImageMode::ImageToImage,
		"refine uses feedback image-to-image");
	ok &= expect(imageRequest.sampleSteps == 24,
		"request refine steps override settings");
	ok &= expect(imageRequest.strength == 0.22f,
		"request refine strength override settings");
	ok &= expect(imageRequest.cfgScale == settings.cfgScale,
		"infinite request cfg falls back to settings");
	return ok;
}

int main() {
	bool ok = true;
	ok &= testPreviewWithoutFeedback();
	ok &= testFeedbackTurnsPreviewIntoImg2Img();
	ok &= testRefineUsesRefineBudget();

	if (!ok) {
		return 1;
	}
	std::cout << "Realtime video session planning tests passed" << std::endl;
	return 0;
}
