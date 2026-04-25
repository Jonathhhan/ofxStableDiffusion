#include "core/ofxStableDiffusionImageHelpers.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

bool expect(bool condition, const std::string & message) {
	if (condition) {
		return true;
	}

	std::cerr << "FAIL: " << message << std::endl;
	return false;
}

bool expectNear(float actual, float expected, float epsilon, const std::string & label) {
	if (std::fabs(actual - expected) <= epsilon) {
		return true;
	}

	std::cerr << "FAIL: " << label << " expected " << expected << " but got " << actual << std::endl;
	return false;
}

} // namespace

int main() {
	bool ok = true;

	// Test ofxStableDiffusionImageModeName - comprehensive enum coverage
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::TextToImage)) == "TextToImage", "text label");
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::ImageToImage)) == "ImageToImage", "img2img label");
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::Inpainting)) == "Inpainting", "inpainting label");

	// Test ofxStableDiffusionImageModeUsesInputImage - all modes
	ok &= expect(!ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::TextToImage), "text does not need input image");
	ok &= expect(ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::ImageToImage), "img2img needs input image");
	ok &= expect(ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::Inpainting), "inpainting needs input image");

	// Test ofxStableDiffusionTaskForImageMode - all modes
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::TextToImage) == ofxStableDiffusionTask::TextToImage, "text task");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::ImageToImage) == ofxStableDiffusionTask::ImageToImage, "img2img task");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::Inpainting) == ofxStableDiffusionTask::Inpainting, "inpainting task");

	// Test ofxStableDiffusionDefaultStrengthForImageMode - all modes
	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::TextToImage), 0.50f, 0.0001f, "text strength");
	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::ImageToImage), 0.50f, 0.0001f, "img2img strength");
	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::Inpainting), 0.75f, 0.0001f, "inpainting strength");

	// Test ofxStableDiffusionDefaultCfgScaleForImageMode - all modes
	ok &= expectNear(ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode::TextToImage), 7.0f, 0.0001f, "text cfg");
	ok &= expectNear(ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode::ImageToImage), 7.0f, 0.0001f, "img2img cfg");
	ok &= expectNear(ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode::Inpainting), 7.5f, 0.0001f, "inpainting cfg");

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
