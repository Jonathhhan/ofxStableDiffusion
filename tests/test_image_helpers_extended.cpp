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
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::InstructImage)) == "InstructImage", "instruct label");
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::Variation)) == "Variation", "variation label");
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::Restyle)) == "Restyle", "restyle label");

	// Test ofxStableDiffusionImageModeUsesInputImage - all modes
	ok &= expect(!ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::TextToImage), "text does not need input image");
	ok &= expect(ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::ImageToImage), "img2img needs input image");
	ok &= expect(ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::InstructImage), "instruct needs input image");
	ok &= expect(ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::Variation), "variation needs input image");
	ok &= expect(ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::Restyle), "restyle needs input image");

	// Test ofxStableDiffusionTaskForImageMode - all modes
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::TextToImage) == ofxStableDiffusionTask::TextToImage, "text task");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::ImageToImage) == ofxStableDiffusionTask::ImageToImage, "img2img task");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::InstructImage) == ofxStableDiffusionTask::InstructImage, "instruct task");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::Variation) == ofxStableDiffusionTask::ImageVariation, "variation task");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::Restyle) == ofxStableDiffusionTask::ImageRestyle, "restyle task");

	// Test ofxStableDiffusionDefaultStrengthForImageMode - all modes
	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::TextToImage), 0.50f, 0.0001f, "text strength");
	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::ImageToImage), 0.50f, 0.0001f, "img2img strength");
	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::InstructImage), 0.35f, 0.0001f, "instruct strength");
	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::Variation), 0.25f, 0.0001f, "variation strength");
	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::Restyle), 0.75f, 0.0001f, "restyle strength");

	// Test ofxStableDiffusionDefaultCfgScaleForImageMode - all modes
	ok &= expectNear(ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode::TextToImage), 7.0f, 0.0001f, "text cfg");
	ok &= expectNear(ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode::ImageToImage), 7.0f, 0.0001f, "img2img cfg");
	ok &= expectNear(ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode::InstructImage), 4.5f, 0.0001f, "instruct cfg");
	ok &= expectNear(ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode::Variation), 3.0f, 0.0001f, "variation cfg");
	ok &= expectNear(ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode::Restyle), 9.0f, 0.0001f, "restyle cfg");

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
