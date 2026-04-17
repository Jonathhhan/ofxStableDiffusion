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

	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::TextToImage)) == "TextToImage", "text label");
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::ImageToImage)) == "ImageToImage", "img2img label");
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::InstructImage)) == "InstructImage", "instruct label");
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::Variation)) == "Variation", "variation label");
	ok &= expect(std::string(ofxStableDiffusionImageModeName(ofxStableDiffusionImageMode::Restyle)) == "Restyle", "restyle label");

	ok &= expect(!ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::TextToImage), "text does not need input image");
	ok &= expect(ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::ImageToImage), "img2img needs input image");
	ok &= expect(ofxStableDiffusionImageModeUsesInputImage(ofxStableDiffusionImageMode::InstructImage), "instruct needs input image");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::TextToImage) == ofxStableDiffusionTask::TextToImage, "text task");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::InstructImage) == ofxStableDiffusionTask::InstructImage, "instruct task");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::Variation) == ofxStableDiffusionTask::ImageVariation, "variation task");
	ok &= expect(ofxStableDiffusionTaskForImageMode(ofxStableDiffusionImageMode::Restyle) == ofxStableDiffusionTask::ImageRestyle, "restyle task");

	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::InstructImage), 0.35f, 0.0001f, "instruct strength");
	ok &= expectNear(ofxStableDiffusionDefaultStrengthForImageMode(ofxStableDiffusionImageMode::Variation), 0.25f, 0.0001f, "variation strength");
	ok &= expectNear(ofxStableDiffusionDefaultCfgScaleForImageMode(ofxStableDiffusionImageMode::Restyle), 9.0f, 0.0001f, "restyle cfg");

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
