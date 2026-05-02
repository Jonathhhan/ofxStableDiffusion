#include "core/ofxStableDiffusionValidationHelpers.h"
#include "core/ofxStableDiffusionLimits.h"

#include <iostream>
#include <limits>
#include <string>

namespace {

bool expect(bool condition, const std::string& message) {
	if (!condition) {
		std::cerr << "FAIL: " << message << std::endl;
	}
	return condition;
}

} // namespace

int main() {
	bool ok = true;

	ok &= expect(ofxSdOptionalFloatUsesAutoValue(std::numeric_limits<float>::infinity()),
		"positive infinity is treated as auto");
	ok &= expect(!ofxSdOptionalFloatUsesAutoValue(-std::numeric_limits<float>::infinity()),
		"negative infinity is not treated as auto");
	ok &= expect(ofxSdOptionalFloatIsFiniteWhenProvided(0.5f),
		"finite optional float is accepted");
	ok &= expect(!ofxSdOptionalFloatIsFiniteWhenProvided(std::numeric_limits<float>::quiet_NaN()),
		"NaN optional float is rejected");
	ok &= expect(!ofxSdOptionalFloatIsFiniteWhenProvided(-std::numeric_limits<float>::infinity()),
		"negative infinity optional float is rejected");
	ok &= expect(ofxStableDiffusionLimits::isValidCfgScale(0.0f),
		"cfg scale accepts zero");
	ok &= expect(!ofxStableDiffusionLimits::isValidCfgScale(-0.01f),
		"cfg scale rejects negative values");

	ok &= expect(!ofxSdPathHasParentTraversal("models/model..gguf"),
		"literal dots inside a filename are allowed");
	ok &= expect(ofxSdPathHasParentTraversal("../models/model.gguf"),
		"leading parent traversal is rejected");
	ok &= expect(ofxSdPathHasParentTraversal("models/../model.gguf"),
		"embedded parent traversal is rejected");
	ok &= expect(!ofxSdPathHasParentTraversal("models/subdir/model.gguf"),
		"normal nested paths are allowed");

	ok &= expect(ofxSdIsSafeChildPathComponent("metadata.json"),
		"simple child filename is allowed");
	ok &= expect(!ofxSdIsSafeChildPathComponent("../metadata.json"),
		"parent traversal child filename is rejected");
	ok &= expect(!ofxSdIsSafeChildPathComponent("nested/metadata.json"),
		"nested child filename is rejected");

	if (!ok) {
		return 1;
	}

	std::cout << "Validation helper tests passed" << std::endl;
	return 0;
}
