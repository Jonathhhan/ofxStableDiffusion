#include "video/ofxStableDiffusionVideoHelpers.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const std::string & message) {
	if (condition) {
		return true;
	}

	std::cerr << "FAIL: " << message << std::endl;
	return false;
}

template <typename T>
std::string joinVector(const std::vector<T> & values) {
	std::ostringstream stream;
	for (std::size_t i = 0; i < values.size(); ++i) {
		if (i > 0) {
			stream << ", ";
		}
		stream << values[i];
	}
	return stream.str();
}

bool expectSequence(
	int sourceFrameCount,
	ofxStableDiffusionVideoMode mode,
	const std::vector<int> & expected,
	const std::string & label) {
	const std::vector<int> actual = ofxStableDiffusionBuildVideoFrameSequence(sourceFrameCount, mode);
	if (actual == expected) {
		return true;
	}

	std::cerr << "FAIL: " << label
		<< " expected [" << joinVector(expected)
		<< "] but got [" << joinVector(actual) << "]" << std::endl;
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

	ok &= expectSequence(0, ofxStableDiffusionVideoMode::Standard, {}, "empty standard");
	ok &= expectSequence(1, ofxStableDiffusionVideoMode::Boomerang, {0}, "single-frame boomerang");
	ok &= expectSequence(4, ofxStableDiffusionVideoMode::Standard, {0, 1, 2, 3}, "standard sequence");
	ok &= expectSequence(4, ofxStableDiffusionVideoMode::Loop, {0, 1, 2, 3, 0}, "loop sequence");
	ok &= expectSequence(4, ofxStableDiffusionVideoMode::PingPong, {0, 1, 2, 3, 2, 1}, "ping-pong sequence");
	ok &= expectSequence(4, ofxStableDiffusionVideoMode::Boomerang, {0, 1, 2, 3, 3, 2, 1, 0}, "boomerang sequence");

	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(0, 6), 0.0f, 0.0001f, "empty duration");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(6, 6), 1.0f, 0.0001f, "whole-second duration");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(5, 10), 0.5f, 0.0001f, "fractional duration");

	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(0, 6, 0.1f) == -1, "empty frame lookup");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(4, 0, 1.0f) == 0, "zero-fps lookup clamps to first frame");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(4, 4, -1.0f) == 0, "negative time clamps to first frame");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(4, 4, 0.49f) == 1, "fractional frame lookup");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(4, 4, 99.0f) == 3, "late time clamps to last frame");

	ok &= expect(std::string(ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode::Standard)) == "Standard", "standard label");
	ok &= expect(std::string(ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode::Loop)) == "Loop", "loop label");
	ok &= expect(std::string(ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode::PingPong)) == "PingPong", "ping-pong label");
	ok &= expect(std::string(ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode::Boomerang)) == "Boomerang", "boomerang label");

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
