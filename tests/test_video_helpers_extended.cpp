#include "video/ofxStableDiffusionVideoHelpers.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
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

	// Test basic sequences (from original test)
	ok &= expectSequence(0, ofxStableDiffusionVideoMode::Standard, {}, "empty standard");
	ok &= expectSequence(1, ofxStableDiffusionVideoMode::Boomerang, {0}, "single-frame boomerang");
	ok &= expectSequence(4, ofxStableDiffusionVideoMode::Standard, {0, 1, 2, 3}, "standard sequence");
	ok &= expectSequence(4, ofxStableDiffusionVideoMode::Loop, {0, 1, 2, 3, 0}, "loop sequence");
	ok &= expectSequence(4, ofxStableDiffusionVideoMode::PingPong, {0, 1, 2, 3, 2, 1}, "ping-pong sequence");
	ok &= expectSequence(4, ofxStableDiffusionVideoMode::Boomerang, {0, 1, 2, 3, 3, 2, 1, 0}, "boomerang sequence");

	// Test edge cases - negative frame count
	ok &= expectSequence(-1, ofxStableDiffusionVideoMode::Standard, {}, "negative frame count standard");
	ok &= expectSequence(-5, ofxStableDiffusionVideoMode::Loop, {}, "negative frame count loop");

	// Test edge cases - single frame for all modes
	ok &= expectSequence(1, ofxStableDiffusionVideoMode::Standard, {0}, "single-frame standard");
	ok &= expectSequence(1, ofxStableDiffusionVideoMode::Loop, {0}, "single-frame loop");
	ok &= expectSequence(1, ofxStableDiffusionVideoMode::PingPong, {0}, "single-frame ping-pong");

	// Test edge cases - two frames
	ok &= expectSequence(2, ofxStableDiffusionVideoMode::Standard, {0, 1}, "two-frame standard");
	ok &= expectSequence(2, ofxStableDiffusionVideoMode::Loop, {0, 1, 0}, "two-frame loop");
	ok &= expectSequence(2, ofxStableDiffusionVideoMode::PingPong, {0, 1}, "two-frame ping-pong (no middle)");
	ok &= expectSequence(2, ofxStableDiffusionVideoMode::Boomerang, {0, 1, 1, 0}, "two-frame boomerang");

	// Test larger sequences
	ok &= expectSequence(6, ofxStableDiffusionVideoMode::Standard, {0, 1, 2, 3, 4, 5}, "6-frame standard");
	ok &= expectSequence(6, ofxStableDiffusionVideoMode::Loop, {0, 1, 2, 3, 4, 5, 0}, "6-frame loop");
	ok &= expectSequence(6, ofxStableDiffusionVideoMode::PingPong, {0, 1, 2, 3, 4, 5, 4, 3, 2, 1}, "6-frame ping-pong");
	ok &= expectSequence(6, ofxStableDiffusionVideoMode::Boomerang, {0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0}, "6-frame boomerang");

	// Test duration calculations
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(0, 6), 0.0f, 0.0001f, "empty duration");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(6, 6), 1.0f, 0.0001f, "whole-second duration");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(5, 10), 0.5f, 0.0001f, "fractional duration");

	// Test duration edge cases
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(0, 0), 0.0f, 0.0001f, "zero frames zero fps");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(10, 0), 0.0f, 0.0001f, "zero fps");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(10, -5), 0.0f, 0.0001f, "negative fps");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(1, 1), 1.0f, 0.0001f, "1 frame at 1 fps");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(30, 30), 1.0f, 0.0001f, "30 frames at 30 fps");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(60, 30), 2.0f, 0.0001f, "60 frames at 30 fps");
	ok &= expectNear(ofxStableDiffusionVideoDurationSeconds(1, 60), 1.0f/60.0f, 0.0001f, "1 frame at 60 fps");

	// Test frame index lookup
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(0, 6, 0.1f) == -1, "empty frame lookup");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(4, 0, 1.0f) == 0, "zero-fps lookup clamps to first frame");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(4, 4, -1.0f) == 0, "negative time clamps to first frame");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(4, 4, 0.49f) == 1, "fractional frame lookup");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(4, 4, 99.0f) == 3, "late time clamps to last frame");

	// Test frame index lookup edge cases
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(0, 0, 0.0f) == -1, "zero frames zero fps");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(10, -1, 1.0f) == 0, "negative fps clamps to first");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(1, 30, 0.0f) == 0, "single frame at time 0");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(1, 30, 100.0f) == 0, "single frame at large time");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(10, 30, 0.0f) == 0, "first frame at time 0");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(10, 30, 0.03f) == 0, "frame at boundary");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(10, 30, 0.034f) == 1, "frame just after boundary");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(10, 10, 0.5f) == 5, "frame at middle");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(10, 10, 0.9f) == 9, "frame near end");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(10, 10, 0.99f) == 9, "frame very near end");
	ok &= expect(ofxStableDiffusionVideoFrameIndexForTime(10, 10, 1.0f) == 9, "frame at exact duration");

	// Test mode labels
	ok &= expect(std::string(ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode::Standard)) == "Standard", "standard label");
	ok &= expect(std::string(ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode::Loop)) == "Loop", "loop label");
	ok &= expect(std::string(ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode::PingPong)) == "PingPong", "ping-pong label");
	ok &= expect(std::string(ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode::Boomerang)) == "Boomerang", "boomerang label");

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
