#pragma once

#include "../core/ofxStableDiffusionEnums.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

inline const char * ofxStableDiffusionVideoModeName(ofxStableDiffusionVideoMode mode) {
	switch (mode) {
	case ofxStableDiffusionVideoMode::Loop: return "Loop";
	case ofxStableDiffusionVideoMode::PingPong: return "PingPong";
	case ofxStableDiffusionVideoMode::Boomerang: return "Boomerang";
	case ofxStableDiffusionVideoMode::Standard:
	default:
		return "Standard";
	}
}

inline float ofxStableDiffusionVideoDurationSeconds(std::size_t frameCount, int fps) {
	if (frameCount == 0 || fps <= 0) {
		return 0.0f;
	}
	return static_cast<float>(frameCount) / static_cast<float>(fps);
}

inline int ofxStableDiffusionVideoFrameIndexForTime(std::size_t frameCount, int fps, float seconds) {
	if (frameCount == 0) {
		return -1;
	}
	if (fps <= 0) {
		return 0;
	}
	const float clampedSeconds = std::max(0.0f, seconds);
	const int index = static_cast<int>(std::floor(clampedSeconds * static_cast<float>(fps)));
	return std::min<int>(index, static_cast<int>(frameCount) - 1);
}

inline std::vector<int> ofxStableDiffusionBuildVideoFrameSequence(
	int sourceFrameCount,
	ofxStableDiffusionVideoMode mode) {
	std::vector<int> sequence;
	if (sourceFrameCount <= 0) {
		return sequence;
	}

	sequence.reserve(static_cast<std::size_t>(sourceFrameCount) * 2);
	for (int i = 0; i < sourceFrameCount; ++i) {
		sequence.push_back(i);
	}

	if (sourceFrameCount == 1) {
		return sequence;
	}

	switch (mode) {
	case ofxStableDiffusionVideoMode::Loop:
		sequence.push_back(0);
		break;
	case ofxStableDiffusionVideoMode::PingPong:
		for (int i = sourceFrameCount - 2; i > 0; --i) {
			sequence.push_back(i);
		}
		break;
	case ofxStableDiffusionVideoMode::Boomerang:
		for (int i = sourceFrameCount - 1; i >= 0; --i) {
			sequence.push_back(i);
		}
		break;
	case ofxStableDiffusionVideoMode::Standard:
	default:
		break;
	}

	return sequence;
}
