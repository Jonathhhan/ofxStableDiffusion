#pragma once

#include <cmath>
#include <filesystem>
#include <string>

inline bool ofxSdOptionalFloatUsesAutoValue(float value) {
	return std::isinf(value) && value > 0.0f;
}

inline bool ofxSdOptionalFloatIsFiniteWhenProvided(float value) {
	return ofxSdOptionalFloatUsesAutoValue(value) || std::isfinite(value);
}

inline bool ofxSdPathHasParentTraversal(const std::string& value) {
	if (value.empty()) {
		return false;
	}

	const std::filesystem::path path(value);
	for (const auto& part : path) {
		if (part == "..") {
			return true;
		}
	}
	return false;
}

inline bool ofxSdIsSafeChildPathComponent(const std::string& value) {
	if (value.empty()) {
		return false;
	}
	if (ofxSdPathHasParentTraversal(value)) {
		return false;
	}

	const std::filesystem::path path(value);
	return !path.has_parent_path() &&
		path.filename() == path &&
		path.filename() != "." &&
		path.filename() != "..";
}
