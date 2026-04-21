#pragma once

#include <algorithm>
#include <cctype>
#include <string>

/// Shared string utilities used across multiple helpers.

inline std::string ofxSdToLowerCopy(std::string value) {
	std::transform(
		value.begin(),
		value.end(),
		value.begin(),
		[](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
	return value;
}
