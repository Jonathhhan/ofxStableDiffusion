#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <algorithm>

// Prompt Weight Syntax
struct ofxStableDiffusionPromptWeight {
	std::string token;
	float weight = 1.0f;  // 1.0 = normal, >1.0 = emphasize, <1.0 = de-emphasize

	std::string toString() const {
		if (weight == 1.0f) return token;
		if (weight > 1.0f) {
			// Use parentheses for emphasis: (token:weight)
			return "(" + token + ":" + std::to_string(weight) + ")";
		} else {
			// Use brackets for de-emphasis: [token:weight]
			return "[" + token + ":" + std::to_string(weight) + "]";
		}
	}
};

// Prompt Template with Variables
struct ofxStableDiffusionPromptTemplate {
	std::string templateStr;
	std::vector<std::pair<std::string, std::string>> variables;  // name -> value

	std::string apply() const {
		std::string result = templateStr;
		for (const auto& var : variables) {
			std::string placeholder = "{" + var.first + "}";
			size_t pos = 0;
			while ((pos = result.find(placeholder, pos)) != std::string::npos) {
				result.replace(pos, placeholder.length(), var.second);
				pos += var.second.length();
			}
		}
		return result;
	}
};

// Style Mixing Preset
struct ofxStableDiffusionStyleMix {
	std::string name;
	std::vector<std::pair<std::string, float>> styles;  // style name -> weight

	std::string toPrompt() const {
		std::ostringstream oss;
		bool first = true;
		for (const auto& style : styles) {
			if (!first) oss << ", ";
			first = false;
			if (style.second != 1.0f) {
				oss << "(" << style.first << ":" << style.second << ")";
			} else {
				oss << style.first;
			}
		}
		return oss.str();
	}
};

namespace ofxStableDiffusionPromptHelpers {

// Emphasize tokens in a prompt
inline std::string emphasize(const std::string& token, float weight = 1.5f) {
	return "(" + token + ":" + std::to_string(weight) + ")";
}

// De-emphasize tokens in a prompt
inline std::string deemphasize(const std::string& token, float weight = 0.7f) {
	return "[" + token + ":" + std::to_string(weight) + "]";
}

// Build weighted prompt from components
inline std::string buildWeightedPrompt(const std::vector<ofxStableDiffusionPromptWeight>& components) {
	std::ostringstream oss;
	bool first = true;
	for (const auto& comp : components) {
		if (!first) oss << ", ";
		first = false;
		oss << comp.toString();
	}
	return oss.str();
}

// Common negative prompt presets
inline std::string getNegativePromptPreset(const std::string& presetName) {
	if (presetName == "quality") {
		return "blurry, low quality, low resolution, pixelated, jpeg artifacts, "
			"noise, grain, amateur, poorly drawn, bad anatomy";
	}
	if (presetName == "artifacts") {
		return "watermark, text, signature, logo, copyright, username, "
			"border, frame, cropped, out of frame";
	}
	if (presetName == "distortion") {
		return "distorted, deformed, disfigured, mutated, malformed, "
			"extra limbs, missing limbs, fused fingers, too many fingers";
	}
	if (presetName == "style") {
		return "cartoon, anime, 3d render, illustration, painting, "
			"drawing, sketch, artistic";
	}
	if (presetName == "realistic") {
		return "unrealistic, fake, artificial, cgi, cartoon, painted, "
			"stylized, anime";
	}
	if (presetName == "comprehensive") {
		return "blurry, low quality, distorted, deformed, bad anatomy, "
			"watermark, text, signature, extra limbs, fused fingers, "
			"jpeg artifacts, noise, amateur, poorly drawn";
	}
	return "";
}

// Combine multiple negative prompts
inline std::string combineNegativePrompts(const std::vector<std::string>& prompts) {
	std::ostringstream oss;
	bool first = true;
	for (const auto& prompt : prompts) {
		if (prompt.empty()) continue;
		if (!first) oss << ", ";
		first = false;
		oss << prompt;
	}
	return oss.str();
}

// Style mixing presets
inline ofxStableDiffusionStyleMix getStyleMixPreset(const std::string& presetName) {
	ofxStableDiffusionStyleMix mix;
	mix.name = presetName;

	if (presetName == "cinematic") {
		mix.styles = {
			{"cinematic lighting", 1.3f},
			{"dramatic", 1.1f},
			{"film grain", 0.8f}
		};
	} else if (presetName == "anime") {
		mix.styles = {
			{"anime style", 1.4f},
			{"cel shaded", 1.2f},
			{"vibrant colors", 1.1f}
		};
	} else if (presetName == "photorealistic") {
		mix.styles = {
			{"photorealistic", 1.3f},
			{"detailed", 1.2f},
			{"sharp focus", 1.1f}
		};
	} else if (presetName == "artistic") {
		mix.styles = {
			{"artistic", 1.2f},
			{"painterly", 1.1f},
			{"expressive", 1.0f}
		};
	} else if (presetName == "fantasy") {
		mix.styles = {
			{"fantasy art", 1.3f},
			{"magical", 1.2f},
			{"ethereal", 1.0f}
		};
	} else if (presetName == "scifi") {
		mix.styles = {
			{"sci-fi", 1.3f},
			{"futuristic", 1.2f},
			{"high tech", 1.1f}
		};
	}

	return mix;
}

// Prompt template examples
inline ofxStableDiffusionPromptTemplate getCharacterTemplate() {
	ofxStableDiffusionPromptTemplate tmpl;
	tmpl.templateStr = "{adjective} {subject}, {style}, {lighting}, {quality}";
	tmpl.variables = {
		{"adjective", "beautiful"},
		{"subject", "character"},
		{"style", "detailed illustration"},
		{"lighting", "soft lighting"},
		{"quality", "high quality, detailed"}
	};
	return tmpl;
}

inline ofxStableDiffusionPromptTemplate getSceneTemplate() {
	ofxStableDiffusionPromptTemplate tmpl;
	tmpl.templateStr = "{location} scene, {time_of_day}, {weather}, {style}, {quality}";
	tmpl.variables = {
		{"location", "forest"},
		{"time_of_day", "golden hour"},
		{"weather", "clear sky"},
		{"style", "landscape photography"},
		{"quality", "highly detailed, professional"}
	};
	return tmpl;
}

inline ofxStableDiffusionPromptTemplate getObjectTemplate() {
	ofxStableDiffusionPromptTemplate tmpl;
	tmpl.templateStr = "{adjective} {object}, {material}, {style}, {background}, {quality}";
	tmpl.variables = {
		{"adjective", "sleek"},
		{"object", "product"},
		{"material", "metallic finish"},
		{"style", "product photography"},
		{"background", "white background"},
		{"quality", "studio lighting, high quality"}
	};
	return tmpl;
}

// Token cleanup and optimization
inline std::string cleanupPrompt(const std::string& prompt) {
	std::string result = prompt;

	// Remove duplicate commas
	size_t pos = 0;
	while ((pos = result.find(",,", pos)) != std::string::npos) {
		result.replace(pos, 2, ",");
	}

	// Remove leading/trailing commas and spaces
	while (!result.empty() && (result[0] == ',' || result[0] == ' ')) {
		result = result.substr(1);
	}
	while (!result.empty() && (result[result.length()-1] == ',' || result[result.length()-1] == ' ')) {
		result = result.substr(0, result.length() - 1);
	}

	// Normalize spaces after commas
	pos = 0;
	while ((pos = result.find(",", pos)) != std::string::npos) {
		size_t nextChar = pos + 1;
		// Remove extra spaces
		while (nextChar < result.length() && result[nextChar] == ' ') {
			result.erase(nextChar, 1);
		}
		// Add single space
		if (nextChar < result.length() && result[nextChar] != ',') {
			result.insert(nextChar, " ");
		}
		pos = nextChar + 1;
	}

	return result;
}

// Split prompt into tokens
inline std::vector<std::string> tokenizePrompt(const std::string& prompt) {
	std::vector<std::string> tokens;
	std::istringstream stream(prompt);
	std::string token;

	while (std::getline(stream, token, ',')) {
		// Trim whitespace
		token.erase(0, token.find_first_not_of(" \t\n\r"));
		token.erase(token.find_last_not_of(" \t\n\r") + 1);

		if (!token.empty()) {
			tokens.push_back(token);
		}
	}

	return tokens;
}

// Estimate prompt complexity (rough token count)
inline int estimateTokenCount(const std::string& prompt) {
	auto tokens = tokenizePrompt(prompt);
	int count = 0;

	for (const auto& token : tokens) {
		// Rough estimate: count words in each token
		std::istringstream wordStream(token);
		std::string word;
		while (wordStream >> word) {
			count++;
		}
	}

	return count;
}

// Check if prompt is likely too long
inline bool isPromptTooLong(const std::string& prompt, int maxTokens = 75) {
	return estimateTokenCount(prompt) > maxTokens;
}

// Truncate prompt to approximate token limit
inline std::string truncatePrompt(const std::string& prompt, int maxTokens = 75) {
	auto tokens = tokenizePrompt(prompt);
	int currentTokens = 0;
	std::vector<std::string> keptTokens;

	for (const auto& token : tokens) {
		std::istringstream wordStream(token);
		std::string word;
		int tokenWords = 0;
		while (wordStream >> word) {
			tokenWords++;
		}

		if (currentTokens + tokenWords <= maxTokens) {
			keptTokens.push_back(token);
			currentTokens += tokenWords;
		} else {
			break;
		}
	}

	std::ostringstream oss;
	for (size_t i = 0; i < keptTokens.size(); ++i) {
		if (i > 0) oss << ", ";
		oss << keptTokens[i];
	}

	return oss.str();
}

} // namespace ofxStableDiffusionPromptHelpers
