#include "ofxStableDiffusionPromptHelpers.h"
#include "ofxStableDiffusionStringUtils.h"
#include <algorithm>
#include <regex>
#include <sstream>

ofxStableDiffusionPromptHelpers& ofxStableDiffusionPromptHelpers::getInstance() {
	static ofxStableDiffusionPromptHelpers instance;
	return instance;
}

ofxStableDiffusionPromptHelpers::ofxStableDiffusionPromptHelpers() {
	initializeDefaultTemplates();
	initializeDefaultNegativePresets();
}

std::string ofxStableDiffusionPromptHelpers::applyTemplate(
	const std::string& templateName,
	const std::map<std::string, std::string>& variables) {

	auto it = templates.find(templateName);
	if (it == templates.end()) {
		ofLogWarning("ofxStableDiffusionPromptHelpers")
			<< "Template not found: " << templateName;
		return "";
	}

	std::string result = it->second.templateText;

	// Replace variables in format {variable_name}
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

ofxStableDiffusionPromptAnalysis ofxStableDiffusionPromptHelpers::analyzePrompt(
	const std::string& prompt) {

	ofxStableDiffusionPromptAnalysis analysis;

	// Parse emphasis to get tokens
	analysis.tokens = parseEmphasis(prompt);

	// Estimate token count
	analysis.tokenCount = estimateTokenCount(prompt);

	// Generate warnings
	if (analysis.tokenCount > 75) {
		analysis.warnings.push_back("Prompt may be too long (>75 tokens). Consider shortening.");
	}

	if (prompt.empty()) {
		analysis.warnings.push_back("Empty prompt");
	}

	// Check for potentially problematic patterns
	if (prompt.find("  ") != std::string::npos) {
		analysis.warnings.push_back("Multiple consecutive spaces detected");
	}

	// Estimate impact based on emphasis weights
	float totalWeight = 0.0f;
	for (const auto& token : analysis.tokens) {
		totalWeight += token.weight;
	}
	analysis.estimatedImpact = analysis.tokens.empty() ? 0.0f : totalWeight / analysis.tokens.size();

	return analysis;
}

std::string ofxStableDiffusionPromptHelpers::getNegativePreset(const std::string& presetName) {
	auto it = negativePresets.find(presetName);
	if (it != negativePresets.end()) {
		return it->second;
	}

	ofLogWarning("ofxStableDiffusionPromptHelpers")
		<< "Negative preset not found: " << presetName;
	return "";
}

std::vector<ofxStableDiffusionPromptToken> ofxStableDiffusionPromptHelpers::parseEmphasis(
	const std::string& prompt) {

	std::vector<ofxStableDiffusionPromptToken> tokens;

	// Parse (word:weight) syntax
	std::regex emphasisRegex(R"(\(([^:)]+):([0-9.]+)\))");
	std::string remaining = prompt;
	std::smatch match;

	size_t lastPos = 0;
	std::string::const_iterator searchStart(prompt.cbegin());

	while (std::regex_search(searchStart, prompt.cend(), match, emphasisRegex)) {
		size_t matchPos = match.position() + (searchStart - prompt.cbegin());

		// Add text before match as token with weight 1.0
		if (matchPos > lastPos) {
			std::string beforeText = prompt.substr(lastPos, matchPos - lastPos);
			beforeText = cleanPrompt(beforeText);
			if (!beforeText.empty()) {
				tokens.push_back({beforeText, 1.0f});
			}
		}

		// Add emphasized token
		ofxStableDiffusionPromptToken token;
		token.text = match[1].str();
		token.weight = std::stof(match[2].str());
		tokens.push_back(token);

		lastPos = matchPos + match.length();
		searchStart = match.suffix().first;
	}

	// Add remaining text
	if (lastPos < prompt.length()) {
		std::string remainingText = prompt.substr(lastPos);
		remainingText = cleanPrompt(remainingText);
		if (!remainingText.empty()) {
			tokens.push_back({remainingText, 1.0f});
		}
	}

	// If no emphasis found, return whole prompt as single token
	if (tokens.empty() && !prompt.empty()) {
		tokens.push_back({cleanPrompt(prompt), 1.0f});
	}

	return tokens;
}

int ofxStableDiffusionPromptHelpers::loadTemplates(const std::string& directory) {
	int loaded = 0;

	ofDirectory dir(directory);
	if (!dir.exists()) {
		ofLogWarning("ofxStableDiffusionPromptHelpers")
			<< "Template directory does not exist: " << directory;
		return 0;
	}

	dir.allowExt("json");
	dir.listDir();

	for (size_t i = 0; i < dir.size(); i++) {
		try {
			ofFile file = dir.getFile(i);
			ofJson json = ofLoadJson(file.getAbsolutePath());

			ofxStableDiffusionPromptTemplate templ;
			templ.name = json.value("name", "");
			templ.category = json.value("category", "");
			templ.templateText = json.value("template", "");
			templ.description = json.value("description", "");

			if (json.contains("variables") && json["variables"].is_array()) {
				for (const auto& var : json["variables"]) {
					templ.variables.push_back(var.get<std::string>());
				}
			}

			if (!templ.name.empty() && !templ.templateText.empty()) {
				templates[templ.name] = templ;
				loaded++;
			}
		} catch (const std::exception& e) {
			ofLogWarning("ofxStableDiffusionPromptHelpers")
				<< "Failed to load template: " << dir.getFile(i).getFileName()
				<< " - " << e.what();
		}
	}

	ofLogNotice("ofxStableDiffusionPromptHelpers")
		<< "Loaded " << loaded << " templates from " << directory;

	return loaded;
}

std::vector<std::string> ofxStableDiffusionPromptHelpers::getTemplateCategories() const {
	std::vector<std::string> categories;
	std::set<std::string> uniqueCategories;

	for (const auto& pair : templates) {
		if (!pair.second.category.empty()) {
			uniqueCategories.insert(pair.second.category);
		}
	}

	categories.assign(uniqueCategories.begin(), uniqueCategories.end());
	return categories;
}

std::vector<std::string> ofxStableDiffusionPromptHelpers::getTemplatesInCategory(
	const std::string& category) const {

	std::vector<std::string> templateNames;

	for (const auto& pair : templates) {
		if (pair.second.category == category) {
			templateNames.push_back(pair.first);
		}
	}

	return templateNames;
}

ofxStableDiffusionPromptTemplate ofxStableDiffusionPromptHelpers::getTemplate(
	const std::string& name) const {

	auto it = templates.find(name);
	if (it != templates.end()) {
		return it->second;
	}

	return ofxStableDiffusionPromptTemplate();
}

void ofxStableDiffusionPromptHelpers::addTemplate(const ofxStableDiffusionPromptTemplate& templ) {
	if (!templ.name.empty()) {
		templates[templ.name] = templ;
	}
}

std::map<std::string, std::string> ofxStableDiffusionPromptHelpers::getAllNegativePresets() const {
	return negativePresets;
}

int ofxStableDiffusionPromptHelpers::estimateTokenCount(const std::string& text) const {
	if (text.empty()) return 0;

	// Rough estimation: split by common delimiters and count
	// Real tokenization would require CLIP tokenizer
	std::string cleaned = cleanPrompt(text);

	int count = 1; // Start with 1 for first token
	for (char c : cleaned) {
		if (c == ' ' || c == ',' || c == '.' || c == ';') {
			count++;
		}
	}

	return count;
}

std::string ofxStableDiffusionPromptHelpers::cleanPrompt(const std::string& prompt) const {
	std::string result = prompt;

	// Trim leading/trailing whitespace
	size_t start = result.find_first_not_of(" \t\n\r");
	size_t end = result.find_last_not_of(" \t\n\r");

	if (start == std::string::npos) {
		return "";
	}

	result = result.substr(start, end - start + 1);

	// Replace multiple spaces with single space
	std::regex multiSpace("  +");
	result = std::regex_replace(result, multiSpace, " ");

	return result;
}

void ofxStableDiffusionPromptHelpers::initializeDefaultTemplates() {
	// Photography templates
	{
		ofxStableDiffusionPromptTemplate templ;
		templ.name = "cinematic_portrait";
		templ.category = "photography";
		templ.templateText = "cinematic portrait of {subject}, {lighting}, {mood} mood, professional photography, highly detailed, sharp focus";
		templ.variables = {"subject", "lighting", "mood"};
		templ.description = "Cinematic style portrait with customizable subject, lighting, and mood";
		templates[templ.name] = templ;
	}

	{
		ofxStableDiffusionPromptTemplate templ;
		templ.name = "landscape";
		templ.category = "photography";
		templ.templateText = "{scene} landscape, {time_of_day}, {weather}, professional photography, highly detailed, 8k, award winning";
		templ.variables = {"scene", "time_of_day", "weather"};
		templ.description = "Landscape photography with customizable scene, time, and weather";
		templates[templ.name] = templ;
	}

	// Art style templates
	{
		ofxStableDiffusionPromptTemplate templ;
		templ.name = "anime_character";
		templ.category = "anime";
		templ.templateText = "anime style {character}, {expression}, {clothing}, detailed anime art, vibrant colors, clean lines";
		templ.variables = {"character", "expression", "clothing"};
		templ.description = "Anime character with customizable features";
		templates[templ.name] = templ;
	}

	{
		ofxStableDiffusionPromptTemplate templ;
		templ.name = "oil_painting";
		templ.category = "art";
		templ.templateText = "{subject} in the style of oil painting, {style_artist}, rich colors, textured brushstrokes, masterpiece";
		templ.variables = {"subject", "style_artist"};
		templ.description = "Oil painting style with customizable subject and artist style";
		templates[templ.name] = templ;
	}
}

void ofxStableDiffusionPromptHelpers::initializeDefaultNegativePresets() {
	negativePresets["quality_boost"] =
		"low quality, blurry, distorted, watermark, text, signature, jpeg artifacts, worst quality";

	negativePresets["photo_realistic"] =
		"cartoon, anime, painting, drawing, illustration, 3d render, cg, low quality, blurry";

	negativePresets["anatomy_fix"] =
		"bad anatomy, bad hands, missing fingers, extra fingers, mutated hands, deformed, ugly";

	negativePresets["minimal"] =
		"low quality, blurry";

	negativePresets["comprehensive"] =
		"low quality, worst quality, blurry, distorted, watermark, text, signature, jpeg artifacts, "
		"bad anatomy, bad hands, missing fingers, extra fingers, mutated hands, deformed, ugly, "
		"duplicate, morbid, mutilated, poorly drawn, extra limbs";
}
