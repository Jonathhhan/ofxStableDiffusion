#pragma once

#include "ofMain.h"
#include <map>
#include <string>
#include <vector>

/// Token with emphasis weight
struct ofxStableDiffusionPromptToken {
	std::string text;
	float weight = 1.0f;
};

/// Prompt analysis result
struct ofxStableDiffusionPromptAnalysis {
	int tokenCount = 0;
	float estimatedImpact = 0.0f;
	std::vector<std::string> warnings;
	std::vector<ofxStableDiffusionPromptToken> tokens;
};

/// Prompt template
struct ofxStableDiffusionPromptTemplate {
	std::string name;
	std::string category;
	std::string templateText;
	std::vector<std::string> variables;
	std::string description;
};

/// Singleton helper class for prompt engineering
class ofxStableDiffusionPromptHelpers {
public:
	static ofxStableDiffusionPromptHelpers& getInstance();

	/// Apply a template with variable substitution
	/// @param templateName Name of the template to use
	/// @param variables Map of variable names to values
	/// @return Constructed prompt string
	std::string applyTemplate(
		const std::string& templateName,
		const std::map<std::string, std::string>& variables);

	/// Analyze a prompt for token count and potential issues
	/// @param prompt The prompt to analyze
	/// @return Analysis result with token count, warnings, etc.
	ofxStableDiffusionPromptAnalysis analyzePrompt(const std::string& prompt);

	/// Get a negative prompt preset by name
	/// @param presetName Name of the preset (e.g., "quality_boost", "photo_realistic")
	/// @return Negative prompt string
	std::string getNegativePreset(const std::string& presetName);

	/// Parse emphasis syntax like (word:weight) and return weighted tokens
	/// @param prompt Prompt with emphasis syntax
	/// @return Parsed tokens with weights
	std::vector<ofxStableDiffusionPromptToken> parseEmphasis(const std::string& prompt);

	/// Load templates from a directory
	/// @param directory Path to templates directory
	/// @return Number of templates loaded
	int loadTemplates(const std::string& directory);

	/// Get list of template categories
	/// @return Vector of category names
	std::vector<std::string> getTemplateCategories() const;

	/// Get templates in a specific category
	/// @param category Category name
	/// @return Vector of template names
	std::vector<std::string> getTemplatesInCategory(const std::string& category) const;

	/// Get template by name
	/// @param name Template name
	/// @return Template if found, empty template otherwise
	ofxStableDiffusionPromptTemplate getTemplate(const std::string& name) const;

	/// Add a custom template
	/// @param templ Template to add
	void addTemplate(const ofxStableDiffusionPromptTemplate& templ);

	/// Get all available negative prompt presets
	/// @return Map of preset names to negative prompts
	std::map<std::string, std::string> getAllNegativePresets() const;

	/// Estimate token count (approximate, based on word count)
	/// @param text Text to count tokens for
	/// @return Estimated token count
	int estimateTokenCount(const std::string& text) const;

	/// Clean and normalize a prompt
	/// @param prompt Prompt to clean
	/// @return Cleaned prompt
	std::string cleanPrompt(const std::string& prompt) const;

private:
	ofxStableDiffusionPromptHelpers();
	~ofxStableDiffusionPromptHelpers() = default;

	// Prevent copying
	ofxStableDiffusionPromptHelpers(const ofxStableDiffusionPromptHelpers&) = delete;
	ofxStableDiffusionPromptHelpers& operator=(const ofxStableDiffusionPromptHelpers&) = delete;

	void initializeDefaultTemplates();
	void initializeDefaultNegativePresets();

	std::map<std::string, ofxStableDiffusionPromptTemplate> templates;
	std::map<std::string, std::string> negativePresets;
};
