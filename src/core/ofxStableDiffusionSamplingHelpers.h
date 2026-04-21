#pragma once

#include "stable-diffusion.h"
#include <string>
#include <vector>
#include <map>

/// Sampling parameter preset for common use cases
struct ofxStableDiffusionSamplingPreset {
	std::string name;
	std::string description;
	sample_method_t sampleMethod;
	scheduler_t scheduler;
	int steps;
	float cfgScale;

	/// Quality presets (higher steps, better quality)
	static ofxStableDiffusionSamplingPreset Quality();
	static ofxStableDiffusionSamplingPreset UltraQuality();

	/// Speed presets (fewer steps, faster generation)
	static ofxStableDiffusionSamplingPreset Fast();
	static ofxStableDiffusionSamplingPreset UltraFast();

	/// Balanced presets
	static ofxStableDiffusionSamplingPreset Balanced();

	/// Specialized presets
	static ofxStableDiffusionSamplingPreset LCM();  // Latent Consistency Model
	static ofxStableDiffusionSamplingPreset TCD();  // Trajectory Consistency Distillation
};

/// Sampling method information
struct ofxStableDiffusionSamplerInfo {
	sample_method_t method;
	std::string name;
	std::string description;
	int recommendedMinSteps;
	int recommendedMaxSteps;
	bool supportsKarras;
};

/// Scheduler information
struct ofxStableDiffusionSchedulerInfo {
	scheduler_t scheduler;
	std::string name;
	std::string description;
	bool recommendedForPhotorealism;
	bool recommendedForArt;
};

/// Helper utilities for sampling configuration
class ofxStableDiffusionSamplingHelpers {
public:
	/// Get all available sampling methods with metadata
	static std::vector<ofxStableDiffusionSamplerInfo> getAllSamplers();

	/// Get all available schedulers with metadata
	static std::vector<ofxStableDiffusionSchedulerInfo> getAllSchedulers();

	/// Get sampler info by enum
	static ofxStableDiffusionSamplerInfo getSamplerInfo(sample_method_t method);

	/// Get scheduler info by enum
	static ofxStableDiffusionSchedulerInfo getSchedulerInfo(scheduler_t scheduler);

	/// Get sampler enum by name (case-insensitive)
	static sample_method_t getSamplerByName(const std::string& name);

	/// Get scheduler enum by name (case-insensitive)
	static scheduler_t getSchedulerByName(const std::string& name);

	/// Get human-readable name for sample method
	static std::string getSamplerName(sample_method_t method);

	/// Get human-readable name for scheduler
	static std::string getSchedulerName(scheduler_t scheduler);

	/// Get recommended sampler/scheduler combinations
	static std::vector<std::pair<sample_method_t, scheduler_t>> getRecommendedCombinations();

	/// Validate step count for a given sampler
	static bool isValidStepCount(sample_method_t method, int steps);

	/// Get recommended step count for a sampler at quality level (0.0=fast, 1.0=quality)
	static int getRecommendedSteps(sample_method_t method, float qualityLevel = 0.5f);

	/// All presets as a list
	static std::vector<ofxStableDiffusionSamplingPreset> getAllPresets();
};
