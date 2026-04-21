#pragma once

#include "ofMain.h"
#include <string>
#include <vector>

/// Quantization levels for model compression
enum class ofxStableDiffusionQuantization {
	None,      // Full precision (F32)
	F16,       // Half precision (16-bit float) - ~50% memory reduction
	Q8_0,      // 8-bit quantization - ~50% memory reduction
	Q5_1,      // 5-bit quantization - ~65% memory reduction
	Q5_0,      // 5-bit quantization - ~65% memory reduction
	Q4_1,      // 4-bit quantization - ~75% memory reduction
	Q4_0       // 4-bit quantization - ~75% memory reduction
};

/// Information about a quantization level
struct ofxStableDiffusionQuantizationLevelInfo {
	ofxStableDiffusionQuantization level;
	std::string name;
	std::string description;
	float memoryReductionPercent;
	float speedMultiplier;
	float qualityEstimate; // 0.0-1.0, where 1.0 is best quality
	std::string recommendation;
};

/// Current quantization state information
struct ofxStableDiffusionQuantizationInfo {
	ofxStableDiffusionQuantization level = ofxStableDiffusionQuantization::None;
	float memoryReductionPercent = 0.0f;
	float qualityScore = 1.0f;
	size_t estimatedMemoryMB = 0;
	size_t originalMemoryMB = 0;
	bool isQuantized = false;
};

/// Helper utilities for model quantization
class ofxStableDiffusionQuantizationHelpers {
public:
	/// Get information about all available quantization levels
	/// @return Vector of quantization level information
	static std::vector<ofxStableDiffusionQuantizationLevelInfo> getAvailableLevels();

	/// Get information about a specific quantization level
	/// @param level Quantization level
	/// @return Information about the level
	static ofxStableDiffusionQuantizationLevelInfo getLevelInfo(ofxStableDiffusionQuantization level);

	/// Get quantization level by name
	/// @param name Name of the quantization level (case-insensitive)
	/// @return Quantization level, or None if not found
	static ofxStableDiffusionQuantization getQuantizationByName(const std::string& name);

	/// Get name of a quantization level
	/// @param level Quantization level
	/// @return Name string
	static std::string getQuantizationName(ofxStableDiffusionQuantization level);

	/// Estimate memory usage for a model with given quantization
	/// @param originalSizeMB Original model size in MB
	/// @param level Target quantization level
	/// @return Estimated size in MB after quantization
	static size_t estimateQuantizedSize(size_t originalSizeMB, ofxStableDiffusionQuantization level);

	/// Recommend quantization level based on available VRAM
	/// @param availableVRAM_MB Available VRAM in megabytes
	/// @param modelSizeMB Original model size in megabytes
	/// @return Recommended quantization level
	static ofxStableDiffusionQuantization recommendQuantization(
		size_t availableVRAM_MB,
		size_t modelSizeMB);

	/// Get quality/speed/memory preset
	/// @param preset Preset name: "ultra_fast", "fast", "balanced", "quality", "max_quality"
	/// @return Quantization level for the preset
	static ofxStableDiffusionQuantization getPreset(const std::string& preset);

	/// Check if a quantization level is supported
	/// @param level Quantization level to check
	/// @return True if supported
	static bool isSupported(ofxStableDiffusionQuantization level);

	/// Get memory reduction percentage for a quantization level
	/// @param level Quantization level
	/// @return Memory reduction percentage (0-100)
	static float getMemoryReduction(ofxStableDiffusionQuantization level);

	/// Get speed multiplier for a quantization level
	/// @param level Quantization level
	/// @return Speed multiplier (1.0 = same as F32, >1.0 = faster)
	static float getSpeedMultiplier(ofxStableDiffusionQuantization level);

private:
	static std::vector<ofxStableDiffusionQuantizationLevelInfo> levelInfoCache;
	static void initializeLevelInfo();
};
