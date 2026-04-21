#include "ofxStableDiffusionQuantization.h"
#include "ofxStableDiffusionStringUtils.h"
#include <algorithm>

std::vector<ofxStableDiffusionQuantizationLevelInfo> ofxStableDiffusionQuantizationHelpers::levelInfoCache;

std::vector<ofxStableDiffusionQuantizationLevelInfo> ofxStableDiffusionQuantizationHelpers::getAvailableLevels() {
	if (levelInfoCache.empty()) {
		initializeLevelInfo();
	}
	return levelInfoCache;
}

ofxStableDiffusionQuantizationLevelInfo ofxStableDiffusionQuantizationHelpers::getLevelInfo(
	ofxStableDiffusionQuantization level) {

	if (levelInfoCache.empty()) {
		initializeLevelInfo();
	}

	for (const auto& info : levelInfoCache) {
		if (info.level == level) {
			return info;
		}
	}

	// Return default info for None
	ofxStableDiffusionQuantizationLevelInfo defaultInfo;
	defaultInfo.level = ofxStableDiffusionQuantization::None;
	defaultInfo.name = "None";
	defaultInfo.description = "Full precision (F32)";
	return defaultInfo;
}

ofxStableDiffusionQuantization ofxStableDiffusionQuantizationHelpers::getQuantizationByName(
	const std::string& name) {

	std::string lowerName = ofxSdToLowerCopy(name);

	if (lowerName == "none" || lowerName == "f32") return ofxStableDiffusionQuantization::None;
	if (lowerName == "f16") return ofxStableDiffusionQuantization::F16;
	if (lowerName == "q8_0") return ofxStableDiffusionQuantization::Q8_0;
	if (lowerName == "q5_1") return ofxStableDiffusionQuantization::Q5_1;
	if (lowerName == "q5_0") return ofxStableDiffusionQuantization::Q5_0;
	if (lowerName == "q4_1") return ofxStableDiffusionQuantization::Q4_1;
	if (lowerName == "q4_0") return ofxStableDiffusionQuantization::Q4_0;

	return ofxStableDiffusionQuantization::None;
}

std::string ofxStableDiffusionQuantizationHelpers::getQuantizationName(
	ofxStableDiffusionQuantization level) {

	switch (level) {
		case ofxStableDiffusionQuantization::None: return "None (F32)";
		case ofxStableDiffusionQuantization::F16: return "F16";
		case ofxStableDiffusionQuantization::Q8_0: return "Q8_0";
		case ofxStableDiffusionQuantization::Q5_1: return "Q5_1";
		case ofxStableDiffusionQuantization::Q5_0: return "Q5_0";
		case ofxStableDiffusionQuantization::Q4_1: return "Q4_1";
		case ofxStableDiffusionQuantization::Q4_0: return "Q4_0";
		default: return "Unknown";
	}
}

size_t ofxStableDiffusionQuantizationHelpers::estimateQuantizedSize(
	size_t originalSizeMB,
	ofxStableDiffusionQuantization level) {

	float reduction = getMemoryReduction(level);
	return static_cast<size_t>(originalSizeMB * (1.0f - reduction / 100.0f));
}

ofxStableDiffusionQuantization ofxStableDiffusionQuantizationHelpers::recommendQuantization(
	size_t availableVRAM_MB,
	size_t modelSizeMB) {

	// Leave some headroom for activations and intermediate buffers
	size_t usableVRAM = static_cast<size_t>(availableVRAM_MB * 0.7f);

	// If model fits comfortably, use full precision
	if (modelSizeMB < usableVRAM) {
		return ofxStableDiffusionQuantization::None;
	}

	// Try each quantization level from highest to lowest quality
	std::vector<ofxStableDiffusionQuantization> levels = {
		ofxStableDiffusionQuantization::F16,
		ofxStableDiffusionQuantization::Q8_0,
		ofxStableDiffusionQuantization::Q5_1,
		ofxStableDiffusionQuantization::Q5_0,
		ofxStableDiffusionQuantization::Q4_1,
		ofxStableDiffusionQuantization::Q4_0
	};

	for (auto level : levels) {
		size_t quantizedSize = estimateQuantizedSize(modelSizeMB, level);
		if (quantizedSize < usableVRAM) {
			return level;
		}
	}

	// If even Q4_0 doesn't fit, still recommend it as best option
	return ofxStableDiffusionQuantization::Q4_0;
}

ofxStableDiffusionQuantization ofxStableDiffusionQuantizationHelpers::getPreset(
	const std::string& preset) {

	std::string lowerPreset = ofxSdToLowerCopy(preset);

	if (lowerPreset == "ultra_fast") return ofxStableDiffusionQuantization::Q4_0;
	if (lowerPreset == "fast") return ofxStableDiffusionQuantization::Q5_0;
	if (lowerPreset == "balanced") return ofxStableDiffusionQuantization::Q8_0;
	if (lowerPreset == "quality") return ofxStableDiffusionQuantization::F16;
	if (lowerPreset == "max_quality") return ofxStableDiffusionQuantization::None;

	return ofxStableDiffusionQuantization::None;
}

bool ofxStableDiffusionQuantizationHelpers::isSupported(ofxStableDiffusionQuantization level) {
	// All levels are conceptually supported
	// In practice, support depends on the underlying stable-diffusion.cpp build
	return true;
}

float ofxStableDiffusionQuantizationHelpers::getMemoryReduction(
	ofxStableDiffusionQuantization level) {

	switch (level) {
		case ofxStableDiffusionQuantization::None: return 0.0f;
		case ofxStableDiffusionQuantization::F16: return 50.0f;
		case ofxStableDiffusionQuantization::Q8_0: return 50.0f;
		case ofxStableDiffusionQuantization::Q5_1: return 65.0f;
		case ofxStableDiffusionQuantization::Q5_0: return 65.0f;
		case ofxStableDiffusionQuantization::Q4_1: return 75.0f;
		case ofxStableDiffusionQuantization::Q4_0: return 75.0f;
		default: return 0.0f;
	}
}

float ofxStableDiffusionQuantizationHelpers::getSpeedMultiplier(
	ofxStableDiffusionQuantization level) {

	switch (level) {
		case ofxStableDiffusionQuantization::None: return 1.0f;
		case ofxStableDiffusionQuantization::F16: return 1.2f;
		case ofxStableDiffusionQuantization::Q8_0: return 1.3f;
		case ofxStableDiffusionQuantization::Q5_1: return 1.5f;
		case ofxStableDiffusionQuantization::Q5_0: return 1.5f;
		case ofxStableDiffusionQuantization::Q4_1: return 1.8f;
		case ofxStableDiffusionQuantization::Q4_0: return 2.0f;
		default: return 1.0f;
	}
}

void ofxStableDiffusionQuantizationHelpers::initializeLevelInfo() {
	levelInfoCache.clear();

	// F32 (None)
	{
		ofxStableDiffusionQuantizationLevelInfo info;
		info.level = ofxStableDiffusionQuantization::None;
		info.name = "None (F32)";
		info.description = "Full precision 32-bit floating point. Maximum quality, highest memory usage.";
		info.memoryReductionPercent = 0.0f;
		info.speedMultiplier = 1.0f;
		info.qualityEstimate = 1.0f;
		info.recommendation = "Use when VRAM is abundant and maximum quality is needed.";
		levelInfoCache.push_back(info);
	}

	// F16
	{
		ofxStableDiffusionQuantizationLevelInfo info;
		info.level = ofxStableDiffusionQuantization::F16;
		info.name = "F16";
		info.description = "Half precision 16-bit floating point. Excellent quality with 50% memory reduction.";
		info.memoryReductionPercent = 50.0f;
		info.speedMultiplier = 1.2f;
		info.qualityEstimate = 0.98f;
		info.recommendation = "Best balance for most use cases. Minimal quality loss with good memory savings.";
		levelInfoCache.push_back(info);
	}

	// Q8_0
	{
		ofxStableDiffusionQuantizationLevelInfo info;
		info.level = ofxStableDiffusionQuantization::Q8_0;
		info.name = "Q8_0";
		info.description = "8-bit integer quantization. Very good quality with 50% memory reduction.";
		info.memoryReductionPercent = 50.0f;
		info.speedMultiplier = 1.3f;
		info.qualityEstimate = 0.95f;
		info.recommendation = "Good for systems with moderate VRAM. Slight quality loss, faster inference.";
		levelInfoCache.push_back(info);
	}

	// Q5_1
	{
		ofxStableDiffusionQuantizationLevelInfo info;
		info.level = ofxStableDiffusionQuantization::Q5_1;
		info.name = "Q5_1";
		info.description = "5-bit quantization with improved accuracy. Balanced compression at 65% reduction.";
		info.memoryReductionPercent = 65.0f;
		info.speedMultiplier = 1.5f;
		info.qualityEstimate = 0.90f;
		info.recommendation = "Good for 6-8GB VRAM systems. Noticeable but acceptable quality trade-off.";
		levelInfoCache.push_back(info);
	}

	// Q5_0
	{
		ofxStableDiffusionQuantizationLevelInfo info;
		info.level = ofxStableDiffusionQuantization::Q5_0;
		info.name = "Q5_0";
		info.description = "5-bit quantization. Good compression at 65% reduction.";
		info.memoryReductionPercent = 65.0f;
		info.speedMultiplier = 1.5f;
		info.qualityEstimate = 0.88f;
		info.recommendation = "Suitable for limited VRAM scenarios. Some quality loss visible.";
		levelInfoCache.push_back(info);
	}

	// Q4_1
	{
		ofxStableDiffusionQuantizationLevelInfo info;
		info.level = ofxStableDiffusionQuantization::Q4_1;
		info.name = "Q4_1";
		info.description = "4-bit quantization with improved accuracy. Maximum compression at 75% reduction.";
		info.memoryReductionPercent = 75.0f;
		info.speedMultiplier = 1.8f;
		info.qualityEstimate = 0.82f;
		info.recommendation = "For very limited VRAM (4-6GB). Significant quality loss, fastest inference.";
		levelInfoCache.push_back(info);
	}

	// Q4_0
	{
		ofxStableDiffusionQuantizationLevelInfo info;
		info.level = ofxStableDiffusionQuantization::Q4_0;
		info.name = "Q4_0";
		info.description = "4-bit quantization. Ultra-compressed at 75% reduction.";
		info.memoryReductionPercent = 75.0f;
		info.speedMultiplier = 2.0f;
		info.qualityEstimate = 0.80f;
		info.recommendation = "Last resort for minimal VRAM systems. Maximum speed, lowest quality.";
		levelInfoCache.push_back(info);
	}
}
