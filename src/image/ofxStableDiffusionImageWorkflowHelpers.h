#pragma once

#include "../core/ofxStableDiffusionTypes.h"
#include "../core/ofxStableDiffusionParameterTuningHelpers.h"
#include <string>
#include <vector>
#include <cmath>

// Image Generation Workflow Presets
enum class ofxStableDiffusionImageWorkflowPreset {
	QuickDraft = 0,     // 4 steps, fastest iteration
	FastPreview,        // 12 steps, quick feedback
	Balanced,           // 24 steps, default quality
	HighQuality,        // 50 steps, maximum detail
	DetailEnhance,      // Optimized for upscaling/refinement
	StyleTransfer,      // Tuned for img2img artistic work
	ProductionReady,    // Conservative settings for final output
	ExperimentalHigh    // Aggressive settings for exploration
};

// Image Progress Tracking
struct ofxStableDiffusionImageProgressInfo {
	int currentImage = 0;
	int totalImages = 0;
	int currentStep = 0;
	int totalSteps = 0;
	float elapsedSeconds = 0.0f;
	float estimatedRemainingSeconds = 0.0f;
	float percentComplete = 0.0f;
	std::string currentPhase;  // "generating", "ranking", "complete"

	std::string getETA() const {
		if (estimatedRemainingSeconds <= 0.0f) return "calculating...";
		int mins = static_cast<int>(estimatedRemainingSeconds / 60.0f);
		int secs = static_cast<int>(estimatedRemainingSeconds) % 60;
		return std::to_string(mins) + "m " + std::to_string(secs) + "s";
	}
};

// Dry-Run Estimation for Images
struct ofxStableDiffusionImageDryRunEstimate {
	int totalImages = 0;
	int totalSteps = 0;
	float estimatedMinutes = 0.0f;
	float estimatedMemoryMB = 0.0f;
	std::string recommendation;
	bool feasible = true;
	std::vector<std::string> warnings;
};

// Batch Diversity Settings
enum class ofxStableDiffusionBatchDiversityMode {
	None = 0,           // All images use same seed
	Sequential,         // Increment seed by 1
	LargeSteps,         // Increment seed by 1000
	Random,             // Completely random seeds
	ParameterSweep      // Vary parameters systematically
};

struct ofxStableDiffusionBatchDiversitySettings {
	ofxStableDiffusionBatchDiversityMode mode = ofxStableDiffusionBatchDiversityMode::Sequential;
	int seedIncrement = 1;
	float cfgScaleStart = -1.0f;  // -1 means don't sweep
	float cfgScaleEnd = -1.0f;
	float strengthStart = -1.0f;
	float strengthEnd = -1.0f;
};

// Image Template for common workflows
struct ofxStableDiffusionImageTemplate {
	std::string name;
	std::string description;
	ofxStableDiffusionImageRequest baseRequest;
	std::vector<std::string> promptSuggestions;
	std::vector<std::string> negativePromptSuggestions;
};

// Validation Result with Auto-Correction
struct ofxStableDiffusionImageValidationResult {
	bool isValid = true;
	std::vector<std::string> errors;
	std::vector<std::string> warnings;
	std::vector<std::string> suggestions;
	ofxStableDiffusionImageRequest correctedRequest;
	bool hasCorrectedRequest = false;
};

// Seed Exploration Settings
struct ofxStableDiffusionSeedExplorationSettings {
	int64_t centerSeed = -1;
	int gridSize = 3;  // 3x3 = 9 variations
	int seedRadius = 100;  // How far from center to explore
	bool useRadialPattern = true;  // vs grid pattern
};

// Parameter Sweep Settings
struct ofxStableDiffusionParameterSweepSettings {
	enum class SweepParameter {
		CfgScale,
		Steps,
		Strength,
		Seed
	};

	SweepParameter parameter = SweepParameter::CfgScale;
	float minValue = 0.0f;
	float maxValue = 0.0f;
	int stepCount = 5;
};

namespace ofxStableDiffusionImageWorkflowHelpers {

// Apply workflow preset to request
inline void applyWorkflowPreset(
	ofxStableDiffusionImageRequest& request,
	ofxStableDiffusionImageWorkflowPreset preset,
	const ofxStableDiffusionContextSettings& contextSettings) {

	auto profile = ofxStableDiffusionParameterTuningHelpers::resolveImageProfile(
		contextSettings, request.mode);

	switch (preset) {
	case ofxStableDiffusionImageWorkflowPreset::QuickDraft:
		request.sampleSteps = 4;
		request.cfgScale = std::max(1.5f, profile.minCfgScale);
		request.width = (request.width / 128) * 128;  // Round to 128
		request.height = (request.height / 128) * 128;
		break;

	case ofxStableDiffusionImageWorkflowPreset::FastPreview:
		request.sampleSteps = 12;
		request.cfgScale = profile.defaultCfgScale * 0.8f;
		break;

	case ofxStableDiffusionImageWorkflowPreset::Balanced:
		request.sampleSteps = profile.defaultSampleSteps;
		request.cfgScale = profile.defaultCfgScale;
		request.strength = profile.defaultStrength;
		break;

	case ofxStableDiffusionImageWorkflowPreset::HighQuality:
		request.sampleSteps = std::min(50, profile.maxSampleSteps);
		request.cfgScale = profile.defaultCfgScale;
		break;

	case ofxStableDiffusionImageWorkflowPreset::DetailEnhance:
		request.sampleSteps = std::min(40, profile.maxSampleSteps);
		request.cfgScale = std::min(profile.defaultCfgScale + 1.0f, profile.maxCfgScale);
		if (request.mode != ofxStableDiffusionImageMode::TextToImage) {
			request.strength = 0.35f;  // Light touch for refinement
		}
		break;

	case ofxStableDiffusionImageWorkflowPreset::StyleTransfer:
		request.sampleSteps = std::min(35, profile.maxSampleSteps);
		request.cfgScale = std::min(profile.defaultCfgScale + 1.5f, profile.maxCfgScale);
		if (request.mode != ofxStableDiffusionImageMode::TextToImage) {
			request.strength = 0.65f;  // Heavier transformation
		}
		break;

	case ofxStableDiffusionImageWorkflowPreset::ProductionReady:
		request.sampleSteps = std::min(45, profile.maxSampleSteps);
		request.cfgScale = profile.defaultCfgScale;
		request.batchCount = 1;  // Single image for reliability
		break;

	case ofxStableDiffusionImageWorkflowPreset::ExperimentalHigh:
		request.sampleSteps = profile.maxSampleSteps;
		request.cfgScale = profile.maxCfgScale;
		break;
	}

	// Clamp to profile limits
	ofxStableDiffusionParameterTuningHelpers::clampImageParametersToProfile(
		profile, request.cfgScale, request.sampleSteps, request.strength, request.clipSkip);
}

// Get preset name
inline const char* getPresetName(ofxStableDiffusionImageWorkflowPreset preset) {
	switch (preset) {
	case ofxStableDiffusionImageWorkflowPreset::QuickDraft: return "QuickDraft";
	case ofxStableDiffusionImageWorkflowPreset::FastPreview: return "FastPreview";
	case ofxStableDiffusionImageWorkflowPreset::Balanced: return "Balanced";
	case ofxStableDiffusionImageWorkflowPreset::HighQuality: return "HighQuality";
	case ofxStableDiffusionImageWorkflowPreset::DetailEnhance: return "DetailEnhance";
	case ofxStableDiffusionImageWorkflowPreset::StyleTransfer: return "StyleTransfer";
	case ofxStableDiffusionImageWorkflowPreset::ProductionReady: return "ProductionReady";
	case ofxStableDiffusionImageWorkflowPreset::ExperimentalHigh: return "ExperimentalHigh";
	default: return "Unknown";
	}
}

// Get preset description
inline const char* getPresetDescription(ofxStableDiffusionImageWorkflowPreset preset) {
	switch (preset) {
	case ofxStableDiffusionImageWorkflowPreset::QuickDraft:
		return "4 steps, fastest iteration for rapid experimentation";
	case ofxStableDiffusionImageWorkflowPreset::FastPreview:
		return "12 steps, quick feedback with reasonable quality";
	case ofxStableDiffusionImageWorkflowPreset::Balanced:
		return "24 steps, balanced quality and speed (default)";
	case ofxStableDiffusionImageWorkflowPreset::HighQuality:
		return "50 steps, maximum detail and refinement";
	case ofxStableDiffusionImageWorkflowPreset::DetailEnhance:
		return "Optimized for upscaling and subtle refinement";
	case ofxStableDiffusionImageWorkflowPreset::StyleTransfer:
		return "Tuned for img2img artistic transformation";
	case ofxStableDiffusionImageWorkflowPreset::ProductionReady:
		return "Conservative settings for reliable final output";
	case ofxStableDiffusionImageWorkflowPreset::ExperimentalHigh:
		return "Aggressive settings for creative exploration";
	default:
		return "";
	}
}

// Dry-run estimation
inline ofxStableDiffusionImageDryRunEstimate estimateImageGeneration(
	const ofxStableDiffusionImageRequest& request,
	const ofxStableDiffusionContextSettings& contextSettings) {

	ofxStableDiffusionImageDryRunEstimate estimate;
	estimate.totalImages = request.batchCount;
	estimate.totalSteps = request.sampleSteps * request.batchCount;

	// Rough time estimation (varies greatly by hardware)
	// Assume ~0.5 sec per step for SD1.5, scale by model and resolution
	float baseTimePerStep = 0.5f;

	auto modelFamily = ofxStableDiffusionCapabilityHelpers::inferModelFamily(contextSettings);
	if (modelFamily == ofxStableDiffusionModelFamily::SDXL ||
		modelFamily == ofxStableDiffusionModelFamily::SD3) {
		baseTimePerStep *= 2.5f;  // SDXL is slower
	} else if (modelFamily == ofxStableDiffusionModelFamily::FLUX ||
		modelFamily == ofxStableDiffusionModelFamily::FLUX2) {
		baseTimePerStep *= 3.5f;  // FLUX is even slower
	}

	// Scale by resolution (relative to 512x512 baseline)
	float resolutionScale = (request.width * request.height) / (512.0f * 512.0f);
	baseTimePerStep *= std::sqrt(resolutionScale);

	estimate.estimatedMinutes = (estimate.totalSteps * baseTimePerStep) / 60.0f;

	// Memory estimation (very rough)
	float baseMemoryMB = 2048.0f;  // Base model
	if (modelFamily == ofxStableDiffusionModelFamily::SDXL) {
		baseMemoryMB = 5120.0f;
	} else if (modelFamily == ofxStableDiffusionModelFamily::FLUX ||
		modelFamily == ofxStableDiffusionModelFamily::FLUX2) {
		baseMemoryMB = 8192.0f;
	}

	// Add memory for batch
	float batchMemoryMB = (request.width * request.height * 4 * request.batchCount) / (1024.0f * 1024.0f);
	estimate.estimatedMemoryMB = baseMemoryMB + batchMemoryMB;

	// Feasibility checks
	estimate.feasible = true;

	if (estimate.estimatedMinutes > 30.0f) {
		estimate.warnings.push_back("Generation may take over 30 minutes");
	}

	if (estimate.estimatedMemoryMB > 12000.0f) {
		estimate.warnings.push_back("May require more than 12GB VRAM");
		if (estimate.estimatedMemoryMB > 24000.0f) {
			estimate.feasible = false;
		}
	}

	if (request.batchCount > 8 && request.width * request.height > 768 * 768) {
		estimate.warnings.push_back("Large batch with high resolution may cause OOM");
	}

	// Recommendations
	if (estimate.estimatedMinutes < 1.0f) {
		estimate.recommendation = "Quick generation - good for iteration";
	} else if (estimate.estimatedMinutes < 5.0f) {
		estimate.recommendation = "Reasonable generation time";
	} else if (estimate.estimatedMinutes < 15.0f) {
		estimate.recommendation = "Longer generation - consider reducing steps or batch size";
	} else {
		estimate.recommendation = "Very long generation - consider FastPreview preset first";
	}

	return estimate;
}

// Enhanced validation with auto-correction
inline ofxStableDiffusionImageValidationResult validateImageRequestWithCorrection(
	const ofxStableDiffusionImageRequest& request,
	const ofxStableDiffusionContextSettings& contextSettings) {

	ofxStableDiffusionImageValidationResult result;
	result.correctedRequest = request;
	result.hasCorrectedRequest = false;

	// Dimension validation
	if (request.width <= 0 || request.height <= 0) {
		result.isValid = false;
		result.errors.push_back("Width and height must be positive");
	} else if (request.width > 2048 || request.height > 2048) {
		result.isValid = false;
		result.errors.push_back("Width and height must be <= 2048");

		// Auto-correct: scale down proportionally
		float scale = std::min(2048.0f / request.width, 2048.0f / request.height);
		result.correctedRequest.width = static_cast<int>(request.width * scale / 64) * 64;
		result.correctedRequest.height = static_cast<int>(request.height * scale / 64) * 64;
		result.hasCorrectedRequest = true;
		result.suggestions.push_back("Suggested: " +
			std::to_string(result.correctedRequest.width) + "x" +
			std::to_string(result.correctedRequest.height));
	}

	// Check multiples of 64
	if (request.width % 64 != 0 || request.height % 64 != 0) {
		result.warnings.push_back("Dimensions should be multiples of 64 for best results");

		// Auto-correct: round to nearest 64
		result.correctedRequest.width = ((request.width + 32) / 64) * 64;
		result.correctedRequest.height = ((request.height + 32) / 64) * 64;
		result.hasCorrectedRequest = true;
		result.suggestions.push_back("Suggested: " +
			std::to_string(result.correctedRequest.width) + "x" +
			std::to_string(result.correctedRequest.height));
	}

	// Batch count validation
	if (request.batchCount < 1 || request.batchCount > 16) {
		result.isValid = false;
		result.errors.push_back("Batch count must be between 1 and 16");

		result.correctedRequest.batchCount = std::max(1, std::min(16, request.batchCount));
		result.hasCorrectedRequest = true;
		result.suggestions.push_back("Suggested batch count: " +
			std::to_string(result.correctedRequest.batchCount));
	}

	// Sample steps validation
	if (request.sampleSteps < 1) {
		result.isValid = false;
		result.errors.push_back("Sample steps must be at least 1");

		result.correctedRequest.sampleSteps = 20;
		result.hasCorrectedRequest = true;
		result.suggestions.push_back("Suggested steps: 20");
	}

	// Check for turbo models with high steps
	if (ofxStableDiffusionParameterTuningHelpers::isTurboLikeModel(contextSettings) &&
		request.sampleSteps > 12) {
		result.warnings.push_back("Turbo/Lightning models work best with 4-8 steps");
		result.suggestions.push_back("Consider using QuickDraft or FastPreview preset");
	}

	// CFG scale validation
	auto profile = ofxStableDiffusionParameterTuningHelpers::resolveImageProfile(
		contextSettings, request.mode);

	if (request.cfgScale < profile.minCfgScale || request.cfgScale > profile.maxCfgScale) {
		result.warnings.push_back("CFG scale outside recommended range for this model");

		result.correctedRequest.cfgScale = std::max(profile.minCfgScale,
			std::min(profile.maxCfgScale, request.cfgScale));
		result.hasCorrectedRequest = true;
		result.suggestions.push_back("Suggested CFG: " +
			std::to_string(result.correctedRequest.cfgScale));
	}

	// Mode-specific validation
	if (ofxStableDiffusionImageModeUsesInputImage(request.mode)) {
		if (request.initImage.data == nullptr) {
			result.isValid = false;
			result.errors.push_back("This mode requires an input image");
		}
	}

	return result;
}

// Apply batch diversity
inline void applyBatchDiversity(
	std::vector<ofxStableDiffusionImageRequest>& requests,
	const ofxStableDiffusionBatchDiversitySettings& diversity) {

	if (requests.empty()) return;

	for (size_t i = 1; i < requests.size(); ++i) {
		switch (diversity.mode) {
		case ofxStableDiffusionBatchDiversityMode::Sequential:
			if (requests[0].seed >= 0) {
				requests[i].seed = requests[0].seed + static_cast<int64_t>(i * diversity.seedIncrement);
			}
			break;

		case ofxStableDiffusionBatchDiversityMode::LargeSteps:
			if (requests[0].seed >= 0) {
				requests[i].seed = requests[0].seed + static_cast<int64_t>(i * 1000);
			}
			break;

		case ofxStableDiffusionBatchDiversityMode::Random:
			// Let SD generate random seed
			requests[i].seed = -1;
			break;

		case ofxStableDiffusionBatchDiversityMode::ParameterSweep:
			if (diversity.cfgScaleStart >= 0 && diversity.cfgScaleEnd >= 0) {
				float t = static_cast<float>(i) / static_cast<float>(requests.size() - 1);
				requests[i].cfgScale = diversity.cfgScaleStart +
					t * (diversity.cfgScaleEnd - diversity.cfgScaleStart);
			}
			if (diversity.strengthStart >= 0 && diversity.strengthEnd >= 0) {
				float t = static_cast<float>(i) / static_cast<float>(requests.size() - 1);
				requests[i].strength = diversity.strengthStart +
					t * (diversity.strengthEnd - diversity.strengthStart);
			}
			break;

		case ofxStableDiffusionBatchDiversityMode::None:
		default:
			break;
		}
	}
}

// Generate seed exploration grid
inline std::vector<int64_t> generateSeedExplorationGrid(
	const ofxStableDiffusionSeedExplorationSettings& settings) {

	std::vector<int64_t> seeds;

	if (settings.centerSeed < 0) return seeds;

	if (settings.useRadialPattern) {
		// Radial pattern: center + rings
		seeds.push_back(settings.centerSeed);

		int count = settings.gridSize * settings.gridSize - 1;
		for (int i = 0; i < count; ++i) {
			float angle = (i / static_cast<float>(count)) * 2.0f * 3.14159265f;
			float radius = settings.seedRadius * (1.0f + (i % 3) * 0.5f);
			int64_t offset = static_cast<int64_t>(std::cos(angle) * radius);
			seeds.push_back(settings.centerSeed + offset);
		}
	} else {
		// Grid pattern
		int halfSize = settings.gridSize / 2;
		for (int y = -halfSize; y <= halfSize; ++y) {
			for (int x = -halfSize; x <= halfSize; ++x) {
				int64_t offset = static_cast<int64_t>(
					(x * settings.seedRadius) + (y * settings.seedRadius * 1000));
				seeds.push_back(settings.centerSeed + offset);
			}
		}
	}

	return seeds;
}

// Built-in templates
inline ofxStableDiffusionImageTemplate getPortraitTemplate() {
	ofxStableDiffusionImageTemplate tmpl;
	tmpl.name = "Portrait Studio";
	tmpl.description = "Optimized for portrait photography";
	tmpl.baseRequest.width = 512;
	tmpl.baseRequest.height = 768;
	tmpl.baseRequest.cfgScale = 7.0f;
	tmpl.baseRequest.sampleSteps = 30;
	tmpl.promptSuggestions = {
		"portrait of a person, professional photography, studio lighting",
		"headshot, professional, neutral background, soft lighting",
		"character portrait, detailed face, dramatic lighting"
	};
	tmpl.negativePromptSuggestions = {
		"blurry, low quality, distorted face, multiple heads",
		"cropped face, bad anatomy"
	};
	return tmpl;
}

inline ofxStableDiffusionImageTemplate getLandscapeTemplate() {
	ofxStableDiffusionImageTemplate tmpl;
	tmpl.name = "Landscape Wide";
	tmpl.description = "Optimized for landscape and scenic views";
	tmpl.baseRequest.width = 768;
	tmpl.baseRequest.height = 512;
	tmpl.baseRequest.cfgScale = 7.5f;
	tmpl.baseRequest.sampleSteps = 28;
	tmpl.promptSuggestions = {
		"landscape photography, scenic vista, golden hour",
		"mountain landscape, dramatic sky, detailed terrain",
		"nature scene, atmospheric perspective, depth"
	};
	tmpl.negativePromptSuggestions = {
		"people, humans, buildings, urban",
		"blurry, low quality, distorted"
	};
	return tmpl;
}

inline ofxStableDiffusionImageTemplate getConceptArtTemplate() {
	ofxStableDiffusionImageTemplate tmpl;
	tmpl.name = "Concept Art";
	tmpl.description = "Optimized for concept art and illustrations";
	tmpl.baseRequest.width = 768;
	tmpl.baseRequest.height = 768;
	tmpl.baseRequest.cfgScale = 8.5f;
	tmpl.baseRequest.sampleSteps = 35;
	tmpl.promptSuggestions = {
		"concept art, detailed illustration, digital painting",
		"fantasy art, highly detailed, cinematic lighting",
		"sci-fi concept, futuristic design, detailed rendering"
	};
	tmpl.negativePromptSuggestions = {
		"blurry, low quality, amateur",
		"photo, photograph, realistic"
	};
	return tmpl;
}

inline ofxStableDiffusionImageTemplate getProductShotTemplate() {
	ofxStableDiffusionImageTemplate tmpl;
	tmpl.name = "Product Shot";
	tmpl.description = "Optimized for product photography";
	tmpl.baseRequest.width = 768;
	tmpl.baseRequest.height = 768;
	tmpl.baseRequest.cfgScale = 6.5f;
	tmpl.baseRequest.sampleSteps = 32;
	tmpl.promptSuggestions = {
		"product photography, studio lighting, white background",
		"commercial product shot, professional, clean composition",
		"ecommerce product photo, detailed, well-lit"
	};
	tmpl.negativePromptSuggestions = {
		"blurry, cluttered, poor lighting",
		"distorted, low quality"
	};
	return tmpl;
}

} // namespace ofxStableDiffusionImageWorkflowHelpers
