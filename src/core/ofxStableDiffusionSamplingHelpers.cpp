#include "ofxStableDiffusionSamplingHelpers.h"
#include <algorithm>
#include <cctype>

// Sampling Presets
ofxStableDiffusionSamplingPreset ofxStableDiffusionSamplingPreset::Quality() {
	return {"Quality", "High quality output with detailed results",
		DPMPP2M_SAMPLE_METHOD, KARRAS_SCHEDULER, 30, 7.5f};
}

ofxStableDiffusionSamplingPreset ofxStableDiffusionSamplingPreset::UltraQuality() {
	return {"Ultra Quality", "Maximum quality for final renders",
		DPMPP2M_SAMPLE_METHOD, KARRAS_SCHEDULER, 50, 8.0f};
}

ofxStableDiffusionSamplingPreset ofxStableDiffusionSamplingPreset::Fast() {
	return {"Fast", "Quick preview with acceptable quality",
		EULER_A_SAMPLE_METHOD, DISCRETE_SCHEDULER, 15, 7.0f};
}

ofxStableDiffusionSamplingPreset ofxStableDiffusionSamplingPreset::UltraFast() {
	return {"Ultra Fast", "Fastest generation for rapid iteration",
		EULER_A_SAMPLE_METHOD, DISCRETE_SCHEDULER, 8, 6.0f};
}

ofxStableDiffusionSamplingPreset ofxStableDiffusionSamplingPreset::Balanced() {
	return {"Balanced", "Good balance of quality and speed",
		EULER_A_SAMPLE_METHOD, KARRAS_SCHEDULER, 20, 7.0f};
}

ofxStableDiffusionSamplingPreset ofxStableDiffusionSamplingPreset::LCM() {
	return {"LCM", "Latent Consistency Model (4-8 steps)",
		LCM_SAMPLE_METHOD, LCM_SCHEDULER, 4, 1.0f};
}

ofxStableDiffusionSamplingPreset ofxStableDiffusionSamplingPreset::TCD() {
	return {"TCD", "Trajectory Consistency Distillation",
		TCD_SAMPLE_METHOD, SGM_UNIFORM_SCHEDULER, 8, 3.0f};
}

// Sampler Information
std::vector<ofxStableDiffusionSamplerInfo> ofxStableDiffusionSamplingHelpers::getAllSamplers() {
	return {
		{EULER_SAMPLE_METHOD, "Euler",
			"Simple and fast, good for quick iterations", 10, 30, true},
		{EULER_A_SAMPLE_METHOD, "Euler A",
			"Ancestral sampling, more creative/varied results", 15, 40, true},
		{HEUN_SAMPLE_METHOD, "Heun",
			"More accurate but slower than Euler", 15, 40, true},
		{DPM2_SAMPLE_METHOD, "DPM2",
			"DPM-Solver-2, good quality", 15, 30, true},
		{DPMPP2S_A_SAMPLE_METHOD, "DPM++ 2S A",
			"Ancestral variant with good detail", 15, 30, true},
		{DPMPP2M_SAMPLE_METHOD, "DPM++ 2M",
			"Excellent quality, widely used", 20, 50, true},
		{DPMPP2Mv2_SAMPLE_METHOD, "DPM++ 2M v2",
			"Improved version of DPM++ 2M", 20, 50, true},
		{IPNDM_SAMPLE_METHOD, "iPNDM",
			"Fast, good for preview", 10, 25, false},
		{IPNDM_V_SAMPLE_METHOD, "iPNDM_v",
			"V-prediction variant of iPNDM", 10, 25, false},
		{LCM_SAMPLE_METHOD, "LCM",
			"Latent Consistency Model, 4-8 steps only", 4, 8, false},
		{DDIM_TRAILING_SAMPLE_METHOD, "DDIM Trailing",
			"Classic DDIM with trailing", 20, 50, false},
		{TCD_SAMPLE_METHOD, "TCD",
			"Trajectory Consistency Distillation, fast", 4, 12, false},
		{RES_MULTISTEP_SAMPLE_METHOD, "Restart Multistep",
			"Restart sampling, experimental", 15, 40, false},
		{RES_2S_SAMPLE_METHOD, "Restart 2S",
			"Restart 2-stage, experimental", 15, 40, false},
		{ER_SDE_SAMPLE_METHOD, "ER-SDE",
			"Exponential integrator SDE, experimental", 15, 40, false}
	};
}

std::vector<ofxStableDiffusionSchedulerInfo> ofxStableDiffusionSamplingHelpers::getAllSchedulers() {
	return {
		{DISCRETE_SCHEDULER, "Discrete",
			"Standard discrete scheduler", true, true},
		{KARRAS_SCHEDULER, "Karras",
			"Karras sigma schedule, often better quality", true, true},
		{EXPONENTIAL_SCHEDULER, "Exponential",
			"Exponential noise schedule", false, true},
		{AYS_SCHEDULER, "AYS",
			"Aligned Your Steps scheduler", true, false},
		{GITS_SCHEDULER, "GITS",
			"GITS scheduler for specific models", false, true},
		{SGM_UNIFORM_SCHEDULER, "SGM Uniform",
			"Uniform schedule for SGM models", false, true},
		{SIMPLE_SCHEDULER, "Simple",
			"Simple linear schedule", false, true},
		{SMOOTHSTEP_SCHEDULER, "Smoothstep",
			"Smoothstep interpolation schedule", false, true},
		{KL_OPTIMAL_SCHEDULER, "KL Optimal",
			"KL-optimal schedule", true, false},
		{LCM_SCHEDULER, "LCM",
			"LCM-specific scheduler", false, false},
		{BONG_TANGENT_SCHEDULER, "Bong Tangent",
			"Bong tangent schedule, experimental", false, true}
	};
}

ofxStableDiffusionSamplerInfo ofxStableDiffusionSamplingHelpers::getSamplerInfo(sample_method_t method) {
	auto samplers = getAllSamplers();
	for (const auto& info : samplers) {
		if (info.method == method) {
			return info;
		}
	}
	return {EULER_A_SAMPLE_METHOD, "Unknown", "Unknown sampler", 10, 30, false};
}

ofxStableDiffusionSchedulerInfo ofxStableDiffusionSamplingHelpers::getSchedulerInfo(scheduler_t scheduler) {
	auto schedulers = getAllSchedulers();
	for (const auto& info : schedulers) {
		if (info.scheduler == scheduler) {
			return info;
		}
	}
	return {DISCRETE_SCHEDULER, "Unknown", "Unknown scheduler", false, false};
}

static std::string toLower(const std::string& str) {
	std::string result = str;
	std::transform(result.begin(), result.end(), result.begin(),
		[](unsigned char c) { return std::tolower(c); });
	return result;
}

sample_method_t ofxStableDiffusionSamplingHelpers::getSamplerByName(const std::string& name) {
	std::string lowerName = toLower(name);
	auto samplers = getAllSamplers();
	for (const auto& info : samplers) {
		if (toLower(info.name) == lowerName) {
			return info.method;
		}
	}
	return EULER_A_SAMPLE_METHOD;  // Default
}

scheduler_t ofxStableDiffusionSamplingHelpers::getSchedulerByName(const std::string& name) {
	std::string lowerName = toLower(name);
	auto schedulers = getAllSchedulers();
	for (const auto& info : schedulers) {
		if (toLower(info.name) == lowerName) {
			return info.scheduler;
		}
	}
	return DISCRETE_SCHEDULER;  // Default
}

std::string ofxStableDiffusionSamplingHelpers::getSamplerName(sample_method_t method) {
	return getSamplerInfo(method).name;
}

std::string ofxStableDiffusionSamplingHelpers::getSchedulerName(scheduler_t scheduler) {
	return getSchedulerInfo(scheduler).name;
}

std::vector<std::pair<sample_method_t, scheduler_t>>
ofxStableDiffusionSamplingHelpers::getRecommendedCombinations() {
	return {
		{DPMPP2M_SAMPLE_METHOD, KARRAS_SCHEDULER},      // Best overall quality
		{DPMPP2Mv2_SAMPLE_METHOD, KARRAS_SCHEDULER},    // Improved DPM++
		{EULER_A_SAMPLE_METHOD, KARRAS_SCHEDULER},      // Good balance
		{EULER_A_SAMPLE_METHOD, DISCRETE_SCHEDULER},    // Fast preview
		{LCM_SAMPLE_METHOD, LCM_SCHEDULER},             // Ultra fast (LCM models)
		{TCD_SAMPLE_METHOD, SGM_UNIFORM_SCHEDULER},     // Fast (distilled models)
		{HEUN_SAMPLE_METHOD, KARRAS_SCHEDULER},         // High accuracy
		{DPM2_SAMPLE_METHOD, EXPONENTIAL_SCHEDULER}     // Alternative quality
	};
}

bool ofxStableDiffusionSamplingHelpers::isValidStepCount(sample_method_t method, int steps) {
	auto info = getSamplerInfo(method);
	return steps >= info.recommendedMinSteps && steps <= info.recommendedMaxSteps;
}

int ofxStableDiffusionSamplingHelpers::getRecommendedSteps(sample_method_t method, float qualityLevel) {
	auto info = getSamplerInfo(method);

	// Clamp quality level to [0.0, 1.0]
	qualityLevel = std::max(0.0f, std::min(1.0f, qualityLevel));

	// Interpolate between min and max steps based on quality level
	int steps = static_cast<int>(
		info.recommendedMinSteps +
		qualityLevel * (info.recommendedMaxSteps - info.recommendedMinSteps)
	);

	return steps;
}

std::vector<ofxStableDiffusionSamplingPreset> ofxStableDiffusionSamplingHelpers::getAllPresets() {
	return {
		ofxStableDiffusionSamplingPreset::UltraQuality(),
		ofxStableDiffusionSamplingPreset::Quality(),
		ofxStableDiffusionSamplingPreset::Balanced(),
		ofxStableDiffusionSamplingPreset::Fast(),
		ofxStableDiffusionSamplingPreset::UltraFast(),
		ofxStableDiffusionSamplingPreset::LCM(),
		ofxStableDiffusionSamplingPreset::TCD()
	};
}
