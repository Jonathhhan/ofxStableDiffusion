#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "../src/core/ofxStableDiffusionSamplingHelpers.h"
#include "../src/core/ofxStableDiffusionSamplingHelpers.cpp"

void testPresets() {
	std::cout << "Testing sampling presets...";

	auto quality = ofxStableDiffusionSamplingPreset::Quality();
	assert(quality.name == "Quality");
	assert(quality.sampleMethod == DPMPP2M_SAMPLE_METHOD);
	assert(quality.scheduler == KARRAS_SCHEDULER);
	assert(quality.steps == 30);
	assert(quality.cfgScale == 7.5f);

	auto ultraQuality = ofxStableDiffusionSamplingPreset::UltraQuality();
	assert(ultraQuality.steps == 50);

	auto fast = ofxStableDiffusionSamplingPreset::Fast();
	assert(fast.steps == 15);
	assert(fast.sampleMethod == EULER_A_SAMPLE_METHOD);

	auto ultraFast = ofxStableDiffusionSamplingPreset::UltraFast();
	assert(ultraFast.steps == 8);

	auto balanced = ofxStableDiffusionSamplingPreset::Balanced();
	assert(balanced.steps == 20);

	auto lcm = ofxStableDiffusionSamplingPreset::LCM();
	assert(lcm.sampleMethod == LCM_SAMPLE_METHOD);
	assert(lcm.scheduler == LCM_SCHEDULER);
	assert(lcm.steps == 4);
	assert(lcm.cfgScale == 1.0f);

	auto tcd = ofxStableDiffusionSamplingPreset::TCD();
	assert(tcd.sampleMethod == TCD_SAMPLE_METHOD);

	std::cout << " ✓" << std::endl;
}

void testGetAllSamplers() {
	std::cout << "Testing get all samplers...";

	auto samplers = ofxStableDiffusionSamplingHelpers::getAllSamplers();

	// Should have entries for all major samplers
	assert(samplers.size() > 10);

	// Check some specific samplers exist
	bool foundEulerA = false;
	bool foundDPMPP2M = false;
	bool foundLCM = false;

	for (const auto& sampler : samplers) {
		if (sampler.method == EULER_A_SAMPLE_METHOD) {
			foundEulerA = true;
			assert(!sampler.name.empty());
			assert(!sampler.description.empty());
			assert(sampler.recommendedMinSteps > 0);
			assert(sampler.recommendedMaxSteps >= sampler.recommendedMinSteps);
		}
		if (sampler.method == DPMPP2M_SAMPLE_METHOD) {
			foundDPMPP2M = true;
			assert(sampler.supportsKarras);
		}
		if (sampler.method == LCM_SAMPLE_METHOD) {
			foundLCM = true;
			assert(sampler.recommendedMinSteps == 4);
			assert(sampler.recommendedMaxSteps == 8);
		}
	}

	assert(foundEulerA);
	assert(foundDPMPP2M);
	assert(foundLCM);

	std::cout << " ✓" << std::endl;
}

void testGetAllSchedulers() {
	std::cout << "Testing get all schedulers...";

	auto schedulers = ofxStableDiffusionSamplingHelpers::getAllSchedulers();

	// Should have entries for all major schedulers
	assert(schedulers.size() > 8);

	// Check some specific schedulers exist
	bool foundKarras = false;
	bool foundDiscrete = false;
	bool foundLCM = false;

	for (const auto& scheduler : schedulers) {
		if (scheduler.scheduler == KARRAS_SCHEDULER) {
			foundKarras = true;
			assert(scheduler.name == "Karras");
			assert(!scheduler.description.empty());
		}
		if (scheduler.scheduler == DISCRETE_SCHEDULER) {
			foundDiscrete = true;
		}
		if (scheduler.scheduler == LCM_SCHEDULER) {
			foundLCM = true;
		}
	}

	assert(foundKarras);
	assert(foundDiscrete);
	assert(foundLCM);

	std::cout << " ✓" << std::endl;
}

void testGetSamplerInfo() {
	std::cout << "Testing get sampler info...";

	auto eulerInfo = ofxStableDiffusionSamplingHelpers::getSamplerInfo(EULER_A_SAMPLE_METHOD);
	assert(eulerInfo.method == EULER_A_SAMPLE_METHOD);
	assert(eulerInfo.name == "Euler A");
	assert(eulerInfo.recommendedMinSteps > 0);

	auto dpmInfo = ofxStableDiffusionSamplingHelpers::getSamplerInfo(DPMPP2M_SAMPLE_METHOD);
	assert(dpmInfo.method == DPMPP2M_SAMPLE_METHOD);
	assert(dpmInfo.supportsKarras);

	std::cout << " ✓" << std::endl;
}

void testGetSchedulerInfo() {
	std::cout << "Testing get scheduler info...";

	auto karrasInfo = ofxStableDiffusionSamplingHelpers::getSchedulerInfo(KARRAS_SCHEDULER);
	assert(karrasInfo.scheduler == KARRAS_SCHEDULER);
	assert(karrasInfo.name == "Karras");

	auto discreteInfo = ofxStableDiffusionSamplingHelpers::getSchedulerInfo(DISCRETE_SCHEDULER);
	assert(discreteInfo.scheduler == DISCRETE_SCHEDULER);

	std::cout << " ✓" << std::endl;
}

void testGetSamplerByName() {
	std::cout << "Testing get sampler by name...";

	// Exact case
	assert(ofxStableDiffusionSamplingHelpers::getSamplerByName("Euler A") == EULER_A_SAMPLE_METHOD);
	assert(ofxStableDiffusionSamplingHelpers::getSamplerByName("DPM++ 2M") == DPMPP2M_SAMPLE_METHOD);
	assert(ofxStableDiffusionSamplingHelpers::getSamplerByName("LCM") == LCM_SAMPLE_METHOD);

	// Case insensitive
	assert(ofxStableDiffusionSamplingHelpers::getSamplerByName("euler a") == EULER_A_SAMPLE_METHOD);
	assert(ofxStableDiffusionSamplingHelpers::getSamplerByName("EULER A") == EULER_A_SAMPLE_METHOD);

	// Unknown defaults to EULER_A
	assert(ofxStableDiffusionSamplingHelpers::getSamplerByName("NonExistent") == EULER_A_SAMPLE_METHOD);

	std::cout << " ✓" << std::endl;
}

void testGetSchedulerByName() {
	std::cout << "Testing get scheduler by name...";

	// Exact case
	assert(ofxStableDiffusionSamplingHelpers::getSchedulerByName("Karras") == KARRAS_SCHEDULER);
	assert(ofxStableDiffusionSamplingHelpers::getSchedulerByName("Discrete") == DISCRETE_SCHEDULER);
	assert(ofxStableDiffusionSamplingHelpers::getSchedulerByName("LCM") == LCM_SCHEDULER);

	// Case insensitive
	assert(ofxStableDiffusionSamplingHelpers::getSchedulerByName("karras") == KARRAS_SCHEDULER);
	assert(ofxStableDiffusionSamplingHelpers::getSchedulerByName("KARRAS") == KARRAS_SCHEDULER);

	// Unknown defaults to DISCRETE
	assert(ofxStableDiffusionSamplingHelpers::getSchedulerByName("NonExistent") == DISCRETE_SCHEDULER);

	std::cout << " ✓" << std::endl;
}

void testGetSamplerName() {
	std::cout << "Testing get sampler name...";

	assert(ofxStableDiffusionSamplingHelpers::getSamplerName(EULER_A_SAMPLE_METHOD) == "Euler A");
	assert(ofxStableDiffusionSamplingHelpers::getSamplerName(DPMPP2M_SAMPLE_METHOD) == "DPM++ 2M");
	assert(ofxStableDiffusionSamplingHelpers::getSamplerName(LCM_SAMPLE_METHOD) == "LCM");

	std::cout << " ✓" << std::endl;
}

void testGetSchedulerName() {
	std::cout << "Testing get scheduler name...";

	assert(ofxStableDiffusionSamplingHelpers::getSchedulerName(KARRAS_SCHEDULER) == "Karras");
	assert(ofxStableDiffusionSamplingHelpers::getSchedulerName(DISCRETE_SCHEDULER) == "Discrete");
	assert(ofxStableDiffusionSamplingHelpers::getSchedulerName(LCM_SCHEDULER) == "LCM");

	std::cout << " ✓" << std::endl;
}

void testRecommendedCombinations() {
	std::cout << "Testing recommended combinations...";

	auto combos = ofxStableDiffusionSamplingHelpers::getRecommendedCombinations();

	// Should have several recommended combinations
	assert(combos.size() >= 5);

	// Check DPM++ 2M + Karras is recommended (best quality)
	bool foundBestCombo = false;
	for (const auto& combo : combos) {
		if (combo.first == DPMPP2M_SAMPLE_METHOD && combo.second == KARRAS_SCHEDULER) {
			foundBestCombo = true;
			break;
		}
	}
	assert(foundBestCombo);

	// Check LCM + LCM scheduler is recommended
	bool foundLCMCombo = false;
	for (const auto& combo : combos) {
		if (combo.first == LCM_SAMPLE_METHOD && combo.second == LCM_SCHEDULER) {
			foundLCMCombo = true;
			break;
		}
	}
	assert(foundLCMCombo);

	std::cout << " ✓" << std::endl;
}

void testIsValidStepCount() {
	std::cout << "Testing valid step count...";

	// Euler A: 15-40 steps
	assert(ofxStableDiffusionSamplingHelpers::isValidStepCount(EULER_A_SAMPLE_METHOD, 20));
	assert(ofxStableDiffusionSamplingHelpers::isValidStepCount(EULER_A_SAMPLE_METHOD, 15));
	assert(ofxStableDiffusionSamplingHelpers::isValidStepCount(EULER_A_SAMPLE_METHOD, 40));
	assert(!ofxStableDiffusionSamplingHelpers::isValidStepCount(EULER_A_SAMPLE_METHOD, 5));
	assert(!ofxStableDiffusionSamplingHelpers::isValidStepCount(EULER_A_SAMPLE_METHOD, 50));

	// LCM: 4-8 steps
	assert(ofxStableDiffusionSamplingHelpers::isValidStepCount(LCM_SAMPLE_METHOD, 4));
	assert(ofxStableDiffusionSamplingHelpers::isValidStepCount(LCM_SAMPLE_METHOD, 6));
	assert(ofxStableDiffusionSamplingHelpers::isValidStepCount(LCM_SAMPLE_METHOD, 8));
	assert(!ofxStableDiffusionSamplingHelpers::isValidStepCount(LCM_SAMPLE_METHOD, 20));

	std::cout << " ✓" << std::endl;
}

void testGetRecommendedSteps() {
	std::cout << "Testing get recommended steps...";

	// Test quality level interpolation
	int minSteps = ofxStableDiffusionSamplingHelpers::getRecommendedSteps(EULER_A_SAMPLE_METHOD, 0.0f);
	int midSteps = ofxStableDiffusionSamplingHelpers::getRecommendedSteps(EULER_A_SAMPLE_METHOD, 0.5f);
	int maxSteps = ofxStableDiffusionSamplingHelpers::getRecommendedSteps(EULER_A_SAMPLE_METHOD, 1.0f);

	assert(minSteps == 15);  // Min for Euler A
	assert(midSteps >= minSteps && midSteps <= maxSteps);
	assert(maxSteps == 40);  // Max for Euler A

	// Test clamping
	int clampedLow = ofxStableDiffusionSamplingHelpers::getRecommendedSteps(EULER_A_SAMPLE_METHOD, -0.5f);
	int clampedHigh = ofxStableDiffusionSamplingHelpers::getRecommendedSteps(EULER_A_SAMPLE_METHOD, 1.5f);
	assert(clampedLow == minSteps);
	assert(clampedHigh == maxSteps);

	std::cout << " ✓" << std::endl;
}

void testGetAllPresets() {
	std::cout << "Testing get all presets...";

	auto presets = ofxStableDiffusionSamplingHelpers::getAllPresets();

	// Should have all 7 presets
	assert(presets.size() == 7);

	// Check that each preset has valid data
	for (const auto& preset : presets) {
		assert(!preset.name.empty());
		assert(!preset.description.empty());
		assert(preset.steps > 0);
		assert(preset.cfgScale > 0.0f);
	}

	// Check specific presets are included
	bool foundQuality = false;
	bool foundLCM = false;
	for (const auto& preset : presets) {
		if (preset.name == "Quality") foundQuality = true;
		if (preset.name == "LCM") foundLCM = true;
	}
	assert(foundQuality);
	assert(foundLCM);

	std::cout << " ✓" << std::endl;
}

int main() {
	try {
		testPresets();
		testGetAllSamplers();
		testGetAllSchedulers();
		testGetSamplerInfo();
		testGetSchedulerInfo();
		testGetSamplerByName();
		testGetSchedulerByName();
		testGetSamplerName();
		testGetSchedulerName();
		testRecommendedCombinations();
		testIsValidStepCount();
		testGetRecommendedSteps();
		testGetAllPresets();

		std::cout << "\n✅ All sampling helpers tests passed!" << std::endl;
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
		return 1;
	}
}
