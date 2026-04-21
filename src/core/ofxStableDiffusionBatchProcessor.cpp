#include "ofxStableDiffusionBatchProcessor.h"
#include "ofxStableDiffusion.h"
#include <cmath>
#include <sstream>

ofxStableDiffusionBatchProcessor::ofxStableDiffusionBatchProcessor() {
	// Default quality scoring: simple placeholder
	qualityScoreFunc = [](const ofxStableDiffusionResult& result) {
		return result.success ? 1.0f : 0.0f;
	};
}

ofxStableDiffusionBatchProcessor::~ofxStableDiffusionBatchProcessor() {
	cancel();
}

ofxStableDiffusionBatchResult ofxStableDiffusionBatchProcessor::generateGrid(
	const ofxStableDiffusionGridSettings& settings) {

	ofxStableDiffusionBatchResult batchResult;
	running = true;
	cancelRequested = false;

	int totalCombinations = settings.xValues.size() * settings.yValues.size();
	int currentIndex = 0;

	std::vector<std::vector<ofxStableDiffusionResult>> grid(
		settings.yValues.size(),
		std::vector<ofxStableDiffusionResult>(settings.xValues.size()));

	uint64_t startTime = ofGetElapsedTimeMillis();

	for (size_t y = 0; y < settings.yValues.size(); y++) {
		for (size_t x = 0; x < settings.xValues.size(); x++) {
			if (cancelRequested) {
				running = false;
				return batchResult;
			}

			// Create request with x and y parameter values
			ofxStableDiffusionImageRequest request = settings.baseRequest;
			applyParameterValue(request, settings.xAxis, settings.xValues[x]);
			applyParameterValue(request, settings.yAxis, settings.yValues[y]);

			if (progressCallback) {
				std::ostringstream oss;
				oss << "Grid [" << y << "," << x << "] - "
					<< getParameterName(settings.yAxis) << "=" << settings.yValues[y]
					<< ", " << getParameterName(settings.xAxis) << "=" << settings.xValues[x];
				progressCallback(currentIndex, totalCombinations, oss.str());
			}

			// Note: This is a placeholder. In real implementation, this would use
			// ofxStableDiffusion instance to actually generate the images
			// For now, we create a placeholder result
			ofxStableDiffusionResult result;
			result.success = true;
			grid[y][x] = result;

			currentIndex++;
		}
	}

	uint64_t endTime = ofGetElapsedTimeMillis();
	batchResult.totalTimeSeconds = (endTime - startTime) / 1000.0f;
	batchResult.successCount = totalCombinations;

	// Store metadata
	batchResult.metadata["grid_type"] = "x_y_grid";
	batchResult.metadata["x_axis"] = getParameterName(settings.xAxis);
	batchResult.metadata["y_axis"] = getParameterName(settings.yAxis);
	batchResult.metadata["x_count"] = std::to_string(settings.xValues.size());
	batchResult.metadata["y_count"] = std::to_string(settings.yValues.size());

	running = false;
	return batchResult;
}

ofxStableDiffusionSweepResult ofxStableDiffusionBatchProcessor::parameterSweep(
	const ofxStableDiffusionSweepSettings& settings) {

	ofxStableDiffusionSweepResult sweepResult;
	sweepResult.parameter = settings.parameter;
	running = true;
	cancelRequested = false;

	auto stepValues = generateStepValues(
		settings.rangeMin,
		settings.rangeMax,
		settings.steps,
		settings.stepMode);

	float bestScore = -1.0f;

	for (size_t i = 0; i < stepValues.size(); i++) {
		if (cancelRequested) {
			running = false;
			return sweepResult;
		}

		float value = stepValues[i];

		ofxStableDiffusionImageRequest request = settings.baseRequest;
		applyParameterValue(request, settings.parameter, value);

		if (progressCallback) {
			std::ostringstream oss;
			oss << "Sweep " << getParameterName(settings.parameter)
				<< "=" << value << " (" << (i + 1) << "/" << stepValues.size() << ")";
			progressCallback(i, stepValues.size(), oss.str());
		}

		// Placeholder result
		ofxStableDiffusionResult result;
		result.success = true;

		float score = qualityScoreFunc(result);

		ofxStableDiffusionSweepResult::Entry entry;
		entry.parameterValue = value;
		entry.result = result;
		entry.qualityScore = score;
		sweepResult.results.push_back(entry);

		if (score > bestScore) {
			bestScore = score;
			sweepResult.bestValue = value;
			sweepResult.bestIndex = i;
		}
	}

	running = false;
	return sweepResult;
}

ofxStableDiffusionComparisonResult ofxStableDiffusionBatchProcessor::compareAB(
	const ofxStableDiffusionImageRequest& requestA,
	const ofxStableDiffusionImageRequest& requestB) {

	ofxStableDiffusionComparisonResult comparison;
	running = true;

	if (progressCallback) {
		progressCallback(0, 2, "Generating option A");
	}

	// Placeholder results
	comparison.resultA.success = true;
	comparison.nameA = "Option A";

	if (progressCallback) {
		progressCallback(1, 2, "Generating option B");
	}

	comparison.resultB.success = true;
	comparison.nameB = "Option B";

	comparison.scoreA = qualityScoreFunc(comparison.resultA);
	comparison.scoreB = qualityScoreFunc(comparison.resultB);

	running = false;
	return comparison;
}

ofxStableDiffusionBatchResult ofxStableDiffusionBatchProcessor::processBatch(
	const std::vector<ofxStableDiffusionImageRequest>& requests,
	const std::string& outputDirectory) {

	ofxStableDiffusionBatchResult batchResult;
	running = true;
	cancelRequested = false;

	uint64_t startTime = ofGetElapsedTimeMillis();

	for (size_t i = 0; i < requests.size(); i++) {
		if (cancelRequested) {
			running = false;
			return batchResult;
		}

		if (progressCallback) {
			std::ostringstream oss;
			oss << "Processing request " << (i + 1) << "/" << requests.size();
			progressCallback(i, requests.size(), oss.str());
		}

		// Placeholder result
		ofxStableDiffusionResult result;
		result.success = true;
		batchResult.results.push_back(result);
		batchResult.successCount++;
	}

	uint64_t endTime = ofGetElapsedTimeMillis();
	batchResult.totalTimeSeconds = (endTime - startTime) / 1000.0f;

	running = false;
	return batchResult;
}

void ofxStableDiffusionBatchProcessor::setProgressCallback(
	std::function<void(int, int, const std::string&)> callback) {
	progressCallback = callback;
}

void ofxStableDiffusionBatchProcessor::setQualityScoringFunction(
	std::function<float(const ofxStableDiffusionResult&)> scoreFunc) {
	qualityScoreFunc = scoreFunc;
}

void ofxStableDiffusionBatchProcessor::cancel() {
	cancelRequested = true;
}

bool ofxStableDiffusionBatchProcessor::isRunning() const {
	return running;
}

void ofxStableDiffusionBatchProcessor::applyParameterValue(
	ofxStableDiffusionImageRequest& request,
	ofxStableDiffusionParameter param,
	float value) {

	switch (param) {
		case ofxStableDiffusionParameter::CfgScale:
			request.cfgScale = value;
			break;
		case ofxStableDiffusionParameter::SampleSteps:
			request.sampleSteps = static_cast<int>(value);
			break;
		case ofxStableDiffusionParameter::Strength:
			request.strength = value;
			break;
		case ofxStableDiffusionParameter::Seed:
			request.seed = static_cast<int>(value);
			break;
		case ofxStableDiffusionParameter::Width:
			request.width = static_cast<int>(value);
			break;
		case ofxStableDiffusionParameter::Height:
			request.height = static_cast<int>(value);
			break;
		case ofxStableDiffusionParameter::SamplerMethod:
			request.samplerMethod = static_cast<sample_method_t>(static_cast<int>(value));
			break;
		case ofxStableDiffusionParameter::Schedule:
			request.schedule = static_cast<schedule_t>(static_cast<int>(value));
			break;
		case ofxStableDiffusionParameter::BatchCount:
			request.batchCount = static_cast<int>(value);
			break;
	}
}

float ofxStableDiffusionBatchProcessor::getParameterValue(
	const ofxStableDiffusionImageRequest& request,
	ofxStableDiffusionParameter param) const {

	switch (param) {
		case ofxStableDiffusionParameter::CfgScale:
			return request.cfgScale;
		case ofxStableDiffusionParameter::SampleSteps:
			return static_cast<float>(request.sampleSteps);
		case ofxStableDiffusionParameter::Strength:
			return request.strength;
		case ofxStableDiffusionParameter::Seed:
			return static_cast<float>(request.seed);
		case ofxStableDiffusionParameter::Width:
			return static_cast<float>(request.width);
		case ofxStableDiffusionParameter::Height:
			return static_cast<float>(request.height);
		case ofxStableDiffusionParameter::SamplerMethod:
			return static_cast<float>(request.samplerMethod);
		case ofxStableDiffusionParameter::Schedule:
			return static_cast<float>(request.schedule);
		case ofxStableDiffusionParameter::BatchCount:
			return static_cast<float>(request.batchCount);
		default:
			return 0.0f;
	}
}

std::string ofxStableDiffusionBatchProcessor::getParameterName(
	ofxStableDiffusionParameter param) const {

	switch (param) {
		case ofxStableDiffusionParameter::CfgScale: return "CFG Scale";
		case ofxStableDiffusionParameter::SampleSteps: return "Sample Steps";
		case ofxStableDiffusionParameter::Strength: return "Strength";
		case ofxStableDiffusionParameter::Seed: return "Seed";
		case ofxStableDiffusionParameter::Width: return "Width";
		case ofxStableDiffusionParameter::Height: return "Height";
		case ofxStableDiffusionParameter::SamplerMethod: return "Sampler Method";
		case ofxStableDiffusionParameter::Schedule: return "Schedule";
		case ofxStableDiffusionParameter::BatchCount: return "Batch Count";
		default: return "Unknown";
	}
}

std::vector<float> ofxStableDiffusionBatchProcessor::generateStepValues(
	float minVal,
	float maxVal,
	int steps,
	ofxStableDiffusionStepMode mode) const {

	std::vector<float> values;
	if (steps <= 0) return values;
	if (steps == 1) {
		values.push_back((minVal + maxVal) / 2.0f);
		return values;
	}

	switch (mode) {
		case ofxStableDiffusionStepMode::Linear: {
			float step = (maxVal - minVal) / (steps - 1);
			for (int i = 0; i < steps; i++) {
				values.push_back(minVal + i * step);
			}
			break;
		}
		case ofxStableDiffusionStepMode::Logarithmic: {
			if (minVal <= 0.0f) minVal = 0.001f; // Avoid log(0)
			float logMin = std::log(minVal);
			float logMax = std::log(maxVal);
			float step = (logMax - logMin) / (steps - 1);
			for (int i = 0; i < steps; i++) {
				values.push_back(std::exp(logMin + i * step));
			}
			break;
		}
	}

	return values;
}

bool ofxStableDiffusionBatchResult::exportMetadata(const std::string& path) const {
	ofJson json;
	json["total_time_seconds"] = totalTimeSeconds;
	json["success_count"] = successCount;
	json["failure_count"] = failureCount;
	json["metadata"] = metadata;

	return ofSaveJson(path, json);
}

bool ofxStableDiffusionComparisonResult::exportComparison(const std::string& path) const {
	// Placeholder: would create side-by-side comparison image
	ofLogNotice("ofxStableDiffusionBatchProcessor")
		<< "Comparison export to " << path << " (placeholder implementation)";
	return true;
}
