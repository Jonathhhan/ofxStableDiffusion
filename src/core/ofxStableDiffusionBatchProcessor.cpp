#include "ofxStableDiffusionBatchProcessor.h"
#include "ofxStableDiffusion.h"
#include <cmath>

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
	ofLogWarning("ofxStableDiffusionBatchProcessor") << "generateGrid is not yet implemented; no images were generated";
	batchResult.failureCount = static_cast<int>(settings.xValues.size() * settings.yValues.size());
	return batchResult;
}

ofxStableDiffusionSweepResult ofxStableDiffusionBatchProcessor::parameterSweep(
	const ofxStableDiffusionSweepSettings& settings) {

	ofxStableDiffusionSweepResult sweepResult;
	sweepResult.parameter = settings.parameter;
	ofLogWarning("ofxStableDiffusionBatchProcessor") << "parameterSweep is not yet implemented; no images were generated";
	return sweepResult;
}

ofxStableDiffusionComparisonResult ofxStableDiffusionBatchProcessor::compareAB(
	const ofxStableDiffusionImageRequest& requestA,
	const ofxStableDiffusionImageRequest& requestB) {

	ofxStableDiffusionComparisonResult comparison;
	ofLogWarning("ofxStableDiffusionBatchProcessor") << "compareAB is not yet implemented; no images were generated";
	return comparison;
}

ofxStableDiffusionBatchResult ofxStableDiffusionBatchProcessor::processBatch(
	const std::vector<ofxStableDiffusionImageRequest>& requests,
	const std::string& outputDirectory) {

	ofxStableDiffusionBatchResult batchResult;
	ofLogWarning("ofxStableDiffusionBatchProcessor") << "processBatch is not yet implemented; no images were generated";
	batchResult.failureCount = static_cast<int>(requests.size());
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
			request.sampleMethod = static_cast<sample_method_t>(static_cast<int>(value));
			break;
		case ofxStableDiffusionParameter::Schedule:
			request.schedule = static_cast<scheduler_t>(static_cast<int>(value));
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
			return static_cast<float>(request.sampleMethod);
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
