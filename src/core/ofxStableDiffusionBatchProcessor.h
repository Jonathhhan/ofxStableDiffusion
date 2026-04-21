#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionTypes.h"
#include <vector>
#include <map>
#include <functional>

/// Parameter types for batch processing
enum class ofxStableDiffusionParameter {
	CfgScale,
	SampleSteps,
	Strength,
	Seed,
	Width,
	Height,
	SamplerMethod,
	Schedule,
	BatchCount
};

/// Step mode for parameter sweeps
enum class ofxStableDiffusionStepMode {
	Linear,       // Equal steps
	Logarithmic   // Logarithmic steps
};

/// Result from batch processing
struct ofxStableDiffusionBatchResult {
	std::vector<ofxStableDiffusionResult> results;
	std::map<std::string, std::string> metadata;
	float totalTimeSeconds = 0.0f;
	int successCount = 0;
	int failureCount = 0;

	/// Export metadata to JSON file
	/// @param path Output file path
	/// @return True if successful
	bool exportMetadata(const std::string& path) const;
};

/// Settings for X/Y/Z grid generation
struct ofxStableDiffusionGridSettings {
	ofxStableDiffusionImageRequest baseRequest;
	ofxStableDiffusionParameter xAxis;
	std::vector<float> xValues;
	ofxStableDiffusionParameter yAxis;
	std::vector<float> yValues;
	std::string outputPath;
	int gridCellWidth = 512;
	int gridCellHeight = 512;
	bool addLabels = true;
};

/// Settings for parameter sweeps
struct ofxStableDiffusionSweepSettings {
	ofxStableDiffusionImageRequest baseRequest;
	ofxStableDiffusionParameter parameter;
	float rangeMin = 0.0f;
	float rangeMax = 1.0f;
	int steps = 10;
	ofxStableDiffusionStepMode stepMode = ofxStableDiffusionStepMode::Linear;
};

/// Result from parameter sweep
struct ofxStableDiffusionSweepResult {
	struct Entry {
		float parameterValue;
		ofxStableDiffusionResult result;
		float qualityScore = 0.0f;
	};

	std::vector<Entry> results;
	ofxStableDiffusionParameter parameter;
	float bestValue = 0.0f;
	int bestIndex = -1;
};

/// Result from A/B comparison
struct ofxStableDiffusionComparisonResult {
	ofxStableDiffusionResult resultA;
	ofxStableDiffusionResult resultB;
	std::string nameA;
	std::string nameB;
	float scoreA = 0.0f;
	float scoreB = 0.0f;

	/// Export side-by-side comparison image
	/// @param path Output file path
	/// @return True if successful
	bool exportComparison(const std::string& path) const;
};

/// Batch processing utilities for systematic parameter exploration
class ofxStableDiffusionBatchProcessor {
public:
	ofxStableDiffusionBatchProcessor();
	~ofxStableDiffusionBatchProcessor();

	/// Generate X/Y parameter grid
	/// @param settings Grid generation settings
	/// @return Grid result with composite image
	ofxStableDiffusionBatchResult generateGrid(const ofxStableDiffusionGridSettings& settings);

	/// Perform parameter sweep
	/// @param settings Sweep settings
	/// @return Sweep results with all generated images
	ofxStableDiffusionSweepResult parameterSweep(const ofxStableDiffusionSweepSettings& settings);

	/// Compare two requests side-by-side
	/// @param requestA First request
	/// @param requestB Second request
	/// @return Comparison result
	ofxStableDiffusionComparisonResult compareAB(
		const ofxStableDiffusionImageRequest& requestA,
		const ofxStableDiffusionImageRequest& requestB);

	/// Process multiple requests in batch
	/// @param requests Vector of requests to process
	/// @param outputDirectory Directory for output files
	/// @return Batch results
	ofxStableDiffusionBatchResult processBatch(
		const std::vector<ofxStableDiffusionImageRequest>& requests,
		const std::string& outputDirectory = "");

	/// Set progress callback for batch operations
	/// @param callback Progress callback function
	void setProgressCallback(std::function<void(int current, int total, const std::string& status)> callback);

	/// Set quality scoring function for ranking results
	/// @param scoreFunc Function that scores a result (higher is better)
	void setQualityScoringFunction(std::function<float(const ofxStableDiffusionResult&)> scoreFunc);

	/// Cancel current batch operation
	void cancel();

	/// Check if batch operation is running
	/// @return True if running
	bool isRunning() const;

private:
	void applyParameterValue(
		ofxStableDiffusionImageRequest& request,
		ofxStableDiffusionParameter param,
		float value);

	float getParameterValue(
		const ofxStableDiffusionImageRequest& request,
		ofxStableDiffusionParameter param) const;

	std::string getParameterName(ofxStableDiffusionParameter param) const;

	std::vector<float> generateStepValues(
		float minVal,
		float maxVal,
		int steps,
		ofxStableDiffusionStepMode mode) const;

	ofImage createGridImage(
		const std::vector<std::vector<ofxStableDiffusionResult>>& grid,
		const ofxStableDiffusionGridSettings& settings) const;

	std::function<void(int, int, const std::string&)> progressCallback;
	std::function<float(const ofxStableDiffusionResult&)> qualityScoreFunc;
	bool running = false;
	bool cancelRequested = false;
};
