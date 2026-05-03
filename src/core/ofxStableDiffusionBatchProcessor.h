#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionTypes.h"
#include <functional>
#include <map>
#include <vector>

class ofxStableDiffusion;

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

/// Experimental batch-processing scaffold for systematic parameter exploration.
///
/// The request/result structs, parameter helpers, and metadata export are available
/// today, but the generation methods below do not run native image generation yet.
/// They currently return placeholder/empty results while logging a warning.
class ofxStableDiffusionBatchProcessor {
public:
	ofxStableDiffusionBatchProcessor();
	~ofxStableDiffusionBatchProcessor();

	/// Attach the generator used to execute batch requests.
	/// @param sd Generator instance to use (not owned)
	void setGenerator(ofxStableDiffusion* sd);

	/// Get the attached generator.
	ofxStableDiffusion* getGenerator() const;

	/// Check whether a generator is attached.
	bool hasGenerator() const;

	/// Generate X/Y parameter grid
	/// @param settings Grid generation settings
	/// @return Batch result containing one entry per grid cell.
	ofxStableDiffusionBatchResult generateGrid(const ofxStableDiffusionGridSettings& settings);

	/// Perform parameter sweep
	/// @param settings Sweep settings
	/// @return Sweep result ordered by sweep value.
	ofxStableDiffusionSweepResult parameterSweep(const ofxStableDiffusionSweepSettings& settings);

	/// Compare two requests side-by-side
	/// @param requestA First request
	/// @param requestB Second request
	/// @return Comparison result with generated outputs and quality scores.
	ofxStableDiffusionComparisonResult compareAB(
		const ofxStableDiffusionImageRequest& requestA,
		const ofxStableDiffusionImageRequest& requestB);

	/// Process multiple requests in batch
	/// @param requests Vector of requests to process
	/// @param outputDirectory Directory for output files
	/// @return Batch result containing every generated request result.
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

	/// Set how often blocking batch waits poll the async generator.
	void setPollIntervalMs(int pollIntervalMs);
	int getPollIntervalMs() const;

	/// Set the maximum wait time for one batch request before it is marked failed.
	void setExecutionTimeoutMs(int timeoutMs);
	int getExecutionTimeoutMs() const;

private:
	bool runImageRequest(
		const ofxStableDiffusionImageRequest& request,
		ofxStableDiffusionResult& result,
		std::string& errorMessage);
	bool waitForCurrentGeneration(
		ofxStableDiffusionResult& result,
		std::string& errorMessage);
	float scoreResult(const ofxStableDiffusionResult& result) const;
	void recordBatchEntry(
		ofxStableDiffusionBatchResult& batchResult,
		const ofxStableDiffusionResult& result,
		int entryIndex,
		const std::map<std::string, std::string>& entryMetadata);
	bool exportResultImage(
		const ofxStableDiffusionResult& result,
		const std::string& outputPath) const;

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
	ofxStableDiffusion* generator = nullptr;
	bool running = false;
	bool cancelRequested = false;
	int pollIntervalMs = 10;
	int executionTimeoutMs = 600000;
};
