#include "ofxStableDiffusionBatchProcessor.h"

#include "ofxStableDiffusion.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <thread>

namespace {

constexpr float kMinimumPositiveLogValue = 0.001f;

std::string formatBatchFloat(float value) {
	std::ostringstream stream;
	stream.setf(std::ios::fixed);
	stream.precision(3);
	stream << value;
	std::string formatted = stream.str();
	while (!formatted.empty() && formatted.back() == '0') {
		formatted.pop_back();
	}
	if (!formatted.empty() && formatted.back() == '.') {
		formatted.pop_back();
	}
	return formatted.empty() ? "0" : formatted;
}

const ofxStableDiffusionImageFrame* pickRepresentativeImage(
	const ofxStableDiffusionResult& result) {
	if (result.selectedImageIndex >= 0 &&
		result.selectedImageIndex < static_cast<int>(result.images.size())) {
		return &result.images[static_cast<std::size_t>(result.selectedImageIndex)];
	}

	for (const auto& image : result.images) {
		if (image.isSelected) {
			return &image;
		}
	}

	for (const auto& image : result.images) {
		if (image.isAllocated()) {
			return &image;
		}
	}

	return nullptr;
}

ofImageType imageTypeForChannels(int channels) {
	switch (channels) {
	case 1: return OF_IMAGE_GRAYSCALE;
	case 4: return OF_IMAGE_COLOR_ALPHA;
	case 3:
	default:
		return OF_IMAGE_COLOR;
	}
}

void blitScaledFrame(
	const ofxStableDiffusionImageFrame& frame,
	std::vector<unsigned char>& canvas,
	int canvasWidth,
	int canvasHeight,
	int canvasChannels,
	int destX,
	int destY,
	int destWidth,
	int destHeight) {
	const int srcWidth = frame.width();
	const int srcHeight = frame.height();
	const int srcChannels = frame.channels();
	const unsigned char* srcData = frame.pixels.getData();
	if (srcWidth <= 0 || srcHeight <= 0 || srcChannels <= 0 || srcData == nullptr) {
		return;
	}

	for (int y = 0; y < destHeight; ++y) {
		const int srcY = std::min(srcHeight - 1, (y * srcHeight) / std::max(1, destHeight));
		for (int x = 0; x < destWidth; ++x) {
			const int srcX = std::min(srcWidth - 1, (x * srcWidth) / std::max(1, destWidth));
			const std::size_t srcOffset =
				(static_cast<std::size_t>(srcY) * static_cast<std::size_t>(srcWidth) +
					static_cast<std::size_t>(srcX)) * static_cast<std::size_t>(srcChannels);
			const std::size_t destOffset =
				(static_cast<std::size_t>(destY + y) * static_cast<std::size_t>(canvasWidth) +
					static_cast<std::size_t>(destX + x)) * static_cast<std::size_t>(canvasChannels);

			for (int channel = 0; channel < canvasChannels; ++channel) {
				const int srcChannel = std::min(channel, srcChannels - 1);
				canvas[destOffset + static_cast<std::size_t>(channel)] =
					srcData[srcOffset + static_cast<std::size_t>(srcChannel)];
			}
		}
	}
}

bool ensureParentDirectory(const std::string& outputPath) {
	if (outputPath.empty()) {
		return false;
	}

	std::error_code error;
	const std::filesystem::path path(outputPath);
	const auto parent = path.parent_path();
	if (parent.empty()) {
		return true;
	}
	if (std::filesystem::exists(parent, error)) {
		return true;
	}
	return std::filesystem::create_directories(parent, error);
}

std::string defaultEntryLabel(int index) {
	return "run_" + ofToString(index);
}

} // namespace

ofxStableDiffusionBatchProcessor::ofxStableDiffusionBatchProcessor() {
	qualityScoreFunc = [](const ofxStableDiffusionResult& result) {
		return result.success ? 1.0f : 0.0f;
	};
}

ofxStableDiffusionBatchProcessor::~ofxStableDiffusionBatchProcessor() {
	cancel();
}

void ofxStableDiffusionBatchProcessor::setGenerator(ofxStableDiffusion* sd) {
	generator = sd;
}

ofxStableDiffusion* ofxStableDiffusionBatchProcessor::getGenerator() const {
	return generator;
}

bool ofxStableDiffusionBatchProcessor::hasGenerator() const {
	return generator != nullptr;
}

ofxStableDiffusionBatchResult ofxStableDiffusionBatchProcessor::generateGrid(
	const ofxStableDiffusionGridSettings& settings) {
	ofxStableDiffusionBatchResult batchResult;
	if (!generator) {
		batchResult.failureCount = static_cast<int>(settings.xValues.size() * settings.yValues.size());
		batchResult.metadata["error"] = "No generator attached";
		return batchResult;
	}

	const auto startedAt = std::chrono::steady_clock::now();
	running = true;
	cancelRequested = false;
	const int totalCells =
		static_cast<int>(settings.xValues.size() * settings.yValues.size());
	std::vector<std::vector<ofxStableDiffusionResult>> gridResults(
		settings.yValues.size(),
		std::vector<ofxStableDiffusionResult>(settings.xValues.size()));
	int entryIndex = 0;

	for (std::size_t y = 0; y < settings.yValues.size() && !cancelRequested; ++y) {
		for (std::size_t x = 0; x < settings.xValues.size() && !cancelRequested; ++x) {
			if (progressCallback) {
				progressCallback(
					entryIndex,
					totalCells,
					"Generating grid cell " + ofToString(entryIndex + 1) + "/" +
						ofToString(totalCells));
			}

			ofxStableDiffusionImageRequest request = settings.baseRequest;
			applyParameterValue(request, settings.xAxis, settings.xValues[x]);
			applyParameterValue(request, settings.yAxis, settings.yValues[y]);

			ofxStableDiffusionResult result;
			std::string errorMessage;
			if (!runImageRequest(request, result, errorMessage)) {
				result.success = false;
				result.error = errorMessage;
			}
			gridResults[y][x] = result;

			const float score = scoreResult(result);
			recordBatchEntry(
				batchResult,
				result,
				entryIndex,
				{
					{"label", defaultEntryLabel(entryIndex)},
					{"x_parameter", getParameterName(settings.xAxis)},
					{"x_value", formatBatchFloat(settings.xValues[x])},
					{"y_parameter", getParameterName(settings.yAxis)},
					{"y_value", formatBatchFloat(settings.yValues[y])},
					{"quality_score", formatBatchFloat(score)}
				});

			if (!settings.outputPath.empty()) {
				const std::filesystem::path outputRoot(settings.outputPath);
				exportResultImage(
					result,
					(outputRoot / ("grid_" + ofToString(static_cast<int>(y)) + "_" +
						ofToString(static_cast<int>(x)) + ".png")).string());
			}

			++entryIndex;
		}
	}

	if (!settings.outputPath.empty()) {
		ensureParentDirectory((std::filesystem::path(settings.outputPath) / "grid.png").string());
		createGridImage(gridResults, settings)
			.save((std::filesystem::path(settings.outputPath) / "grid.png").string());
	}

	batchResult.metadata["grid_x_axis"] = getParameterName(settings.xAxis);
	batchResult.metadata["grid_y_axis"] = getParameterName(settings.yAxis);
	batchResult.metadata["cancelled"] = cancelRequested ? "true" : "false";
	batchResult.totalTimeSeconds = std::chrono::duration<float>(
		std::chrono::steady_clock::now() - startedAt).count();
	running = false;
	return batchResult;
}

ofxStableDiffusionSweepResult ofxStableDiffusionBatchProcessor::parameterSweep(
	const ofxStableDiffusionSweepSettings& settings) {
	ofxStableDiffusionSweepResult sweepResult;
	sweepResult.parameter = settings.parameter;
	if (!generator) {
		return sweepResult;
	}

	running = true;
	cancelRequested = false;
	const auto values = generateStepValues(
		settings.rangeMin,
		settings.rangeMax,
		settings.steps,
		settings.stepMode);
	float bestScore = -std::numeric_limits<float>::infinity();

	for (std::size_t index = 0; index < values.size() && !cancelRequested; ++index) {
		if (progressCallback) {
			progressCallback(
				static_cast<int>(index),
				static_cast<int>(values.size()),
				"Sweeping " + getParameterName(settings.parameter));
		}

		ofxStableDiffusionImageRequest request = settings.baseRequest;
		applyParameterValue(request, settings.parameter, values[index]);

		ofxStableDiffusionResult result;
		std::string errorMessage;
		runImageRequest(request, result, errorMessage);
		if (!errorMessage.empty() && !result.success) {
			result.error = errorMessage;
		}

		const float score = scoreResult(result);
		sweepResult.results.push_back({values[index], result, score});
		if (score > bestScore) {
			bestScore = score;
			sweepResult.bestIndex = static_cast<int>(index);
			sweepResult.bestValue = values[index];
		}
	}

	running = false;
	return sweepResult;
}

ofxStableDiffusionComparisonResult ofxStableDiffusionBatchProcessor::compareAB(
	const ofxStableDiffusionImageRequest& requestA,
	const ofxStableDiffusionImageRequest& requestB) {
	ofxStableDiffusionComparisonResult comparison;
	comparison.nameA = "A";
	comparison.nameB = "B";
	if (!generator) {
		comparison.resultA.error = "No generator attached";
		comparison.resultB.error = "No generator attached";
		return comparison;
	}

	running = true;
	cancelRequested = false;

	std::string errorMessage;
	runImageRequest(requestA, comparison.resultA, errorMessage);
	if (!comparison.resultA.success && comparison.resultA.error.empty()) {
		comparison.resultA.error = errorMessage;
	}
	comparison.scoreA = scoreResult(comparison.resultA);

	errorMessage.clear();
	if (!cancelRequested) {
		runImageRequest(requestB, comparison.resultB, errorMessage);
		if (!comparison.resultB.success && comparison.resultB.error.empty()) {
			comparison.resultB.error = errorMessage;
		}
		comparison.scoreB = scoreResult(comparison.resultB);
	}

	running = false;
	return comparison;
}

ofxStableDiffusionBatchResult ofxStableDiffusionBatchProcessor::processBatch(
	const std::vector<ofxStableDiffusionImageRequest>& requests,
	const std::string& outputDirectory) {
	ofxStableDiffusionBatchResult batchResult;
	if (!generator) {
		batchResult.failureCount = static_cast<int>(requests.size());
		batchResult.metadata["error"] = "No generator attached";
		return batchResult;
	}

	const auto startedAt = std::chrono::steady_clock::now();
	running = true;
	cancelRequested = false;

	for (std::size_t index = 0; index < requests.size() && !cancelRequested; ++index) {
		if (progressCallback) {
			progressCallback(
				static_cast<int>(index),
				static_cast<int>(requests.size()),
				"Processing batch request " + ofToString(static_cast<int>(index + 1)));
		}

		ofxStableDiffusionResult result;
		std::string errorMessage;
		runImageRequest(requests[index], result, errorMessage);
		if (!result.success && result.error.empty()) {
			result.error = errorMessage;
		}

		const float score = scoreResult(result);
		recordBatchEntry(
			batchResult,
			result,
			static_cast<int>(index),
			{
				{"label", defaultEntryLabel(static_cast<int>(index))},
				{"prompt", requests[index].prompt},
				{"quality_score", formatBatchFloat(score)}
			});

		if (!outputDirectory.empty()) {
			exportResultImage(
				result,
				(std::filesystem::path(outputDirectory) /
					(defaultEntryLabel(static_cast<int>(index)) + ".png")).string());
		}
	}

	batchResult.metadata["cancelled"] = cancelRequested ? "true" : "false";
	batchResult.totalTimeSeconds = std::chrono::duration<float>(
		std::chrono::steady_clock::now() - startedAt).count();
	running = false;
	return batchResult;
}

void ofxStableDiffusionBatchProcessor::setProgressCallback(
	std::function<void(int, int, const std::string&)> callback) {
	progressCallback = callback;
}

void ofxStableDiffusionBatchProcessor::setQualityScoringFunction(
	std::function<float(const ofxStableDiffusionResult&)> scoreFunc) {
	qualityScoreFunc = scoreFunc ? scoreFunc : [](const ofxStableDiffusionResult& result) {
		return result.success ? 1.0f : 0.0f;
	};
}

void ofxStableDiffusionBatchProcessor::cancel() {
	cancelRequested = true;
	if (running && generator) {
		generator->requestCancellation();
	}
}

bool ofxStableDiffusionBatchProcessor::isRunning() const {
	return running;
}

void ofxStableDiffusionBatchProcessor::setPollIntervalMs(int intervalMs) {
	pollIntervalMs = std::max(1, intervalMs);
}

int ofxStableDiffusionBatchProcessor::getPollIntervalMs() const {
	return pollIntervalMs;
}

void ofxStableDiffusionBatchProcessor::setExecutionTimeoutMs(int timeoutMs) {
	executionTimeoutMs = std::max(1, timeoutMs);
}

int ofxStableDiffusionBatchProcessor::getExecutionTimeoutMs() const {
	return executionTimeoutMs;
}

bool ofxStableDiffusionBatchProcessor::runImageRequest(
	const ofxStableDiffusionImageRequest& request,
	ofxStableDiffusionResult& result,
	std::string& errorMessage) {
	if (!generator) {
		errorMessage = "No generator attached";
		result = ofxStableDiffusionResult();
		result.success = false;
		result.error = errorMessage;
		return false;
	}

	if (cancelRequested) {
		errorMessage = "Batch operation cancelled";
		result = ofxStableDiffusionResult();
		result.success = false;
		result.error = errorMessage;
		return false;
	}

	generator->generate(request);
	return waitForCurrentGeneration(result, errorMessage);
}

bool ofxStableDiffusionBatchProcessor::waitForCurrentGeneration(
	ofxStableDiffusionResult& result,
	std::string& errorMessage) {
	if (!generator) {
		errorMessage = "No generator attached";
		return false;
	}

	const auto startedAt = std::chrono::steady_clock::now();
	while (generator->isGenerating()) {
		if (cancelRequested) {
			generator->requestCancellation();
		}

		const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::steady_clock::now() - startedAt).count();
		if (elapsedMs > executionTimeoutMs) {
			errorMessage = "Batch request timed out";
			result = ofxStableDiffusionResult();
			result.success = false;
			result.error = errorMessage;
			return false;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(pollIntervalMs));
	}

	result = generator->getLastResult();
	if (result.success) {
		errorMessage.clear();
		return true;
	}

	errorMessage = result.error.empty() ? "Generation failed" : result.error;
	return false;
}

float ofxStableDiffusionBatchProcessor::scoreResult(
	const ofxStableDiffusionResult& result) const {
	return qualityScoreFunc ? qualityScoreFunc(result) : 0.0f;
}

void ofxStableDiffusionBatchProcessor::recordBatchEntry(
	ofxStableDiffusionBatchResult& batchResult,
	const ofxStableDiffusionResult& result,
	int entryIndex,
	const std::map<std::string, std::string>& entryMetadata) {
	batchResult.results.push_back(result);
	if (result.success) {
		batchResult.successCount++;
	} else {
		batchResult.failureCount++;
	}

	for (const auto& entry : entryMetadata) {
		batchResult.metadata[defaultEntryLabel(entryIndex) + "." + entry.first] = entry.second;
	}
}

bool ofxStableDiffusionBatchProcessor::exportResultImage(
	const ofxStableDiffusionResult& result,
	const std::string& outputPath) const {
	const auto* image = pickRepresentativeImage(result);
	if (image == nullptr || !image->isAllocated()) {
		return false;
	}

	if (!ensureParentDirectory(outputPath)) {
		return false;
	}

	ofImage output;
	output.setFromPixels(image->pixels);
	return output.save(outputPath);
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
		case ofxStableDiffusionParameter::Seed: {
			const double rounded = std::llround(static_cast<double>(value));
			const double minSeed =
				static_cast<double>(std::numeric_limits<int64_t>::lowest());
			const double maxSeed =
				static_cast<double>(std::numeric_limits<int64_t>::max());
			request.seed = static_cast<int64_t>(
				std::max(minSeed, std::min(maxSeed, rounded)));
			break;
		}
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
			const float step = (maxVal - minVal) / static_cast<float>(steps - 1);
			float currentValue = minVal;
			for (int i = 0; i < steps; ++i) {
				values.push_back(currentValue);
				currentValue += step;
			}
			values.back() = maxVal;
			break;
		}
		case ofxStableDiffusionStepMode::Logarithmic: {
			if (minVal <= 0.0f) {
				ofLogWarning("ofxStableDiffusionBatchProcessor")
					<< "Logarithmic sweeps require positive values; clamping the minimum "
					<< "to " << kMinimumPositiveLogValue;
				minVal = kMinimumPositiveLogValue;
			}
			const float logMin = std::log(minVal);
			const float logMax = std::log(maxVal);
			const float step = (logMax - logMin) / static_cast<float>(steps - 1);
			for (int i = 0; i < steps; ++i) {
				values.push_back(std::exp(logMin + static_cast<float>(i) * step));
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
	json["results"] = ofJson::array();
	for (const auto& result : results) {
		ofJson entry;
		entry["success"] = result.success;
		entry["error"] = result.error;
		entry["elapsed_ms"] = result.elapsedMs;
		entry["actual_seed_used"] = result.actualSeedUsed;
		entry["image_count"] = static_cast<int>(result.images.size());
		entry["video_frame_count"] = static_cast<int>(result.video.frames.size());
		entry["selected_image_index"] = result.selectedImageIndex;
		json["results"].push_back(entry);
	}

	return ofSaveJson(path, json);
}

bool ofxStableDiffusionComparisonResult::exportComparison(const std::string& path) const {
	const auto* frameA = pickRepresentativeImage(resultA);
	const auto* frameB = pickRepresentativeImage(resultB);
	if (frameA == nullptr || frameB == nullptr ||
		!frameA->isAllocated() || !frameB->isAllocated()) {
		return false;
	}

	const int outputHeight = std::max(frameA->height(), frameB->height());
	const int outputWidth = frameA->width() + frameB->width();
	const int outputChannels = std::max(frameA->channels(), frameB->channels());
	if (outputHeight <= 0 || outputWidth <= 0 || outputChannels <= 0) {
		return false;
	}

	ensureParentDirectory(path);
	std::vector<unsigned char> canvas(
		static_cast<std::size_t>(outputWidth) *
			static_cast<std::size_t>(outputHeight) *
			static_cast<std::size_t>(outputChannels),
		0);

	blitScaledFrame(
		*frameA,
		canvas,
		outputWidth,
		outputHeight,
		outputChannels,
		0,
		0,
		frameA->width(),
		outputHeight);
	blitScaledFrame(
		*frameB,
		canvas,
		outputWidth,
		outputHeight,
		outputChannels,
		frameA->width(),
		0,
		frameB->width(),
		outputHeight);

	ofPixels pixels;
	pixels.setFromPixels(
		canvas.data(),
		outputWidth,
		outputHeight,
		imageTypeForChannels(outputChannels));
	ofImage output;
	output.setFromPixels(pixels);
	return output.save(path);
}

ofImage ofxStableDiffusionBatchProcessor::createGridImage(
	const std::vector<std::vector<ofxStableDiffusionResult>>& grid,
	const ofxStableDiffusionGridSettings& settings) const {
	const int rows = static_cast<int>(grid.size());
	const int cols = rows > 0 ? static_cast<int>(grid.front().size()) : 0;
	const int cellWidth = std::max(1, settings.gridCellWidth);
	const int cellHeight = std::max(1, settings.gridCellHeight);
	const int outputWidth = std::max(1, cols * cellWidth);
	const int outputHeight = std::max(1, rows * cellHeight);

	int channels = 3;
	for (const auto& row : grid) {
		for (const auto& result : row) {
			const auto* frame = pickRepresentativeImage(result);
			if (frame != nullptr && frame->isAllocated() && frame->channels() > 0) {
				channels = frame->channels();
				break;
			}
		}
	}

	std::vector<unsigned char> canvas(
		static_cast<std::size_t>(outputWidth) *
			static_cast<std::size_t>(outputHeight) *
			static_cast<std::size_t>(channels),
		0);

	for (int row = 0; row < rows; ++row) {
		for (int col = 0; col < cols; ++col) {
			const auto* frame = pickRepresentativeImage(grid[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
			if (frame == nullptr || !frame->isAllocated()) {
				continue;
			}
			blitScaledFrame(
				*frame,
				canvas,
				outputWidth,
				outputHeight,
				channels,
				col * cellWidth,
				row * cellHeight,
				cellWidth,
				cellHeight);
		}
	}

	ofPixels pixels;
	pixels.setFromPixels(
		canvas.data(),
		outputWidth,
		outputHeight,
		imageTypeForChannels(channels));
	ofImage image;
	image.setFromPixels(pixels);
	return image;
}
