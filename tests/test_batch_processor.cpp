#include "core/ofxStableDiffusionBatchProcessor.h"
#include "ofxStableDiffusion.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <vector>

namespace {

bool expect(bool condition, const std::string& message) {
	if (!condition) {
		std::cerr << "FAIL: " << message << std::endl;
	}
	return condition;
}

ofxStableDiffusionResult makeImageResult(bool success, int64_t seed, unsigned char color) {
	ofxStableDiffusionResult result;
	result.success = success;
	result.actualSeedUsed = seed;
	result.selectedImageIndex = 0;
	if (!success) {
		result.error = "generation failed";
		return result;
	}

	std::vector<unsigned char> pixels = {
		color, 0, 0,
		0, color, 0,
		0, 0, color,
		color, color, color
	};
	ofxStableDiffusionImageFrame frame;
	frame.index = 0;
	frame.isSelected = true;
	frame.seed = seed;
	frame.pixels.setFromPixels(pixels.data(), 2, 2, OF_IMAGE_COLOR);
	result.images.push_back(frame);
	return result;
}

bool testProcessBatchCollectsResults() {
	bool ok = true;
	ofxStableDiffusion generator;
	generator.setQueuedResults({
		makeImageResult(true, 11, 64),
		makeImageResult(false, 12, 0)
	});

	ofxStableDiffusionBatchProcessor processor;
	processor.setGenerator(&generator);

	std::vector<ofxStableDiffusionImageRequest> requests(2);
	requests[0].prompt = "first";
	requests[1].prompt = "second";

	const auto result = processor.processBatch(
		requests,
		"/tmp/ofxStableDiffusion-batch-test/process");

	ok &= expect(result.results.size() == 2, "processBatch returns both results");
	ok &= expect(result.successCount == 1, "processBatch counts successes");
	ok &= expect(result.failureCount == 1, "processBatch counts failures");
	ok &= expect(generator.getImageRequests().size() == 2, "processBatch invoked generator twice");
	ok &= expect(
		result.exportMetadata("/tmp/ofxStableDiffusion-batch-test/process/metadata.json"),
		"processBatch exports metadata");
	ok &= expect(
		std::filesystem::exists("/tmp/ofxStableDiffusion-batch-test/process/metadata.json"),
		"metadata file exists");
	return ok;
}

bool testParameterSweepTracksBestScore() {
	bool ok = true;
	ofxStableDiffusion generator;
	generator.setQueuedResults({
		makeImageResult(true, 10, 32),
		makeImageResult(true, 30, 96),
		makeImageResult(true, 20, 64)
	});

	ofxStableDiffusionBatchProcessor processor;
	processor.setGenerator(&generator);
	processor.setQualityScoringFunction([](const ofxStableDiffusionResult& result) {
		return static_cast<float>(result.actualSeedUsed);
	});

	ofxStableDiffusionSweepSettings settings;
	settings.baseRequest.prompt = "sweep";
	settings.parameter = ofxStableDiffusionParameter::CfgScale;
	settings.rangeMin = 1.0f;
	settings.rangeMax = 3.0f;
	settings.steps = 3;

	const auto sweep = processor.parameterSweep(settings);

	ok &= expect(sweep.results.size() == 3, "parameterSweep returns all entries");
	ok &= expect(sweep.bestIndex == 1, "parameterSweep selects the best score");
	ok &= expect(sweep.bestValue == 2.0f, "parameterSweep tracks the best parameter value");
	return ok;
}

bool testGenerateGridAppliesParametersAndComparisonExports() {
	bool ok = true;
	ofxStableDiffusion generator;
	generator.setQueuedResults({
		makeImageResult(true, 1, 80),
		makeImageResult(true, 2, 120),
		makeImageResult(true, 3, 160),
		makeImageResult(true, 4, 200)
	});

	ofxStableDiffusionBatchProcessor processor;
	processor.setGenerator(&generator);

	ofxStableDiffusionGridSettings settings;
	settings.baseRequest.prompt = "grid";
	settings.baseRequest.sampleSteps = 8;
	settings.xAxis = ofxStableDiffusionParameter::CfgScale;
	settings.xValues = {2.5f, 4.0f};
	settings.yAxis = ofxStableDiffusionParameter::SampleSteps;
	settings.yValues = {16.0f};
	settings.outputPath = "/tmp/ofxStableDiffusion-batch-test/grid";

	const auto grid = processor.generateGrid(settings);

	ok &= expect(grid.successCount == 2, "generateGrid runs all cells");
	ok &= expect(generator.getImageRequests().size() >= 2, "generateGrid submitted requests");
	ok &= expect(generator.getImageRequests()[0].cfgScale == 2.5f, "grid x-axis applied");
	ok &= expect(generator.getImageRequests()[0].sampleSteps == 16, "grid y-axis applied");
	ok &= expect(
		std::filesystem::exists("/tmp/ofxStableDiffusion-batch-test/grid/grid.png"),
		"generateGrid exports a composed grid image");

	const auto comparison = processor.compareAB(
		generator.getImageRequests()[0],
		generator.getImageRequests()[1]);
	ok &= expect(
		comparison.exportComparison("/tmp/ofxStableDiffusion-batch-test/grid/comparison.png"),
		"compareAB exports a side-by-side image");
	ok &= expect(
		std::filesystem::exists("/tmp/ofxStableDiffusion-batch-test/grid/comparison.png"),
		"comparison image exists");
	return ok;
}

} // namespace

int main() {
	std::filesystem::remove_all("/tmp/ofxStableDiffusion-batch-test");
	bool ok = true;
	ok &= testProcessBatchCollectsResults();
	ok &= testParameterSweepTracksBestScore();
	ok &= testGenerateGridAppliesParametersAndComparisonExports();
	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
