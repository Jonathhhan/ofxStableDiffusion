#pragma once

#include "ofMain.h"
#include "../core/ofxStableDiffusionTypes.h"

#include <cstdint>
#include <string>
#include <vector>

enum class ofxStableDiffusionLongVideoPreset {
	FastPreview = 0,
	LowVram,
	Balanced,
	Quality,
	BatchStoryboard
};

struct ofxStableDiffusionLongVideoChunk {
	std::string id;
	std::string title;
	std::string prompt;
	std::string negativePrompt;
	std::string sectionGoal;
	std::string continuityNote;
	int width = 640;
	int height = 384;
	int frameCount = 49;
	int fps = 12;
	int sampleSteps = 20;
	float cfgScale = 6.5f;
	float strength = 0.65f;
	int64_t seed = -1;
	bool usePreviousLastFrame = true;
	std::string outputPrefix = "chunk";
};

struct ofxStableDiffusionLongVideoManifest {
	std::string projectName;
	std::string conceptText;
	std::string continuityBible;
	std::string outputDirectory;
	ofxStableDiffusionLongVideoPreset preset =
		ofxStableDiffusionLongVideoPreset::Balanced;
	bool lowVram = false;
	bool resumeEnabled = true;
	std::vector<ofxStableDiffusionLongVideoChunk> chunks;
};

struct ofxStableDiffusionLongVideoValidation {
	bool ok = true;
	std::vector<std::string> errors;
	std::vector<std::string> warnings;
};

struct ofxStableDiffusionLongVideoChunkResult {
	std::string chunkId;
	bool success = false;
	std::string error;
	std::string clipDirectory;
	std::string metadataPath;
	std::string manifestPath;
	int64_t actualSeed = -1;
	int renderedFrameCount = 0;
};

struct ofxStableDiffusionLongVideoRunResult {
	bool success = false;
	std::string error;
	std::vector<ofxStableDiffusionLongVideoChunkResult> chunks;
	std::string playlistManifestJson;
};
