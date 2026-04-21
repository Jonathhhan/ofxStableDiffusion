#include "../core/ofxStableDiffusionTypes.h"
#include "../core/ofxStableDiffusionImageHelpers.h"
#include "ofxStableDiffusionVideoHelpers.h"
#include "ofxStableDiffusionNativeVideoExport.h"

#include <functional>

namespace {

void appendFrameCopy(
	std::vector<ofxStableDiffusionImageFrame> & destination,
	const ofxStableDiffusionImageFrame & source,
	int sequenceIndex) {
	ofxStableDiffusionImageFrame frame = source;
	frame.index = sequenceIndex;
	destination.push_back(std::move(frame));
}

} // namespace

bool ofxStableDiffusionVideoClip::empty() const {
	return frames.empty();
}

std::size_t ofxStableDiffusionVideoClip::size() const {
	return frames.size();
}

float ofxStableDiffusionVideoClip::durationSeconds() const {
	return ofxStableDiffusionVideoDurationSeconds(frames.size(), fps);
}

int ofxStableDiffusionVideoClip::frameIndexForTime(float seconds) const {
	return ofxStableDiffusionVideoFrameIndexForTime(frames.size(), fps, seconds);
}

const ofxStableDiffusionImageFrame * ofxStableDiffusionVideoClip::frameForTime(float seconds) const {
	const int index = frameIndexForTime(seconds);
	if (index < 0 || index >= static_cast<int>(frames.size())) {
		return nullptr;
	}
	return &frames[static_cast<std::size_t>(index)];
}

std::vector<int64_t> ofxStableDiffusionVideoClip::seeds() const {
	std::vector<int64_t> output;
	output.reserve(frames.size());
	for (const auto& frame : frames) {
		output.push_back(frame.seed);
	}
	return output;
}

bool ofxStableDiffusionVideoClip::saveFrameSequence(
	const std::string & directory,
	const std::string & prefix) const {
	if (frames.empty()) {
		return false;
	}

	ofDirectory::createDirectory(directory, true, true);
	for (std::size_t i = 0; i < frames.size(); ++i) {
		const auto & frame = frames[i];
		if (!frame.isAllocated()) {
			return false;
		}
		const std::string filename = prefix + "_" + ofToString(static_cast<int>(i), 4, '0') + ".png";
		const std::string path = ofFilePath::join(directory, filename);
		if (!ofSaveImage(frame.pixels, path, OF_IMAGE_QUALITY_BEST)) {
			return false;
		}
	}

	return true;
}

bool ofxStableDiffusionVideoClip::saveMetadataJson(const std::string & path) const {
	if (frames.empty()) {
		return false;
	}

	ofJson root;
	root["fps"] = fps;
	root["frame_count"] = static_cast<int>(frames.size());
	root["source_frame_count"] = sourceFrameCount;
	root["mode"] = ofxStableDiffusionVideoModeName(mode);
	root["duration_seconds"] = durationSeconds();

	ofJson frameArray = ofJson::array();
	for (const auto& frame : frames) {
		ofJson frameJson;
		frameJson["index"] = frame.index;
		frameJson["source_index"] = frame.sourceIndex;
		frameJson["seed"] = frame.seed;
		frameJson["width"] = frame.width();
		frameJson["height"] = frame.height();
		frameJson["channels"] = frame.channels();
		if (!frame.generation.prompt.empty()) {
			frameJson["prompt"] = frame.generation.prompt;
		}
		if (!frame.generation.negativePrompt.empty()) {
			frameJson["negative_prompt"] = frame.generation.negativePrompt;
		}
		if (frame.generation.cfgScale >= 0.0f) {
			frameJson["cfg_scale"] = frame.generation.cfgScale;
		}
		if (frame.generation.strength >= 0.0f) {
			frameJson["strength"] = frame.generation.strength;
		}
		frameArray.push_back(std::move(frameJson));
	}

	root["frames"] = std::move(frameArray);
	return ofSavePrettyJson(path, root);
}

bool ofxStableDiffusionVideoClip::saveFrameSequenceWithMetadata(
	const std::string & directory,
	const std::string & prefix,
	const std::string & metadataFilename) const {
	if (!saveFrameSequence(directory, prefix)) {
		return false;
	}
	return saveMetadataJson(ofFilePath::join(directory, metadataFilename));
}

bool ofxStableDiffusionVideoClip::saveWebm(const std::string & path, int quality) const {
	return ofxStableDiffusionNativeVideoExport::saveWebm(path, *this, quality);
}

const char * ofxStableDiffusionTaskLabel(ofxStableDiffusionTask task) {
	switch (task) {
	case ofxStableDiffusionTask::LoadModel: return "LoadModel";
	case ofxStableDiffusionTask::TextToImage: return "TextToImage";
	case ofxStableDiffusionTask::ImageToImage: return "ImageToImage";
	case ofxStableDiffusionTask::InstructImage: return "InstructImage";
	case ofxStableDiffusionTask::ImageVariation: return "ImageVariation";
	case ofxStableDiffusionTask::ImageRestyle: return "ImageRestyle";
	case ofxStableDiffusionTask::Inpainting: return "Inpainting";
	case ofxStableDiffusionTask::ImageToVideo: return "ImageToVideo";
	case ofxStableDiffusionTask::Upscale: return "Upscale";
	case ofxStableDiffusionTask::None:
	default:
		return "None";
	}
}

const char * ofxStableDiffusionImageModeLabel(ofxStableDiffusionImageMode mode) {
	return ofxStableDiffusionImageModeName(mode);
}

const char * ofxStableDiffusionImageSelectionModeLabel(
	ofxStableDiffusionImageSelectionMode mode) {
	return ofxStableDiffusionImageSelectionModeName(mode);
}

const char * ofxStableDiffusionVideoModeLabel(ofxStableDiffusionVideoMode mode) {
	return ofxStableDiffusionVideoModeName(mode);
}

const char * ofxStableDiffusionErrorCodeLabel(ofxStableDiffusionErrorCode code) {
	switch (code) {
	case ofxStableDiffusionErrorCode::ModelNotFound: return "ModelNotFound";
	case ofxStableDiffusionErrorCode::ModelCorrupted: return "ModelCorrupted";
	case ofxStableDiffusionErrorCode::ModelLoadFailed: return "ModelLoadFailed";
	case ofxStableDiffusionErrorCode::OutOfMemory: return "OutOfMemory";
	case ofxStableDiffusionErrorCode::InvalidDimensions: return "InvalidDimensions";
	case ofxStableDiffusionErrorCode::InvalidBatchCount: return "InvalidBatchCount";
	case ofxStableDiffusionErrorCode::InvalidFrameCount: return "InvalidFrameCount";
	case ofxStableDiffusionErrorCode::InvalidParameter: return "InvalidParameter";
	case ofxStableDiffusionErrorCode::MissingInputImage: return "MissingInputImage";
	case ofxStableDiffusionErrorCode::GenerationFailed: return "GenerationFailed";
	case ofxStableDiffusionErrorCode::ThreadBusy: return "ThreadBusy";
	case ofxStableDiffusionErrorCode::UpscaleFailed: return "UpscaleFailed";
	case ofxStableDiffusionErrorCode::Unknown: return "Unknown";
	case ofxStableDiffusionErrorCode::None:
	default:
		return "None";
	}
}

std::string ofxStableDiffusionErrorCodeSuggestion(ofxStableDiffusionErrorCode code) {
	switch (code) {
	case ofxStableDiffusionErrorCode::ModelNotFound:
		return "Verify the model file path exists and is accessible";
	case ofxStableDiffusionErrorCode::ModelCorrupted:
		return "Re-download the model file or try a different model";
	case ofxStableDiffusionErrorCode::ModelLoadFailed:
		return "Check model format compatibility and ensure sufficient RAM/VRAM";
	case ofxStableDiffusionErrorCode::OutOfMemory:
		return "Reduce batch count, image dimensions, or enable VAE tiling";
	case ofxStableDiffusionErrorCode::InvalidDimensions:
		return "Use positive multiples of 64 (e.g., 512, 768, 1024) for width and height";
	case ofxStableDiffusionErrorCode::InvalidBatchCount:
		return "Set batch count between 1 and 16";
	case ofxStableDiffusionErrorCode::InvalidFrameCount:
		return "Set frame count between 1 and 100";
	case ofxStableDiffusionErrorCode::InvalidParameter:
		return "Verify numeric parameters are within supported ranges";
	case ofxStableDiffusionErrorCode::MissingInputImage:
		return "Load an input image using loadImage() before calling this operation";
	case ofxStableDiffusionErrorCode::GenerationFailed:
		return "Check model compatibility and system resources";
	case ofxStableDiffusionErrorCode::ThreadBusy:
		return "Wait for current generation to complete or use isGenerating() to check status";
	case ofxStableDiffusionErrorCode::UpscaleFailed:
		return "Verify upscaler model path and compatibility";
	case ofxStableDiffusionErrorCode::Unknown:
		return "Check logs for more details";
	case ofxStableDiffusionErrorCode::None:
	default:
		return "";
	}
}

std::vector<ofxStableDiffusionImageFrame> ofxStableDiffusionBuildVideoFrames(
	const std::vector<ofxStableDiffusionImageFrame> & sourceFrames,
	ofxStableDiffusionVideoMode mode) {
	std::vector<ofxStableDiffusionImageFrame> frames;
	if (sourceFrames.empty()) {
		return frames;
	}

	const std::vector<int> sequence =
		ofxStableDiffusionBuildVideoFrameSequence(static_cast<int>(sourceFrames.size()), mode);
	frames.reserve(sequence.size());
	for (std::size_t i = 0; i < sequence.size(); ++i) {
		const int sourceIndex = sequence[i];
		appendFrameCopy(frames, sourceFrames[static_cast<std::size_t>(sourceIndex)], static_cast<int>(i));
	}

	return frames;
}

int64_t ofxStableDiffusionHashStringToSeed(const std::string& text) {
	if (text.empty()) {
		return -1;
	}
	std::hash<std::string> hasher;
	size_t hash = hasher(text);
	// Convert to int64_t, ensuring we stay in valid seed range
	return static_cast<int64_t>(hash & 0x7FFFFFFFFFFFFFFF);
}

