#include "../core/ofxStableDiffusionTypes.h"
#include "../core/ofxStableDiffusionImageHelpers.h"
#include "ofxStableDiffusionVideoHelpers.h"

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

const char * ofxStableDiffusionTaskLabel(ofxStableDiffusionTask task) {
	switch (task) {
	case ofxStableDiffusionTask::LoadModel: return "LoadModel";
	case ofxStableDiffusionTask::TextToImage: return "TextToImage";
	case ofxStableDiffusionTask::ImageToImage: return "ImageToImage";
	case ofxStableDiffusionTask::InstructImage: return "InstructImage";
	case ofxStableDiffusionTask::ImageVariation: return "ImageVariation";
	case ofxStableDiffusionTask::ImageRestyle: return "ImageRestyle";
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
