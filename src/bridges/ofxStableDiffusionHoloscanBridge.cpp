#include "bridges/ofxStableDiffusionHoloscanBridge.h"

#include "ofxStableDiffusion.h"

#include <deque>
#include <mutex>

namespace {

bool copyPixelsToPreview(
	const ofxStableDiffusionImageFrame& image,
	ofTexture* texture,
	ofxStableDiffusionHoloscanPreviewFrame* preview) {
	if (!texture || !preview || !image.isAllocated()) {
		return false;
	}
	preview->valid = true;
	preview->pixels = image.pixels;
	texture->loadData(preview->pixels);
	return true;
}

ofxStableDiffusionImageRequest makeBridgeRequest(
	const ofxStableDiffusionHoloscanConditioningPacket& packet) {
	ofxStableDiffusionImageRequest request;
	request.mode = ofxStableDiffusionImageMode::ImageToImage;
	request.prompt = packet.prompt;
	request.negativePrompt = packet.negativePrompt;
	request.strength = packet.strength;
	request.width = static_cast<int>(packet.initImage->getWidth());
	request.height = static_cast<int>(packet.initImage->getHeight());
	request.initImage = {
		static_cast<uint32_t>(packet.initImage->getWidth()),
		static_cast<uint32_t>(packet.initImage->getHeight()),
		static_cast<uint32_t>(packet.initImage->getNumChannels()),
		packet.initImage->getData()
	};
	return request;
}

} // namespace

struct ofxStableDiffusionHoloscanBridge::Impl {
	ofxStableDiffusion* diffusion = nullptr;
	ofxStableDiffusionHoloscanSettings settings;
	bool configured = false;
	bool running = false;
	bool holoscanAvailable = false;
	std::string lastError;
	std::string latestPrompt;
	std::string latestNegativePrompt;
	uint64_t nextFrameIndex = 1;
	ofTexture previewTexture;
	ofxStableDiffusionHoloscanPreviewFrame previewFrame;
	std::deque<ofxStableDiffusionHoloscanFramePacket> pendingFrames;
	std::vector<ofxStableDiffusionImageFrame> finishedImages;
	std::mutex mutex;
	bool requestInFlight = false;
	uint64_t inFlightFrameIndex = 0;
	double inFlightTimestampSeconds = 0.0;

	void clearQueues() {
		std::lock_guard<std::mutex> lock(mutex);
		pendingFrames.clear();
		finishedImages.clear();
		previewFrame = {};
		previewTexture.clear();
		requestInFlight = false;
		inFlightFrameIndex = 0;
		inFlightTimestampSeconds = 0.0;
	}
};

ofxStableDiffusionHoloscanBridge::ofxStableDiffusionHoloscanBridge()
	: impl_(std::make_unique<Impl>()) {
#if defined(__has_include)
#  if __has_include(<holoscan/holoscan.hpp>)
	impl_->holoscanAvailable = true;
#  endif
#endif
}

ofxStableDiffusionHoloscanBridge::~ofxStableDiffusionHoloscanBridge() = default;

bool ofxStableDiffusionHoloscanBridge::setup(
	ofxStableDiffusion* diffusion,
	const ofxStableDiffusionHoloscanSettings& settings) {
	impl_->diffusion = diffusion;
	impl_->settings = settings;
	impl_->configured = (diffusion != nullptr);
	impl_->lastError.clear();
	if (!impl_->configured) {
		impl_->lastError = "Holoscan bridge setup requires a valid ofxStableDiffusion instance.";
	}
	return impl_->configured;
}

void ofxStableDiffusionHoloscanBridge::shutdown() {
	stop();
	impl_->configured = false;
	impl_->diffusion = nullptr;
	impl_->lastError.clear();
}

bool ofxStableDiffusionHoloscanBridge::startImagePipeline() {
	if (!impl_->configured || impl_->diffusion == nullptr) {
		impl_->lastError = "Holoscan bridge is not configured.";
		return false;
	}
	impl_->running = true;
	impl_->lastError.clear();
	return true;
}

void ofxStableDiffusionHoloscanBridge::stop() {
	impl_->running = false;
	impl_->clearQueues();
}

void ofxStableDiffusionHoloscanBridge::update() {
	if (!impl_->running || !impl_->diffusion) {
		return;
	}

	if (impl_->requestInFlight) {
		if (!impl_->diffusion->isGenerating() && impl_->diffusion->isDiffused()) {
			const auto result = impl_->diffusion->getLastResult();
			if (result.success && !result.images.empty()) {
				auto imageResult = result.images.front();
				copyPixelsToPreview(imageResult, &impl_->previewTexture, &impl_->previewFrame);
				impl_->previewFrame.frameIndex = impl_->inFlightFrameIndex;
				impl_->previewFrame.timestampSeconds = impl_->inFlightTimestampSeconds;
				std::lock_guard<std::mutex> lock(impl_->mutex);
				impl_->finishedImages.push_back(std::move(imageResult));
			} else if (!result.success) {
				impl_->lastError = result.error;
			}
			impl_->requestInFlight = false;
			impl_->inFlightFrameIndex = 0;
			impl_->inFlightTimestampSeconds = 0.0;
		}
		return;
	}

	if (impl_->diffusion->isGenerating()) {
		return;
	}

	ofxStableDiffusionHoloscanFramePacket frame;
	{
		std::lock_guard<std::mutex> lock(impl_->mutex);
		if (impl_->pendingFrames.empty()) {
			return;
		}
		frame = impl_->pendingFrames.front();
		impl_->pendingFrames.pop_front();
	}

	if (!frame.isValid()) {
		return;
	}
	if (impl_->latestPrompt.empty()) {
		impl_->lastError = "Holoscan bridge needs a prompt before it can submit a frame.";
		return;
	}

	ofxStableDiffusionHoloscanConditioningPacket conditioning;
	conditioning.frameIndex = frame.frameIndex;
	conditioning.timestampSeconds = frame.timestampSeconds;
	conditioning.prompt = impl_->latestPrompt;
	conditioning.negativePrompt = impl_->latestNegativePrompt;
	conditioning.initImage = frame.pixels;

	auto request = makeBridgeRequest(conditioning);
	impl_->diffusion->generate(request);
	impl_->requestInFlight = true;
	impl_->inFlightFrameIndex = conditioning.frameIndex;
	impl_->inFlightTimestampSeconds = conditioning.timestampSeconds;
}

void ofxStableDiffusionHoloscanBridge::submitFrame(
	const ofPixels& pixels,
	double timestampSeconds,
	const std::string& sourceLabel) {
	if (!pixels.isAllocated()) {
		impl_->lastError = "Holoscan bridge received an empty frame.";
		return;
	}
	auto ownedPixels = std::make_shared<ofPixels>(pixels);
	ofxStableDiffusionHoloscanFramePacket packet;
	packet.frameIndex = impl_->nextFrameIndex++;
	packet.timestampSeconds = timestampSeconds;
	packet.pixels = ownedPixels;
	packet.sourceLabel = sourceLabel;
	std::lock_guard<std::mutex> lock(impl_->mutex);
	impl_->pendingFrames.push_back(std::move(packet));
}

void ofxStableDiffusionHoloscanBridge::submitPrompt(
	const std::string& prompt,
	const std::string& negativePrompt) {
	impl_->latestPrompt = prompt;
	impl_->latestNegativePrompt = negativePrompt;
}

bool ofxStableDiffusionHoloscanBridge::hasPreviewFrame() const {
	return impl_->previewFrame.valid && impl_->previewTexture.isAllocated();
}

const ofTexture& ofxStableDiffusionHoloscanBridge::getPreviewTexture() const {
	return impl_->previewTexture;
}

ofxStableDiffusionHoloscanPreviewFrame
ofxStableDiffusionHoloscanBridge::getPreviewFrameCopy() const {
	return impl_->previewFrame;
}

std::vector<ofxStableDiffusionImageFrame>
ofxStableDiffusionHoloscanBridge::consumeFinishedImages() {
	std::lock_guard<std::mutex> lock(impl_->mutex);
	auto images = std::move(impl_->finishedImages);
	impl_->finishedImages.clear();
	return images;
}

bool ofxStableDiffusionHoloscanBridge::isConfigured() const {
	return impl_->configured;
}

bool ofxStableDiffusionHoloscanBridge::isRunning() const {
	return impl_->running;
}

bool ofxStableDiffusionHoloscanBridge::isHoloscanAvailable() const {
	return impl_->holoscanAvailable;
}

std::string ofxStableDiffusionHoloscanBridge::getLastError() const {
	return impl_->lastError;
}

const ofxStableDiffusionHoloscanSettings&
ofxStableDiffusionHoloscanBridge::getSettings() const {
	return impl_->settings;
}
