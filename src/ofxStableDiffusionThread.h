#pragma once

#include "ofMain.h"
#include "core/ofxStableDiffusionTypes.h"
#include "core/ofxStableDiffusionNativeApi.h"

#include <atomic>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

class stableDiffusionThread : public ofThread {
public:
	struct OwnedImage {
		sd_image_t image{0, 0, 0, nullptr};
		std::vector<uint8_t> storage;

		OwnedImage() = default;

		OwnedImage(const OwnedImage& other)
			: image(other.image)
			, storage(other.storage) {
			rebindImageData();
		}

		OwnedImage& operator=(const OwnedImage& other) {
			if (this != &other) {
				image = other.image;
				storage = other.storage;
				rebindImageData();
			}
			return *this;
		}

		OwnedImage(OwnedImage&& other) noexcept
			: image(other.image)
			, storage(std::move(other.storage)) {
			rebindImageData();
			other.image = {0, 0, 0, nullptr};
		}

		OwnedImage& operator=(OwnedImage&& other) noexcept {
			if (this != &other) {
				image = other.image;
				storage = std::move(other.storage);
				rebindImageData();
				other.image = {0, 0, 0, nullptr};
			}
			return *this;
		}

		void clear() {
			storage.clear();
			image = {0, 0, 0, nullptr};
		}

		bool assign(const sd_image_t& source) {
			if (!source.data || source.width == 0 || source.height == 0 || source.channel == 0) {
				clear();
				return false;
			}

			// Check for potential overflow in byte count calculation
			const std::size_t width = static_cast<std::size_t>(source.width);
			const std::size_t height = static_cast<std::size_t>(source.height);
			const std::size_t channels = static_cast<std::size_t>(source.channel);

			// Check for overflow: if width * height would overflow, fail
			if (width > 0 && height > SIZE_MAX / width) {
				ofLogError("OwnedImage") << "Image dimensions too large: " << width << "x" << height;
				clear();
				return false;
			}
			const std::size_t pixelCount = width * height;

			// Check for overflow: if pixelCount * channels would overflow, fail
			if (pixelCount > 0 && channels > SIZE_MAX / pixelCount) {
				ofLogError("OwnedImage") << "Image byte count too large: " << pixelCount << " pixels x " << channels << " channels";
				clear();
				return false;
			}
			const std::size_t byteCount = pixelCount * channels;

			storage.resize(byteCount);
			std::memcpy(storage.data(), source.data, byteCount);
			image = {source.width, source.height, source.channel, storage.data()};
			return true;
		}

		bool isAllocated() const {
			return image.data != nullptr;
		}

	private:
		void rebindImageData() {
			if (storage.empty() || image.width == 0 || image.height == 0 || image.channel == 0) {
				image = {0, 0, 0, nullptr};
				return;
			}
			image.data = storage.data();
		}
	};

	struct ContextTaskData {
		ofxStableDiffusionContextSettings contextSettings;
		ofxStableDiffusionUpscalerSettings upscalerSettings;
	};

	struct ImageTaskData {
		ofxStableDiffusionTask task = ofxStableDiffusionTask::TextToImage;
		ofxStableDiffusionContextSettings contextSettings;
		ofxStableDiffusionUpscalerSettings upscalerSettings;
		ofxStableDiffusionImageRequest request;
		OwnedImage initImage;
		OwnedImage maskImage;
		OwnedImage controlImage;
		std::function<void(int step, int steps, float time)> progressCallback;
		std::function<std::vector<ofxStableDiffusionImageScore>(
			const ofxStableDiffusionImageRequest& request,
			const std::vector<ofxStableDiffusionImageFrame>& images)> imageRankCallback;

		void syncViews() {
			request.initImage = initImage.image;
			request.maskImage = maskImage.image;
			request.controlCond = controlImage.isAllocated() ? &controlImage.image : nullptr;
		}
	};

	struct VideoTaskData {
		ofxStableDiffusionTask task = ofxStableDiffusionTask::ImageToVideo;
		ofxStableDiffusionContextSettings contextSettings;
		ofxStableDiffusionUpscalerSettings upscalerSettings;
		ofxStableDiffusionVideoRequest request;
		OwnedImage initImage;
		OwnedImage endImage;
		std::vector<OwnedImage> controlFrames;
		std::vector<sd_image_t> controlFrameViews;
		std::function<void(int step, int steps, float time)> progressCallback;
		bool animationProgressEnabled = false;
		int animationFrameIndex = 0;
		int animationFrameCount = 0;
		int animationSampleSteps = 0;

		void syncViews() {
			request.initImage = initImage.image;
			request.endImage = endImage.image;
			controlFrameViews.clear();
			controlFrameViews.reserve(controlFrames.size());
			for (auto& frame : controlFrames) {
				if (frame.isAllocated()) {
					controlFrameViews.push_back(frame.image);
				}
			}
			request.controlFrames = controlFrameViews;
		}
	};

	void* userData = nullptr;
	upscaler_ctx_t* upscalerCtx = nullptr;
	sd_ctx_t* sdCtx = nullptr;
	ofxStableDiffusionTask task = ofxStableDiffusionTask::None;
	ContextTaskData contextTaskData;
	ImageTaskData imageTaskData;
	VideoTaskData videoTaskData;
	~stableDiffusionThread() override;
	void clearContexts();
	void prepareContextTask(const ContextTaskData& data);
	void prepareImageTask(const ImageTaskData& data);
	void prepareVideoTask(const VideoTaskData& data);

	/// Request cancellation of current operation
	void requestCancellation();

	/// Check if cancellation was requested
	bool isCancellationRequested() const;

	/// Reset cancellation flag
	void resetCancellation();

private:
	void threadedFunction();
	std::string computeContextFingerprint(const ofxStableDiffusionContextSettings& settings);
	bool isSdCtxLoaded = false;
	bool isUpscalerCtxLoaded = false;
	bool generationContextNeedsRefresh = false;
	std::vector<sd_lora_t> loraBuffer;
	std::atomic<bool> cancellationRequested{false};

	// Context reuse optimization
	std::string lastContextFingerprint;
	int generationsSinceRebuild = 0;
	static constexpr int MAX_REUSE_COUNT = 10; // Safety valve: rebuild after N generations
};
