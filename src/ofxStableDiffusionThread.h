#pragma once

#include "ofMain.h"
#include "core/ofxStableDiffusionTypes.h"
#include "stable-diffusion.h"

#include <cstring>
#include <functional>
#include <string>
#include <vector>

class stableDiffusionThread : public ofThread {
public:
	struct OwnedImage {
		sd_image_t image{0, 0, 0, nullptr};
		std::vector<uint8_t> storage;

		void clear() {
			storage.clear();
			image = {0, 0, 0, nullptr};
		}

		bool assign(const sd_image_t& source) {
			if (!source.data || source.width == 0 || source.height == 0 || source.channel == 0) {
				clear();
				return false;
			}

			const std::size_t byteCount =
				static_cast<std::size_t>(source.width) *
				static_cast<std::size_t>(source.height) *
				static_cast<std::size_t>(source.channel);
			storage.resize(byteCount);
			std::memcpy(storage.data(), source.data, byteCount);
			image = {source.width, source.height, source.channel, storage.data()};
			return true;
		}

		bool isAllocated() const {
			return image.data != nullptr;
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
private:
	void threadedFunction();
	bool isSdCtxLoaded = false;
	bool isUpscalerCtxLoaded = false;
	bool generationContextNeedsRefresh = false;
	std::vector<sd_lora_t> loraBuffer;
};
