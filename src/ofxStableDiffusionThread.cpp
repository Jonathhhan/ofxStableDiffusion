#include "ofxStableDiffusionThread.h"
#include "ofxStableDiffusion.h"
#include "core/ofxStableDiffusionNativeAdapter.h"
#include "core/ofxStableDiffusionMemoryHelpers.h"
#include "video/ofxStableDiffusionVideoHelpers.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <mutex>
#include <ctime>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::mutex& generationCallbackMutex() {
	static std::mutex mutex;
	return mutex;
}

class ProgressCallbackGuard {
public:
	ProgressCallbackGuard(std::mutex& mutex, sd_progress_cb_t cb, void* data)
		: lock(mutex) {
		sd_set_progress_callback(cb, data);
	}

	~ProgressCallbackGuard() {
		sd_set_progress_callback(nullptr, nullptr);
	}

	ProgressCallbackGuard(const ProgressCallbackGuard&) = delete;
	ProgressCallbackGuard& operator=(const ProgressCallbackGuard&) = delete;

private:
	std::unique_lock<std::mutex> lock;
};

void threadProgressCallback(int step, int steps, float time, void* data) {
	auto* thread = static_cast<stableDiffusionThread*>(data);
	if (thread && thread->task == ofxStableDiffusionTask::ImageToVideo) {
		if (thread->videoTaskData.progressCallback) {
			try {
				if (thread->videoTaskData.animationProgressEnabled) {
					const int stepsPerFrame = std::max(1, steps);
					const int totalSteps = std::max(1, thread->videoTaskData.animationFrameCount) * stepsPerFrame;
					const int compositeStep =
						(thread->videoTaskData.animationFrameIndex * stepsPerFrame) +
						std::max(0, std::min(step, stepsPerFrame));
					thread->videoTaskData.progressCallback(compositeStep, totalSteps, time);
				} else {
					thread->videoTaskData.progressCallback(step, steps, time);
				}
			} catch (const std::exception& e) {
				ofLogWarning("ofxStableDiffusion") << "Progress callback threw: " << e.what();
			} catch (...) {
				ofLogWarning("ofxStableDiffusion") << "Progress callback threw an unknown exception";
			}
		}
		return;
	}

	if (thread && thread->imageTaskData.progressCallback) {
		try {
			thread->imageTaskData.progressCallback(step, steps, time);
		} catch (const std::exception& e) {
			ofLogWarning("ofxStableDiffusion") << "Progress callback threw: " << e.what();
		} catch (...) {
			ofLogWarning("ofxStableDiffusion") << "Progress callback threw an unknown exception";
		}
	}
}

bool areBlendable(const sd_image_t& left, const sd_image_t& right) {
	return left.data != nullptr &&
		right.data != nullptr &&
		left.width == right.width &&
		left.height == right.height &&
		left.channel == right.channel &&
		left.channel > 0;
}

void assignBlendedImage(
	const sd_image_t& startImage,
	const sd_image_t& endImage,
	float t,
	stableDiffusionThread::OwnedImage& output) {
	output.clear();
	if (!areBlendable(startImage, endImage)) {
		return;
	}

	const std::size_t byteCount =
		static_cast<std::size_t>(startImage.width) *
		static_cast<std::size_t>(startImage.height) *
		static_cast<std::size_t>(startImage.channel);
	output.storage.resize(byteCount);
	for (std::size_t i = 0; i < byteCount; ++i) {
		const float startValue = static_cast<float>(startImage.data[i]);
		const float endValue = static_cast<float>(endImage.data[i]);
		output.storage[i] = static_cast<uint8_t>(std::round(startValue + ((endValue - startValue) * t)));
	}
	output.image = {startImage.width, startImage.height, startImage.channel, output.storage.data()};
}

const sd_image_t& resolveAnimatedBaseImage(
	const stableDiffusionThread::VideoTaskData& taskData,
	int frameIndex,
	stableDiffusionThread::OwnedImage& blendedImage) {
	if (!taskData.endImage.isAllocated() || taskData.request.frameCount <= 1) {
		return taskData.initImage.image;
	}

	const float t = static_cast<float>(frameIndex) /
		static_cast<float>(std::max(1, taskData.request.frameCount - 1));
	if (areBlendable(taskData.initImage.image, taskData.endImage.image)) {
		assignBlendedImage(taskData.initImage.image, taskData.endImage.image, t, blendedImage);
		if (blendedImage.isAllocated()) {
			return blendedImage.image;
		}
	}

	return frameIndex >= (taskData.request.frameCount - 1) ?
		taskData.endImage.image :
		taskData.initImage.image;
}

std::vector<int64_t> buildAnimatedVideoSeeds(const stableDiffusionThread::VideoTaskData& taskData) {
	std::vector<int64_t> seeds;
	seeds.reserve(std::max(0, taskData.request.frameCount));
	for (int frameIndex = 0; frameIndex < taskData.request.frameCount; ++frameIndex) {
		seeds.push_back(ofxStableDiffusionGetFrameSeed(taskData.request, frameIndex));
	}
	return seeds;
}

sd_image_t* makeOwnedImageArray(const std::vector<stableDiffusionThread::OwnedImage>& frames) {
	if (frames.empty()) {
		return nullptr;
	}

	auto* output = static_cast<sd_image_t*>(std::malloc(sizeof(sd_image_t) * frames.size()));
	if (!output) {
		return nullptr;
	}

	for (std::size_t i = 0; i < frames.size(); ++i) {
		output[i] = {0, 0, 0, nullptr};
		const auto& frame = frames[i];
		if (!frame.isAllocated()) {
			continue;
		}

		const std::size_t byteCount =
			static_cast<std::size_t>(frame.image.width) *
			static_cast<std::size_t>(frame.image.height) *
			static_cast<std::size_t>(frame.image.channel);
		auto* pixels = static_cast<uint8_t*>(std::malloc(byteCount));
		if (!pixels) {
			ofxSdReleaseImageArray(output, static_cast<int>(i));
			return nullptr;
		}
		std::memcpy(pixels, frame.image.data, byteCount);
		output[i] = {frame.image.width, frame.image.height, frame.image.channel, pixels};
	}

	return output;
}

int64_t resolveNativeGenerationSeed(int64_t seed) {
	if (seed >= 0) {
		return seed;
	}

	// Match the backend's one-time auto-seed resolution so the addon can
	// report and reproduce the exact seed handed to native generation.
	std::srand(static_cast<unsigned int>(std::time(nullptr)));
	return static_cast<int64_t>(std::rand());
}

} // namespace

stableDiffusionThread::~stableDiffusionThread() {
	if (isThreadRunning()) {
		waitForThread(true);
	}
	{
		std::lock_guard<std::mutex> callbackLock(generationCallbackMutex());
		sd_set_progress_callback(nullptr, nullptr);
	}
	clearContexts();
}

void stableDiffusionThread::clearContexts() {
	if (sdCtx) {
		free_sd_ctx(sdCtx);
		sdCtx = nullptr;
	}
	if (upscalerCtx) {
		free_upscaler_ctx(upscalerCtx);
		upscalerCtx = nullptr;
	}
	isSdCtxLoaded = false;
	isUpscalerCtxLoaded = false;
	generationContextNeedsRefresh = false;
}

void stableDiffusionThread::prepareContextTask(const ContextTaskData& data) {
	task = ofxStableDiffusionTask::LoadModel;
	contextTaskData = data;
}

void stableDiffusionThread::prepareImageTask(const ImageTaskData& data) {
	task = data.task;
	imageTaskData = data;
	imageTaskData.syncViews();
}

void stableDiffusionThread::prepareVideoTask(const VideoTaskData& data) {
	task = data.task;
	videoTaskData = data;
	videoTaskData.syncViews();
}

void stableDiffusionThread::threadedFunction() {
	ofxStableDiffusion* sd = static_cast<ofxStableDiffusion*>(userData);
	if (!sd) {
		return;
	}

	if (task == ofxStableDiffusionTask::LoadModel || sd->isModelLoading) {
		if (sdCtx) {
			free_sd_ctx(sdCtx);
			sdCtx = nullptr;
		}

		std::vector<std::string> embeddingNames;
		std::vector<std::string> embeddingPaths;
		std::vector<sd_embedding_t> embeddings;
		const auto& contextSettings = contextTaskData.contextSettings;
		const auto describeMissingPaths = [&contextSettings]() {
			std::vector<std::string> missing;
			const auto addIfMissing = [&missing](const std::string& path, const char* label) {
				if (path.empty()) {
					return;
				}
				const fs::path p(path);
				if (!fs::exists(p)) {
					missing.push_back(std::string(label) + ": " + p.string());
				}
			};
			addIfMissing(contextSettings.modelPath, "model");
			addIfMissing(contextSettings.diffusionModelPath, "diffusion");
			addIfMissing(contextSettings.clipLPath, "clip_l");
			addIfMissing(contextSettings.clipGPath, "clip_g");
			addIfMissing(contextSettings.t5xxlPath, "text_encoder");
			addIfMissing(contextSettings.vaePath, "vae");
			addIfMissing(contextSettings.controlNetPath, "controlnet");
			return missing;
		};
		bool contextErrorReported = false;
		sd_ctx_params_t ctxParams =
			ofxStableDiffusionNativeAdapter::buildContextParams(
				contextTaskData,
				embeddingNames,
				embeddingPaths,
				embeddings);
		sdCtx = new_sd_ctx(&ctxParams);
		if (!sdCtx) {
			const auto missing = describeMissingPaths();
			if (!missing.empty()) {
				std::string message = "Missing model files: ";
				for (std::size_t i = 0; i < missing.size(); ++i) {
					message += missing[i];
					if (i + 1 < missing.size()) {
						message += "; ";
					}
				}
				sd->setLastError(ofxStableDiffusionErrorCode::ModelNotFound, message);
				contextErrorReported = true;
			} else {
				const std::string primary =
					!contextTaskData.contextSettings.modelPath.empty() ?
						contextTaskData.contextSettings.modelPath :
						contextTaskData.contextSettings.diffusionModelPath;
				sd->setLastError(
					ofxStableDiffusionErrorCode::ModelLoadFailed,
					primary.empty() ?
						"Failed to create stable-diffusion context" :
						"Failed to create stable-diffusion context for " + primary);
				contextErrorReported = true;
			}
		}

		if (upscalerCtx) {
			free_upscaler_ctx(upscalerCtx);
			upscalerCtx = nullptr;
			isUpscalerCtxLoaded = false;
		}
		if (contextTaskData.upscalerSettings.enabled && !contextTaskData.upscalerSettings.modelPath.empty()) {
			upscalerCtx = new_upscaler_ctx(
				contextTaskData.upscalerSettings.modelPath.c_str(),
				false,
				false,
				contextTaskData.upscalerSettings.nThreads,
				0);
			isUpscalerCtxLoaded = (upscalerCtx != nullptr);
		}
		isSdCtxLoaded = (sdCtx != nullptr);
		generationContextNeedsRefresh = false;
		{
			std::lock_guard<std::mutex> lock(sd->stateMutex);
			sd->refreshResolvedDefaultCachesNoLock(sdCtx);
		}
		if (contextTaskData.upscalerSettings.enabled &&
			!contextTaskData.upscalerSettings.modelPath.empty() &&
			!isUpscalerCtxLoaded) {
			{
				std::lock_guard<std::mutex> lock(sd->stateMutex);
				sd->isESRGAN = false;
				sd->esrganPath.clear();
			}
			contextTaskData.upscalerSettings.enabled = false;
			contextTaskData.upscalerSettings.modelPath.clear();
			sd->setLastError(ofxStableDiffusionErrorCode::UpscaleFailed, "Failed to create upscaler context");
		}
		sd->isModelLoading = false;
		sd->activeTask = ofxStableDiffusionTask::None;
		task = ofxStableDiffusionTask::None;
		if (!isSdCtxLoaded && !contextErrorReported) {
			sd->setLastError("Failed to create stable-diffusion context");
		}
		return;
	}

	const auto rebuildSdContextForGeneration =
		[this, &sd](const ofxStableDiffusionContextSettings& currentContextSettings) -> bool {
			if (sdCtx) {
				free_sd_ctx(sdCtx);
				sdCtx = nullptr;
			}

			stableDiffusionThread::ContextTaskData reloadTask;
			reloadTask.contextSettings = currentContextSettings;
			std::vector<std::string> embeddingNames;
			std::vector<std::string> embeddingPaths;
			std::vector<sd_embedding_t> embeddings;
			sd_ctx_params_t ctxParams =
				ofxStableDiffusionNativeAdapter::buildContextParams(
					reloadTask,
					embeddingNames,
					embeddingPaths,
					embeddings);
			sdCtx = new_sd_ctx(&ctxParams);
			isSdCtxLoaded = (sdCtx != nullptr);
			{
				std::lock_guard<std::mutex> lock(sd->stateMutex);
				sd->refreshResolvedDefaultCachesNoLock(sdCtx);
			}
			if (!isSdCtxLoaded) {
				sd->setLastError("Failed to recreate stable-diffusion context for generation");
				sd->activeTask = ofxStableDiffusionTask::None;
				task = ofxStableDiffusionTask::None;
				return false;
			}
			return true;
		};

	if (!sdCtx) {
		sd->activeTask = ofxStableDiffusionTask::None;
		task = ofxStableDiffusionTask::None;
		sd->setLastError("Stable Diffusion context is not loaded");
		return;
	}

	const bool isVideoTask = (task == ofxStableDiffusionTask::ImageToVideo || sd->isImageToVideo);
	const ofxStableDiffusionContextSettings& generationContextSettings =
		isVideoTask ? videoTaskData.contextSettings : imageTaskData.contextSettings;
	if (generationContextNeedsRefresh) {
		if (!rebuildSdContextForGeneration(generationContextSettings)) {
			return;
		}
	}

	// After one generation attempt, rebuild the context before the next run.
	generationContextNeedsRefresh = true;

	const bool upscalerAvailable = isUpscalerCtxLoaded && upscalerCtx;

	if (task == ofxStableDiffusionTask::ImageToVideo || sd->isImageToVideo) {
		if (videoTaskData.upscalerSettings.enabled && !upscalerAvailable) {
			videoTaskData.upscalerSettings.enabled = false;
			sd->setLastError(
				ofxStableDiffusionErrorCode::UpscaleFailed,
				"Upscaler context is not available for video generation");
			sd->activeTask = ofxStableDiffusionTask::None;
			task = ofxStableDiffusionTask::None;
			return;
		}

		if (videoTaskData.request.hasAnimation()) {
			const int frameCount = std::max(0, videoTaskData.request.frameCount);
			std::vector<OwnedImage> generatedFrames;
			generatedFrames.reserve(static_cast<std::size_t>(frameCount));
			std::vector<int64_t> frameSeeds = buildAnimatedVideoSeeds(videoTaskData);
			std::vector<ofxStableDiffusionGenerationParameters> frameGeneration;
			frameGeneration.reserve(static_cast<std::size_t>(frameCount));
			std::string generationError;

			videoTaskData.animationProgressEnabled = (videoTaskData.progressCallback != nullptr);
			videoTaskData.animationFrameIndex = 0;
			videoTaskData.animationFrameCount = frameCount;
			videoTaskData.animationSampleSteps = videoTaskData.request.sampleSteps;

			for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
					videoTaskData.animationFrameIndex = frameIndex;

					OwnedImage blendedInitImage;
					const sd_image_t& frameInitImage =
						resolveAnimatedBaseImage(videoTaskData, frameIndex, blendedInitImage);
					const bool hasFrameInitImage = (frameInitImage.data != nullptr);

					ImageTaskData frameTask;
					frameTask.task =
						hasFrameInitImage ?
							ofxStableDiffusionTask::ImageToImage :
							ofxStableDiffusionTask::TextToImage;
					frameTask.contextSettings = videoTaskData.contextSettings;
					frameTask.upscalerSettings = videoTaskData.upscalerSettings;
					frameTask.request.mode =
						hasFrameInitImage ?
							ofxStableDiffusionImageMode::ImageToImage :
							ofxStableDiffusionImageMode::TextToImage;
					frameTask.request.initImage =
						hasFrameInitImage ? frameInitImage : sd_image_t{0, 0, 0, nullptr};
					frameTask.request.prompt =
						ofxStableDiffusionGetFramePrompt(videoTaskData.request, frameIndex);
					frameTask.request.negativePrompt =
						ofxStableDiffusionGetFrameNegativePrompt(videoTaskData.request, frameIndex);
					frameTask.request.clipSkip = videoTaskData.request.clipSkip;
					frameTask.request.cfgScale =
						ofxStableDiffusionGetFrameCfgScale(videoTaskData.request, frameIndex);
					frameTask.request.width = videoTaskData.request.width;
					frameTask.request.height = videoTaskData.request.height;
					frameTask.request.sampleMethod = videoTaskData.request.sampleMethod;
					frameTask.request.sampleSteps = videoTaskData.request.sampleSteps;
					frameTask.request.strength =
						ofxStableDiffusionGetFrameStrength(videoTaskData.request, frameIndex);
					frameTask.request.seed =
						static_cast<std::size_t>(frameIndex) < frameSeeds.size() ?
							frameSeeds[static_cast<std::size_t>(frameIndex)] :
							videoTaskData.request.seed;
					frameTask.request.batchCount = 1;
					frameTask.request.loras = videoTaskData.request.loras;
					frameGeneration.push_back({
						frameTask.request.prompt,
						frameTask.request.negativePrompt,
						frameTask.request.cfgScale,
						frameTask.request.strength
					});
					if (hasFrameInitImage) {
						frameTask.initImage.assign(frameInitImage);
					}
					frameTask.syncViews();

					const std::string effectivePrompt =
						ofxStableDiffusionNativeAdapter::buildEffectivePrompt(frameTask.request);
					std::vector<ofPixels> pmPixels;
					std::vector<sd_image_t> pmImageViews;
					sd_img_gen_params_t frameParams = ofxStableDiffusionNativeAdapter::buildImageParams(
						frameTask,
						sdCtx,
						effectivePrompt,
						loraBuffer,
						pmPixels,
						pmImageViews);

					sd_image_t* frameOutput = nullptr;
					if (videoTaskData.progressCallback) {
						ProgressCallbackGuard progressGuard(
							generationCallbackMutex(),
							threadProgressCallback,
							this);
						frameOutput = generate_image(sdCtx, &frameParams);
					} else {
						frameOutput = generate_image(sdCtx, &frameParams);
					}
					if (!frameOutput || !frameOutput[0].data) {
						ofxSdReleaseImageArray(frameOutput, 1);
						generationError = "Animated video generation returned no frame output";
						break;
					}

					if (videoTaskData.upscalerSettings.enabled) {
						if (!upscalerCtx) {
							ofxSdReleaseImageArray(frameOutput, 1);
							sd->setLastError(
								ofxStableDiffusionErrorCode::UpscaleFailed,
								"Upscaler context is not loaded");
							generationError.clear();
							break;
						}

						sd_image_t upscaled =
							upscale(upscalerCtx, frameOutput[0], videoTaskData.upscalerSettings.multiplier);
						if (!upscaled.data) {
							ofxSdReleaseImageArray(frameOutput, 1);
							sd->setLastError(
								ofxStableDiffusionErrorCode::UpscaleFailed,
								"Upscaling failed for one or more video frames");
							generationError.clear();
							break;
						}

						ofxSdReleaseImage(frameOutput[0]);
						frameOutput[0] = upscaled;
					}

					OwnedImage generatedFrame;
					if (!generatedFrame.assign(frameOutput[0])) {
						ofxSdReleaseImageArray(frameOutput, 1);
						generationError = "Animated video generation produced an invalid frame";
						break;
					}

					ofxSdReleaseImageArray(frameOutput, 1);
					generatedFrames.push_back(std::move(generatedFrame));
				}

			videoTaskData.animationProgressEnabled = false;
			videoTaskData.animationFrameIndex = 0;
			videoTaskData.animationFrameCount = 0;
			videoTaskData.animationSampleSteps = 0;

			if (generatedFrames.size() != static_cast<std::size_t>(frameCount)) {
				if (!generationError.empty()) {
					sd->setLastError(generationError);
				}
				sd->activeTask = ofxStableDiffusionTask::None;
				task = ofxStableDiffusionTask::None;
				return;
			}

			sd_image_t* output = makeOwnedImageArray(generatedFrames);
			const float elapsedMs =
				static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
			if (!output) {
				sd->setLastError("Animated video generation could not allocate output frames");
				sd->activeTask = ofxStableDiffusionTask::None;
				task = ofxStableDiffusionTask::None;
				return;
			}

			const int64_t actualSeedUsed = frameSeeds.empty() ? videoTaskData.request.seed : frameSeeds.front();
			sd->captureVideoResults(
				output,
				static_cast<int>(generatedFrames.size()),
				actualSeedUsed,
				frameSeeds,
				frameGeneration,
				elapsedMs,
				task,
				videoTaskData.request);
			sd->activeTask = ofxStableDiffusionTask::None;
			task = ofxStableDiffusionTask::None;
			generationContextNeedsRefresh = true;
			return;
		}

		sd_vid_gen_params_t params =
			ofxStableDiffusionNativeAdapter::buildVideoParams(videoTaskData, sdCtx, loraBuffer);
		params.seed = resolveNativeGenerationSeed(params.seed);
		const std::string resolvedVideoSummary =
			ofxStableDiffusionNativeAdapter::describeVideoParams(params);
		const std::string resolvedVideoCliCommand =
			ofxStableDiffusionNativeAdapter::buildResolvedVideoCliCommand(
				params,
				videoTaskData.contextSettings);
		sd->setLastResolvedVideoRequestSummary(resolvedVideoSummary);
		sd->setLastResolvedVideoCliCommand(resolvedVideoCliCommand);
		ofLogNotice("ofxStableDiffusion")
			<< "Wrapper video request: "
			<< resolvedVideoSummary;
		int generatedFrameCount = 0;
		sd_image_t* output = nullptr;
		if (videoTaskData.progressCallback) {
			ProgressCallbackGuard progressGuard(
				generationCallbackMutex(),
				threadProgressCallback,
				this);
			output = generate_video(sdCtx, &params, &generatedFrameCount);
		} else {
			output = generate_video(sdCtx, &params, &generatedFrameCount);
		}
		const float elapsedMs = static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
		if (!output || generatedFrameCount <= 0) {
			ofxSdReleaseImageArray(output, generatedFrameCount);
			sd->setLastError("Image-to-video generation returned no frames");
			sd->activeTask = ofxStableDiffusionTask::None;
			task = ofxStableDiffusionTask::None;
			return;
		}
		sd->captureVideoResults(
			output,
			generatedFrameCount,
			params.seed,
			{},
			{},
			elapsedMs,
			task,
			videoTaskData.request);
		sd->activeTask = ofxStableDiffusionTask::None;
		task = ofxStableDiffusionTask::None;
		generationContextNeedsRefresh = true;
		return;
	}

	const std::string effectivePrompt =
		ofxStableDiffusionNativeAdapter::buildEffectivePrompt(imageTaskData.request);
	std::vector<ofPixels> pmPixels;
	std::vector<sd_image_t> pmImageViews;

	if (imageTaskData.upscalerSettings.enabled && !upscalerAvailable) {
		imageTaskData.upscalerSettings.enabled = false;
		sd->setLastError(
			ofxStableDiffusionErrorCode::UpscaleFailed,
			"Upscaler context is not available for image generation");
		sd->activeTask = ofxStableDiffusionTask::None;
		task = ofxStableDiffusionTask::None;
		return;
	}

	sd_img_gen_params_t params =
		ofxStableDiffusionNativeAdapter::buildImageParams(
			imageTaskData,
			sdCtx,
			effectivePrompt,
			loraBuffer,
			pmPixels,
			pmImageViews);
	params.seed = resolveNativeGenerationSeed(params.seed);
	sd_image_t* output = nullptr;
	if (imageTaskData.progressCallback) {
		ProgressCallbackGuard progressGuard(
			generationCallbackMutex(),
			threadProgressCallback,
			this);
		output = generate_image(sdCtx, &params);
	} else {
		output = generate_image(sdCtx, &params);
	}

	if (output && imageTaskData.upscalerSettings.enabled) {
		if (!upscalerCtx) {
			ofxSdReleaseImageArray(output, imageTaskData.request.batchCount);
			sd->setLastError(ofxStableDiffusionErrorCode::UpscaleFailed, "Upscaler context is not loaded");
			sd->activeTask = ofxStableDiffusionTask::None;
			task = ofxStableDiffusionTask::None;
			return;
		}

		for (int i = 0; i < imageTaskData.request.batchCount; i++) {
			sd_image_t upscaled = upscale(upscalerCtx, output[i], imageTaskData.upscalerSettings.multiplier);
			if (!upscaled.data) {
				ofxSdReleaseImage(output[i]);
				ofxSdReleaseImageArray(output, imageTaskData.request.batchCount);
				sd->setLastError(ofxStableDiffusionErrorCode::UpscaleFailed, "Upscaling failed for one or more images");
				sd->activeTask = ofxStableDiffusionTask::None;
				task = ofxStableDiffusionTask::None;
				return;
			}
			ofxSdReleaseImage(output[i]);
			output[i] = upscaled;
		}
	}

	const float elapsedMs = static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
	if (!output) {
		sd->setLastError("Image generation returned no images");
		sd->activeTask = ofxStableDiffusionTask::None;
		task = ofxStableDiffusionTask::None;
		return;
	}

	sd->captureImageResults(
		output,
		imageTaskData.request.batchCount,
		params.seed,
		elapsedMs,
		task,
		imageTaskData.request,
		imageTaskData.imageRankCallback);
	sd->activeTask = ofxStableDiffusionTask::None;
	task = ofxStableDiffusionTask::None;
	generationContextNeedsRefresh = true;
}
