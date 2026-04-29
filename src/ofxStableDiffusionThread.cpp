#include "ofxStableDiffusionThread.h"
#include "ofxStableDiffusion.h"
#include "core/ofxStableDiffusionNativeAdapter.h"
#include "core/ofxStableDiffusionMemoryHelpers.h"
#include "video/ofxStableDiffusionVideoHelpers.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <random>
#include <ctime>
#include <vector>
#include <sstream>

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

	static std::atomic<uint64_t> autoSeedCounter{0};
	uint64_t entropy = static_cast<uint64_t>(
		std::chrono::high_resolution_clock::now().time_since_epoch().count());
	entropy ^= autoSeedCounter.fetch_add(1, std::memory_order_relaxed) +
		0x9e3779b97f4a7c15ull;
	try {
		std::random_device rd;
		entropy ^= static_cast<uint64_t>(rd()) << 32;
		entropy ^= static_cast<uint64_t>(rd());
	} catch (...) {
	}

	std::mt19937_64 rng(entropy);
	std::uniform_int_distribution<int64_t> dist(
		0,
		(std::numeric_limits<int64_t>::max)());
	return dist(rng);
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
	isSdCtxLoaded.store(false, std::memory_order_release);
	if (sdCtx) {
		free_sd_ctx(sdCtx);
		sdCtx = nullptr;
	}
	if (upscalerCtx) {
		free_upscaler_ctx(upscalerCtx);
		upscalerCtx = nullptr;
	}
	isUpscalerCtxLoaded = false;
	generationContextNeedsRefresh = false;
	lastContextFingerprint.clear();
	generationsSinceRebuild = 0;
}

bool stableDiffusionThread::hasLoadedContext() const {
	return isSdCtxLoaded.load(std::memory_order_acquire);
}

std::string stableDiffusionThread::computeContextFingerprint(const ofxStableDiffusionContextSettings& settings) {
	// Create a fingerprint from settings that affect the native context
	// Only include settings that require context rebuild when changed
	std::ostringstream oss;
	oss << settings.modelPath << "|"
		<< settings.diffusionModelPath << "|"
		<< settings.clipLPath << "|"
		<< settings.clipGPath << "|"
		<< settings.t5xxlPath << "|"
		<< settings.vaePath << "|"
		<< settings.taesdPath << "|"
		<< settings.controlNetPath << "|"
		<< settings.loraModelDir << "|"
		<< settings.embedDir << "|"
		<< settings.stackedIdEmbedDir << "|"
		<< (int)settings.vaeDecodeOnly << "|"
		<< (int)settings.vaeTiling << "|"
		<< (int)settings.freeParamsImmediately << "|"
		<< settings.nThreads << "|"
		<< (int)settings.weightType << "|"
		<< (int)settings.rngType << "|"
		<< (int)settings.schedule << "|"
		<< (int)settings.prediction << "|"
		<< (int)settings.loraApplyMode << "|"
		<< (int)settings.keepClipOnCpu << "|"
		<< (int)settings.keepControlNetCpu << "|"
		<< (int)settings.keepVaeOnCpu << "|"
		<< (int)settings.offloadParamsToCpu << "|"
		<< (int)settings.flashAttn << "|"
		<< (int)settings.diffusionFlashAttn << "|"
		<< (int)settings.enableMmap;
	return oss.str();
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

	const auto finishTask =
		[this, &sd](bool cancelled = false, const std::string& cancelMessage = std::string()) {
			sd->finishBackgroundTask(cancelled, cancelMessage);
			task = ofxStableDiffusionTask::None;
		};

	const auto cancelRequested =
		[this, &finishTask](const std::string& message) {
			if (!isCancellationRequested()) {
				return false;
			}
			finishTask(true, message);
			return true;
		};

	if (task == ofxStableDiffusionTask::LoadModel || sd->isModelLoading) {
		if (cancelRequested("Model loading cancelled before the native context was created")) {
			return;
		}

		if (sdCtx) {
			isSdCtxLoaded.store(false, std::memory_order_release);
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
		if (isCancellationRequested()) {
			if (sdCtx) {
				free_sd_ctx(sdCtx);
				sdCtx = nullptr;
			}
			isSdCtxLoaded.store(false, std::memory_order_release);
			finishTask(true, "Model loading cancelled");
			return;
		}
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
		isSdCtxLoaded.store(sdCtx != nullptr, std::memory_order_release);
		generationContextNeedsRefresh = false;
		lastContextFingerprint = isSdCtxLoaded.load(std::memory_order_acquire) ?
			computeContextFingerprint(contextTaskData.contextSettings) :
			std::string();
		generationsSinceRebuild = 0;
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
		if (!isSdCtxLoaded.load(std::memory_order_acquire) && !contextErrorReported) {
			sd->setLastError("Failed to create stable-diffusion context");
		}
		finishTask();
		return;
	}

	const auto rebuildSdContextForGeneration =
		[this, &sd, &finishTask](const ofxStableDiffusionContextSettings& currentContextSettings) -> bool {
			if (sdCtx) {
				isSdCtxLoaded.store(false, std::memory_order_release);
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
			isSdCtxLoaded.store(sdCtx != nullptr, std::memory_order_release);
			{
				std::lock_guard<std::mutex> lock(sd->stateMutex);
				sd->refreshResolvedDefaultCachesNoLock(sdCtx);
			}
			if (!isSdCtxLoaded.load(std::memory_order_acquire)) {
				sd->setLastError("Failed to recreate stable-diffusion context for generation");
				finishTask();
				return false;
			}
			return true;
		};

	if (cancelRequested("Generation cancelled before the request started")) {
		return;
	}

	if (!sdCtx) {
		sd->setLastError("Stable Diffusion context is not loaded");
		finishTask();
		return;
	}

	const bool isVideoTask = (task == ofxStableDiffusionTask::ImageToVideo);
	const ofxStableDiffusionContextSettings& generationContextSettings =
		isVideoTask ? videoTaskData.contextSettings : imageTaskData.contextSettings;

	// Smart context reuse: only rebuild when necessary
	std::string currentFingerprint = computeContextFingerprint(generationContextSettings);
	bool needsRebuild = generationContextNeedsRefresh
		|| (currentFingerprint != lastContextFingerprint)
		|| (generationContextSettings.freeParamsImmediately &&
			generationsSinceRebuild >= MAX_REUSE_COUNT)
		|| !sdCtx;

	if (needsRebuild) {
		if (!rebuildSdContextForGeneration(generationContextSettings)) {
			return;
		}
		lastContextFingerprint = currentFingerprint;
		generationsSinceRebuild = 0;
		generationContextNeedsRefresh = false;
	} else {
		generationsSinceRebuild++;
	}

	const bool upscalerAvailable = isUpscalerCtxLoaded && upscalerCtx;

	if (task == ofxStableDiffusionTask::ImageToVideo) {
		if (videoTaskData.upscalerSettings.enabled && !upscalerAvailable) {
			videoTaskData.upscalerSettings.enabled = false;
			sd->setLastError(
				ofxStableDiffusionErrorCode::UpscaleFailed,
				"Upscaler context is not available for video generation");
			finishTask();
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
					if (isCancellationRequested()) {
						generationError = "Animated video generation cancelled";
						break;
					}

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
						generationError = "Animated video generation returned no frame output for frame " +
							std::to_string(frameIndex);
						ofLogError("ofxStableDiffusion")
							<< "Frame " << frameIndex << " generation failed: " << generationError;
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

					if (isCancellationRequested()) {
						ofxSdReleaseImageArray(frameOutput, 1);
						generationError = "Animated video generation cancelled";
						break;
					}

					OwnedImage generatedFrame;
					if (!generatedFrame.assign(frameOutput[0])) {
						ofxSdReleaseImageArray(frameOutput, 1);
						generationError = "Animated video generation produced an invalid frame at index " +
							std::to_string(frameIndex) + " (width=" +
							std::to_string(frameOutput[0].width) + ", height=" +
							std::to_string(frameOutput[0].height) + ", channels=" +
							std::to_string(frameOutput[0].channel) + ")";
						ofLogError("ofxStableDiffusion")
							<< "Frame " << frameIndex << " assignment failed: invalid image data";
						break;
					}

					ofxSdReleaseImageArray(frameOutput, 1);
					generatedFrames.push_back(std::move(generatedFrame));
				}

			videoTaskData.animationProgressEnabled = false;
			videoTaskData.animationFrameIndex = 0;
			videoTaskData.animationFrameCount = 0;
			videoTaskData.animationSampleSteps = 0;

			if (isCancellationRequested()) {
				finishTask(true, generationError.empty() ? "Animated video generation cancelled" : generationError);
				return;
			}

			if (generatedFrames.size() != static_cast<std::size_t>(frameCount)) {
				if (!generationError.empty()) {
					sd->setLastError(generationError);
				}
				finishTask();
				return;
			}

			sd_image_t* output = makeOwnedImageArray(generatedFrames);
			const float elapsedMs =
				static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
			if (!output) {
				sd->setLastError("Animated video generation could not allocate output frames");
				finishTask();
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
			generationContextNeedsRefresh =
				generationContextSettings.freeParamsImmediately;
			finishTask();
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
			finishTask();
			return;
		}
		if (isCancellationRequested()) {
			ofxSdReleaseImageArray(output, generatedFrameCount);
			finishTask(true, "Video generation cancelled");
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
		generationContextNeedsRefresh =
			generationContextSettings.freeParamsImmediately;
		finishTask();
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
		finishTask();
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
			finishTask();
			return;
		}

		for (int i = 0; i < imageTaskData.request.batchCount; i++) {
			sd_image_t upscaled = upscale(upscalerCtx, output[i], imageTaskData.upscalerSettings.multiplier);
			if (!upscaled.data) {
				ofxSdReleaseImage(output[i]);
				ofxSdReleaseImageArray(output, imageTaskData.request.batchCount);
				sd->setLastError(ofxStableDiffusionErrorCode::UpscaleFailed, "Upscaling failed for one or more images");
				finishTask();
				return;
			}
			ofxSdReleaseImage(output[i]);
			output[i] = upscaled;
		}
	}

	const float elapsedMs = static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
	if (!output) {
		sd->setLastError("Image generation returned no images");
		finishTask();
		return;
	}

	if (isCancellationRequested()) {
		ofxSdReleaseImageArray(output, imageTaskData.request.batchCount);
		finishTask(true, "Image generation cancelled");
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
	generationContextNeedsRefresh =
		generationContextSettings.freeParamsImmediately;
	finishTask();
}

void stableDiffusionThread::requestCancellation() {
	cancellationRequested.store(true);
}

bool stableDiffusionThread::isCancellationRequested() const {
	return cancellationRequested.load();
}

void stableDiffusionThread::resetCancellation() {
	cancellationRequested.store(false);
}
