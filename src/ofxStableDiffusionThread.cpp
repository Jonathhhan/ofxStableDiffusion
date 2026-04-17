#include "ofxStableDiffusionThread.h"
#include "ofxStableDiffusion.h"

stableDiffusionThread::~stableDiffusionThread() {
	if (isThreadRunning()) {
		waitForThread(true);
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
}

void stableDiffusionThread::threadedFunction() {
	ofxStableDiffusion* sd = static_cast<ofxStableDiffusion*>(userData);
	if (!sd) {
		return;
	}

	if (sd->activeTask == ofxStableDiffusionTask::LoadModel || sd->isModelLoading) {
		if (sdCtx) {
			free_sd_ctx(sdCtx);
			sdCtx = nullptr;
		}
		sdCtx = new_sd_ctx(sd->modelPath.c_str(),
			sd->vaePath.c_str(),
			sd->taesdPath.c_str(),
			sd->controlNetPathCStr.c_str(),
			sd->loraModelDir.c_str(),
			sd->embedDirCStr.c_str(),
			sd->stackedIdEmbedDirCStr.c_str(),
			sd->vaeDecodeOnly,
			sd->vaeTiling,
			sd->freeParamsImmediately,
			sd->nThreads,
			sd->wType,
			sd->rngType,
			sd->schedule,
			sd->keepClipOnCpu,
			sd->keepControlNetCpu,
			sd->keepVaeOnCpu);
		if (upscalerCtx) {
			free_upscaler_ctx(upscalerCtx);
			upscalerCtx = nullptr;
			isUpscalerCtxLoaded = false;
		}
		if (sd->isESRGAN && !sd->esrganPath.empty()) {
			upscalerCtx = new_upscaler_ctx(sd->esrganPath.c_str(), sd->nThreads, sd->wType);
			isUpscalerCtxLoaded = (upscalerCtx != nullptr);
		}
		isSdCtxLoaded = (sdCtx != nullptr);
		sd->isModelLoading = false;
		sd->activeTask = ofxStableDiffusionTask::None;
		if (!isSdCtxLoaded) {
			sd->setLastError("Failed to create stable-diffusion context");
		}
		return;
	}

	if (!sdCtx) {
		sd->activeTask = ofxStableDiffusionTask::None;
		sd->setLastError("Stable Diffusion context is not loaded");
		return;
	}

	if (sd->activeTask == ofxStableDiffusionTask::ImageToVideo || sd->isImageToVideo) {
		sd_image_t* output = img2vid(sdCtx,
			sd->inputImage,
			sd->width,
			sd->height,
			sd->videoFrames,
			sd->motionBucketId,
			sd->fps,
			sd->augmentationLevel,
			sd->minCfg,
			sd->cfgScale,
			sd->sampleMethodEnum,
			sd->sampleSteps,
			sd->strength,
			sd->seed);
		const float elapsedMs = static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
		if (!output) {
			sd->activeTask = ofxStableDiffusionTask::None;
			sd->setLastError("Image-to-video generation returned no frames");
			return;
		}
		sd->captureVideoResults(output, sd->videoFrames, sd->seed, elapsedMs);
		sd->activeTask = ofxStableDiffusionTask::None;
		return;
	}

	if (sd->activeTask == ofxStableDiffusionTask::TextToImage || sd->isTextToImage) {
		sd_image_t* output = txt2img(sdCtx,
			sd->prompt.c_str(),
			sd->negativePrompt.c_str(),
			sd->clipSkip,
			sd->cfgScale,
			sd->width,
			sd->height,
			sd->sampleMethodEnum,
			sd->sampleSteps,
			sd->seed,
			sd->batchCount,
			sd->controlCond,
			sd->controlStrength,
			sd->styleStrength,
			sd->normalizeInput,
			sd->inputIdImagesPath.c_str());
		if (output && sd->isESRGAN && upscalerCtx) {
			for (int i = 0; i < sd->batchCount; i++) {
				output[i] = upscale(upscalerCtx, output[i], sd->esrganMultiplier);
			}
		}
		const float elapsedMs = static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
		if (!output) {
			sd->activeTask = ofxStableDiffusionTask::None;
			sd->setLastError("Text-to-image generation returned no images");
			return;
		}
		sd->captureImageResults(output, sd->batchCount, sd->seed, elapsedMs);
		sd->activeTask = ofxStableDiffusionTask::None;
		return;
	}

	const ofxStableDiffusionTask imageTask = sd->activeTask;
	sd_image_t* output = img2img(sdCtx,
		sd->inputImage,
		sd->prompt.c_str(),
		sd->negativePrompt.c_str(),
		sd->clipSkip,
		sd->cfgScale,
		sd->width,
		sd->height,
		sd->sampleMethodEnum,
		sd->sampleSteps,
		sd->strength,
		sd->seed,
		sd->batchCount,
		sd->controlCond,
		sd->controlStrength,
		sd->styleStrength,
		sd->normalizeInput,
		sd->inputIdImagesPath.c_str());
	if (output && sd->isESRGAN && upscalerCtx) {
		for (int i = 0; i < sd->batchCount; i++) {
			output[i] = upscale(upscalerCtx, output[i], sd->esrganMultiplier);
		}
	}
	const float elapsedMs = static_cast<float>(ofGetElapsedTimeMicros() - sd->taskStartMicros) / 1000.0f;
	if (!output) {
		sd->setLastError(
			std::string(ofxStableDiffusionTaskLabel(imageTask)) +
			" returned no images");
		sd->activeTask = ofxStableDiffusionTask::None;
		return;
	}
	sd->captureImageResults(output, sd->batchCount, sd->seed, elapsedMs);
	sd->activeTask = ofxStableDiffusionTask::None;
}
