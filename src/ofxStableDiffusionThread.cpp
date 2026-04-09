#include "ofxStableDiffusionThread.h"
#include "ofxStableDiffusion.h"

void stableDiffusionThread::threadedFunction() {
	ofxStableDiffusion* sd = static_cast<ofxStableDiffusion*>(userData);
	if (sd->isModelLoading) {
		if (isSdCtxLoaded) {
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
		if (isUpscalerCtxLoaded) {
			free_upscaler_ctx(upscalerCtx);
			upscalerCtx = nullptr;
			isUpscalerCtxLoaded = false;
		}
		if (sd->isESRGAN) {
			upscalerCtx = new_upscaler_ctx(sd->esrganPath.c_str(),
				sd->nThreads,
				sd->wType);
			isUpscalerCtxLoaded = (upscalerCtx != nullptr);
		}
		isSdCtxLoaded = (sdCtx != nullptr);
		sd->isModelLoading = false;
	}
	else if (sd->isImageToVideo) {
		sd->outputImages = img2vid(sdCtx,
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
		sd->diffused = true;
	}
	else if (sd->isTextToImage) {
		sd->outputImages = txt2img(sdCtx,
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
		if (sd->isESRGAN && upscalerCtx) {
			for (int i = 0; i < sd->batchCount; i++) {
				sd->outputImages[i] = upscale(upscalerCtx, sd->outputImages[i], sd->esrganMultiplier);
			}
		}
		sd->diffused = true;
	}
	else {
		sd->outputImages = img2img(sdCtx,
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
		if (sd->isESRGAN && upscalerCtx) {
			for (int i = 0; i < sd->batchCount; i++) {
				sd->outputImages[i] = upscale(upscalerCtx, sd->outputImages[i], sd->esrganMultiplier);
			}
		}
		sd->diffused = true;
	}
}