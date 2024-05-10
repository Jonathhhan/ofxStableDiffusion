#include "stableDiffusionThread.h"
#include "ofxStableDiffusion.h"

void stableDiffusionThread::threadedFunction() {
	ofxStableDiffusion* thread = (ofxStableDiffusion*)userData;
	if (thread->isModelLoading) {
		if (isSdCtxLoaded) {
			free_sd_ctx(sdCtx);
		}
		sdCtx = new_sd_ctx(&thread->modelPath[0],
			&thread->vaePath[0],
			&thread->taesdPath[0],
			&thread->controlNetPathCStr[0],
			&thread->loraModelDir[0],
			&thread->embedDirCStr[0],
			&thread->stackedIdEmbedDirCStr[0],
			thread->vaeDecodeOnly,
			thread->vaeTiling,
			thread->freeParamsImmediately,
			thread->nThreads,
			thread->wType,
			thread->rngType,
			thread->schedule,
			thread->keepClipOnCpu,
			thread->keepControlNetCpu,
			thread->keepVaeOnCpu);
		if (isUpscalerCtxLoaded) {
			free_upscaler_ctx(upscalerCtx);
		}
		if (thread->isESRGAN) {
			upscalerCtx = new_upscaler_ctx(&thread->esrganPath[0],
				thread->nThreads,
				thread->wType);
				isUpscalerCtxLoaded = true;
		}
		isSdCtxLoaded = true;
		thread->isModelLoading = false;
	}
	else if (thread->isTextToImage) {
		thread->outputImages = txt2img(sdCtx,
			&thread->prompt[0],
			&thread->negativePrompt[0],
			thread->clipSkip,
			thread->cfgScale,
			thread->width,
			thread->height,
			thread->sampleMethodEnum,
			thread->sampleSteps,
			thread->seed,
			thread->batchCount,
			thread->controlCond,
			thread->controlStrength,
			thread->styleStrength,
			thread->normalizeInput,
			&thread->inputIdImagesPath[0]);
		if (thread->isESRGAN) {
			for (int i = 0; i < thread->batchCount; i++) {
				thread->outputImages[i] = upscale(upscalerCtx, thread->outputImages[i], thread->esrganMultiplier);
			}
		}
		thread->diffused = true;
	}
	else {
		thread->outputImages = img2img(sdCtx,
			thread->inputImage,
			&thread->prompt[0],
			&thread->negativePrompt[0],
			thread->clipSkip,
			thread->cfgScale,
			thread->width,
			thread->height,
			thread->sampleMethodEnum,
			thread->sampleSteps,
			thread->strength,
			thread->seed,
			thread->batchCount);
		if (thread->isESRGAN) {
			for (int i = 0; i < thread->batchCount; i++) {
				thread->outputImages[i] = upscale(upscalerCtx, thread->outputImages[i], thread->esrganMultiplier);
			}
		}
		thread->diffused = true;
	}
}