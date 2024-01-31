#include "stableDiffusionThread.h"
#include "ofApp.h"

void stableDiffusionThread::threadedFunction() {
	ofApp* thread = (ofApp*)userData;
	if (thread->isModelLoading) {
		if (sd_ctx != NULL) {
			free_sd_ctx(sd_ctx);
		}
		sd_ctx = new_sd_ctx(&thread->modelPath[0], &thread->vaePath[0],  &thread->taesdPath[0], &thread->loraModelDir[0],  thread->isVaeDecodeOnly, thread->isVaeTiling, thread->isFreeParamsImmediatly, thread->numThreads, thread->sdType, thread->rngType, thread->schedule);
		if (upscaler_ctx != NULL) {
			free_upscaler_ctx(upscaler_ctx);
		}
		upscaler_ctx = new_upscaler_ctx(&thread->esrganPath[0], thread->numThreads, thread->sdType);
		thread->isModelLoading = false;
	} else if (thread->isTextToImage) {
		thread->output_images = txt2img(sd_ctx, &thread->prompt[0], &thread->negativePrompt[0], thread->clipSkipLayers, thread->cfgScale, thread->width, thread->height, thread->sampleMethodEnum, thread->sampleSteps,   thread->seed, thread->batchSize);
		if (thread->isESRGAN) {
			for (int i = 0; i < thread->batchSize; i++) {
				thread->output_images[i] = upscale(upscaler_ctx, thread->output_images[i], thread->esrganMultiplier);
			}
		}
		thread->diffused = true;
	} else {
		thread->output_images = img2vid(sd_ctx, thread->input_image, thread->width, thread->height, thread->batchSize, 60, 8,0.07, 1,2.9, thread->sampleMethodEnum, thread->sampleSteps, thread->strength, thread->seed);
		if (thread->isESRGAN) {
			for (int i = 0; i < thread->batchSize; i++) {
				thread->output_images[i] = upscale(upscaler_ctx, thread->output_images[i], thread->esrganMultiplier);
			}
		}
		thread->diffused = true;
	}
}