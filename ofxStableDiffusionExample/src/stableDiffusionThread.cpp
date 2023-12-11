#include "stableDiffusionThread.h"
#include "ofApp.h"

void stableDiffusionThread::threadedFunction() {
	ofApp *thread = (ofApp*)userData;
	if (thread->isModelLoading) {
		thread->stableDiffusion.load_from_file(&thread->modelPath[0], &thread->vaePath[0], thread->ggmlType, thread->schedule, thread->clipSkipLayers);
		thread->isModelLoading = false;
	} else if (thread->isTextToImage) {
		thread->stableDiffusionPixelVector = thread->stableDiffusion.txt2img(thread->prompt, thread->negativePrompt, thread->cfgScale, thread->width, thread->height, thread->sampleMethodEnum, thread->sampleSteps, thread->seed, thread->batchSize);
		thread->diffused = true;
	} else {
		thread->stableDiffusionPixelVector = thread->stableDiffusion.img2img(thread->pixels.getData(), thread->prompt, thread->negativePrompt, thread->cfgScale, thread->width, thread->height, thread->sampleMethodEnum, thread->sampleSteps, thread->strength, thread->seed);
		thread->diffused = true;
	}
}