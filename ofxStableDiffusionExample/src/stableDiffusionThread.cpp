#include "stableDiffusionThread.h"

void stableDiffusionThread::threadedFunction() {
	if (isTextToImage) {
		stableDiffusionPixelVector = stableDiffusion.txt2img(prompt, negativePrompt, cfgScale, width, height, sampleMethod, sampleSteps, seed, batch_count);
	} else {
		stableDiffusionPixelVector = stableDiffusion.img2img(pixels, prompt, negativePrompt, cfgScale, width, height, sampleMethod, sampleSteps, strength, seed);
	}
	diffused = true;
}