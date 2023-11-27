#include "stableDiffusionThread.h"

void stableDiffusionThread::threadedFunction() {
	stableDiffusionPixelVector = stableDiffusion.img2img(pixels, prompt, negativePrompt, cfgScale, width, height, sampleMethod, sampleSteps, strength, seed);
	diffused = true;
}
