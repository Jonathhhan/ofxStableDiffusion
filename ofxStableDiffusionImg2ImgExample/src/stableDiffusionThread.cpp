#include "stableDiffusionThread.h"

void stableDiffusionThread::threadedFunction() {
	stableDiffusionPixelVectorTemp = stableDiffusion.img2img(pixels, prompt, negativePrompt, cfgScale, width, height, sampleMethod, sampleSteps, strength, seed);
	stableDiffusionPixelVector = stableDiffusionPixelVectorTemp;
	diffused = true;
}