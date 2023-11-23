#include "stableDiffusionThread.h"

void stableDiffusionThread::threadedFunction() {
	stableDiffusionPixelVectorTemp = stableDiffusion.txt2img(prompt, negativePrompt, cfgScale, width, height, sampleMethod, sampleSteps, seed, batch_count);
	stableDiffusionPixelVector = stableDiffusionPixelVectorTemp;
	diffused = true;
}