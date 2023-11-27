#pragma once

#include "ofMain.h"
#include "stable-diffusion.h"

class stableDiffusionThread : public ofThread {
public:
	StableDiffusion stableDiffusion;
	std::vector<uint8_t*> stableDiffusionPixelVectorTemp;
	std::vector<uint8_t*> stableDiffusionPixelVector;
	std::string prompt;
	std::string negativePrompt;
	float cfgScale;
	int width;
	int height;
	SampleMethod sampleMethod;
	int sampleSteps;
	int seed;
	int batch_count;
	bool diffused;
private:
	void threadedFunction();
};
