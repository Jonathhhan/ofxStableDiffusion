#pragma once

#include "ofMain.h"
#include "stable-diffusion.h"

class stableDiffusionThread : public ofThread {
public:
	StableDiffusion stableDiffusion;
	const uint8_t* pixels;
	std::vector<uint8_t*> stableDiffusionPixelVector;
	std::string prompt;
	std::string negativePrompt;
	float cfgScale;
	int width;
	int height;
	SampleMethod sampleMethod;
	int sampleSteps;
	float strength;
	int seed;
	int batch_count;
	bool isTextToImage;
	bool diffused;
private:
	void threadedFunction();
};
