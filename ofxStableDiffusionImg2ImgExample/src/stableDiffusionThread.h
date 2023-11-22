#pragma once

#include "ofMain.h"
#include "stable-diffusion.h"

class stableDiffusionThread : public ofThread {
public:
	StableDiffusion stableDiffusion;
	std::vector<uint8_t>  pixels;
	std::vector<uint8_t> stableDiffusionPixelVectorTemp;
	std::vector<uint8_t> stableDiffusionPixelVector;
	std::string prompt;
	std::string negativePrompt;
	float cfgScale;
	int width;
	int height;
	sd_sample_method sampleMethod;
	int sampleSteps;
	float strength;
	int seed;
	bool diffused;
private:
	void threadedFunction();
};
