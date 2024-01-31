#pragma once

#include "ofMain.h"
#include "stable-diffusion.h"

class stableDiffusionThread : public ofThread {
public:
	void* userData;
private:
	void threadedFunction();
	sd_ctx_t* sd_ctx;
	upscaler_ctx_t* upscaler_ctx;
};
