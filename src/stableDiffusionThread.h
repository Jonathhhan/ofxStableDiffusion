#pragma once

#include "ofMain.h"
#include "../libs/ofxStableDiffusion/include/stable-diffusion.h"

class stableDiffusionThread : public ofThread {
public:
	void* userData;
	upscaler_ctx_t* upscalerCtx;
	sd_ctx_t* sdCtx;
private:
	void threadedFunction();
	bool isSdCtxLoaded = false;
	bool isUpscalerCtxLoaded = false;
};
