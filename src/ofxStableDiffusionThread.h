#pragma once

#include "ofMain.h"
#include "../libs/stable-diffusion/include/stable-diffusion.h"
#include <vector>

class stableDiffusionThread : public ofThread {
public:
	void* userData = nullptr;
	upscaler_ctx_t* upscalerCtx = nullptr;
	sd_ctx_t* sdCtx = nullptr;
	~stableDiffusionThread() override;
	void clearContexts();
private:
	void threadedFunction();
	bool isSdCtxLoaded = false;
	bool isUpscalerCtxLoaded = false;
	std::vector<sd_lora_t> loraBuffer;
};
