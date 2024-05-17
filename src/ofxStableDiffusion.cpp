#include "ofxStableDiffusion.h"

//--------------------------------------------------------------
void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
	if (level <= SD_LOG_INFO) {
		fputs(log, stdout);
		fflush(stdout);
	}
	else {
		fputs(log, stderr);
		fflush(stderr);
	}
}

//--------------------------------------------------------------
ofxStableDiffusion::ofxStableDiffusion() {
	sd_set_log_callback(sd_log_cb, NULL);
}

//--------------------------------------------------------------
ofxStableDiffusion::~ofxStableDiffusion() {}

//--------------------------------------------------------------
void ofxStableDiffusion::loadImage(ofPixels pixels) {
	inputImage = { (uint32_t)width,
		(uint32_t)height,
		3,
		pixels.getData() };
}

bool ofxStableDiffusion::isDiffused() const {
	return diffused;
}

sd_image_t* ofxStableDiffusion::returnImages() const {
	return outputImages;
}

void ofxStableDiffusion::setDiffused(bool diffused_) {
	diffused = diffused_;
}

//--------------------------------------------------------------
void ofxStableDiffusion::typeName(enum sd_type_t type) {
	sd_type_name(type);
}

//--------------------------------------------------------------
int32_t ofxStableDiffusion::getNumPhysicalCores() {
	return get_num_physical_cores();
}

//--------------------------------------------------------------
const char* ofxStableDiffusion::getSystemInfo() {
	return sd_get_system_info();
}

//--------------------------------------------------------------
void ofxStableDiffusion::newSdCtx(std::string modelPath_,
	std::string vaePath_,
	std::string taesdPath_,
	std::string controlNetPathCStr_,
	std::string loraModelDir_,
	std::string embedDirCStr_,
	std::string stackedIdEmbedDirCStr_,
	bool vaeDecodeOnly_,
	bool vaeTiling_,
	bool freeParamsImmediately_,
	int nThreads_,
	enum sd_type_t wType_,
	enum rng_type_t rngType_,
	enum schedule_t schedule_,
	bool keepClipOnCpu_,
	bool keepControlNetCpu_,
	bool keepVaeOnCpu_) {
	if (!thread.isThreadRunning()) {
		isModelLoading = true;
		modelPath = modelPath_;
		vaePath = vaePath_;
		taesdPath = taesdPath_;
		controlNetPathCStr = controlNetPathCStr_;
		loraModelDir = loraModelDir_;
		embedDirCStr = embedDirCStr_;
		stackedIdEmbedDirCStr = stackedIdEmbedDirCStr_;
		vaeDecodeOnly = vaeDecodeOnly_;
		vaeTiling = vaeTiling_;
		freeParamsImmediately = freeParamsImmediately_;
		nThreads = nThreads_;
		wType = wType_;
		rngType = rngType_;
		schedule = schedule_;
		keepClipOnCpu = keepClipOnCpu_;
		keepControlNetCpu = keepControlNetCpu_;
		keepVaeOnCpu = keepVaeOnCpu_;
		thread.userData = this;
		thread.startThread();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusion::freeSdCtx() {
	free_sd_ctx(thread.sdCtx);
}

//--------------------------------------------------------------
void ofxStableDiffusion::txt2img(std::string prompt_,
	std::string negativePrompt_,
	int clipSkip_,
	float cfgScale_,
	int width_,
	int height_,
	enum sample_method_t sampleMethod_,
	int sampleSteps_,
	int64_t seed_,
	int batchCount_,
	sd_image_t* controlCond_,
	float controlStrength_,
	float styleStrength_,
	bool normalizeInput_,
	std::string inputIdImagesPath_) {
	if (!thread.isThreadRunning()) {
		isTextToImage = true;
		prompt = prompt_;
		negativePrompt = negativePrompt_;
		clipSkip = clipSkip_;
		cfgScale = cfgScale_;
		width = width_;
		height = height_;
		sampleMethodEnum = sampleMethod_;
		sampleSteps = sampleSteps_;
		seed = seed_;
		batchCount = batchCount_;
		controlCond = controlCond_;
		controlStrength = controlStrength_;
		styleStrength = styleStrength_;
		normalizeInput = normalizeInput_;
		inputIdImagesPath = inputIdImagesPath_;
		thread.userData = this;
		thread.startThread();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusion::img2img(sd_image_t initImage_,
	std::string prompt_,
	std::string negativePrompt_,
	int clipSkip_,
	float cfgScale_,
	int width_,
	int height_,
	enum sample_method_t sampleMethod_,
	int sampleSteps_,
	float strength_,
	int64_t seed_,
	int batchCount_,
	sd_image_t* controlCond_,
	float controlStrength_,
	float styleStrength_,
	bool normalizeInput_,
	std::string inputIdImagesPath_) {
	if (!thread.isThreadRunning()) {
		isTextToImage = false;
		inputImage = initImage_;
		prompt = prompt_;
		negativePrompt = negativePrompt_;
		clipSkip = clipSkip_;
		cfgScale = cfgScale_;
		width = width_;
		height = height_;
		sampleMethodEnum = sampleMethod_;
		sampleSteps = sampleSteps_;
		strength = strength_;
		seed = seed_;
		batchCount = batchCount_;
		controlCond = controlCond_;
		controlStrength =controlStrength_;
		styleStrength = styleStrength_;
		normalizeInput = normalizeInput_;
		inputIdImagesPath = inputIdImagesPath_;
		thread.userData = this;
		thread.startThread();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusion::img2vid(sd_image_t initImage_,
	int width_,
	int height_,
	int videoFrames_,
	int motionBucketId_,
	int fps_,
	float augmentationLevel_,
	float minCfg_,
	float cfgScale_,
	enum sample_method_t sampleMethod_,
	int sampleSteps_,
	float strength_,
	int64_t seed_) {
	if (!thread.isThreadRunning()) {
		inputImage = initImage_;
		width = width_;
		height = height_;
		videoFrames = videoFrames_;
		motionBucketId = motionBucketId_;
		fps = fps_;
		sampleMethodEnum = sampleMethod_;
		sampleSteps = sampleSteps_;
		strength = strength_;
		seed = seed_;
		thread.userData = this;
		thread.startThread();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusion::newUpscalerCtx(const char* esrganPath,
	int nThreads,
	enum sd_type_t wType) {
	new_upscaler_ctx(esrganPath, nThreads, wType);
}

//--------------------------------------------------------------
void ofxStableDiffusion::freeUpscalerCtx() {
	free_upscaler_ctx(thread.upscalerCtx);
}

//--------------------------------------------------------------
void ofxStableDiffusion::upscale(upscaler_ctx_t* upscalerCtx, sd_image_t inputImage, uint32_t upscaleFactor) {
	upscale(upscalerCtx, inputImage, upscaleFactor);
}

//--------------------------------------------------------------
bool ofxStableDiffusion::convert(const char* inputPath, const char* vaePath, const char* outputPath, sd_type_t outputType) {
	return convert(inputPath, vaePath, outputPath, outputType);
}

//--------------------------------------------------------------
void ofxStableDiffusion::preprocessCanny(uint8_t* img,
	int width,
	int height,
	float highThreshold,
	float lowThreshold,
	float weak,
	float strong,
	bool inverse) {
	preprocess_canny(img,
		width,
		height,
		highThreshold,
		lowThreshold,
		weak,
		strong,
		inverse);
}