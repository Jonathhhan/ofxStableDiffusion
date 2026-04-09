#include "ofxStableDiffusion.h"

//--------------------------------------------------------------
static void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
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
static void sd_progress_cb(int step, int steps, float time, void* data) {
	auto* self = static_cast<ofxStableDiffusion*>(data);
	if (self && self->progressCallback) {
		self->progressCallback(step, steps, time);
	}
}

//--------------------------------------------------------------
ofxStableDiffusion::ofxStableDiffusion() {
	sd_set_log_callback(sd_log_cb, nullptr);
}

//--------------------------------------------------------------
ofxStableDiffusion::~ofxStableDiffusion() {}

//--------------------------------------------------------------
void ofxStableDiffusion::loadImage(const ofPixels& pixels) {
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
const char* ofxStableDiffusion::typeName(enum sd_type_t type) {
	return sd_type_name(type);
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
void ofxStableDiffusion::setProgressCallback(ofxSdProgressCallback cb) {
	progressCallback = cb;
	if (progressCallback) {
		sd_set_progress_callback(sd_progress_cb, this);
	} else {
		sd_set_progress_callback(nullptr, nullptr);
	}
}

//--------------------------------------------------------------
bool ofxStableDiffusion::isGenerating() const {
	return thread.isThreadRunning();
}

//--------------------------------------------------------------
void ofxStableDiffusion::newSdCtx(const std::string& modelPath_,
	const std::string& vaePath_,
	const std::string& taesdPath_,
	const std::string& controlNetPathCStr_,
	const std::string& loraModelDir_,
	const std::string& embedDirCStr_,
	const std::string& stackedIdEmbedDirCStr_,
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
		isTextToImage = false;
		isImageToVideo = false;
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
	if (thread.sdCtx) {
		free_sd_ctx(thread.sdCtx);
		thread.sdCtx = nullptr;
	}
}

//--------------------------------------------------------------
void ofxStableDiffusion::txt2img(const std::string& prompt_,
	const std::string& negativePrompt_,
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
	const std::string& inputIdImagesPath_) {
	if (!thread.isThreadRunning()) {
		isTextToImage = true;
		isImageToVideo = false;
		isModelLoading = false;
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
	const std::string& prompt_,
	const std::string& negativePrompt_,
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
	const std::string& inputIdImagesPath_) {
	if (!thread.isThreadRunning()) {
		isTextToImage = false;
		isImageToVideo = false;
		isModelLoading = false;
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
		controlStrength = controlStrength_;
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
		isTextToImage = false;
		isImageToVideo = true;
		isModelLoading = false;
		inputImage = initImage_;
		width = width_;
		height = height_;
		videoFrames = videoFrames_;
		motionBucketId = motionBucketId_;
		fps = fps_;
		augmentationLevel = augmentationLevel_;
		minCfg = minCfg_;
		cfgScale = cfgScale_;
		sampleMethodEnum = sampleMethod_;
		sampleSteps = sampleSteps_;
		strength = strength_;
		seed = seed_;
		thread.userData = this;
		thread.startThread();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusion::newUpscalerCtx(const std::string& esrganPath_,
	int nThreads_,
	enum sd_type_t wType_) {
	if (thread.upscalerCtx) {
		free_upscaler_ctx(thread.upscalerCtx);
		thread.upscalerCtx = nullptr;
	}
	thread.upscalerCtx = new_upscaler_ctx(esrganPath_.c_str(), nThreads_, wType_);
}

//--------------------------------------------------------------
void ofxStableDiffusion::freeUpscalerCtx() {
	if (thread.upscalerCtx) {
		free_upscaler_ctx(thread.upscalerCtx);
		thread.upscalerCtx = nullptr;
	}
}

//--------------------------------------------------------------
sd_image_t ofxStableDiffusion::upscaleImage(sd_image_t inputImage_, uint32_t upscaleFactor) {
	return upscale(thread.upscalerCtx, inputImage_, upscaleFactor);
}

//--------------------------------------------------------------
bool ofxStableDiffusion::convert(const char* inputPath_, const char* vaePath_, const char* outputPath_, sd_type_t outputType_) {
	return ::convert(inputPath_, vaePath_, outputPath_, outputType_);
}

//--------------------------------------------------------------
uint8_t* ofxStableDiffusion::preprocessCanny(uint8_t* img,
	int width_,
	int height_,
	float highThreshold,
	float lowThreshold,
	float weak,
	float strong,
	bool inverse) {
	return preprocess_canny(img,
		width_,
		height_,
		highThreshold,
		lowThreshold,
		weak,
		strong,
		inverse);
}