#include "ofxStableDiffusion.h"

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

void ofxStableDiffusion::setDiffused(bool diffuse) {
	diffused = diffuse;
}

//--------------------------------------------------------------
void ofxStableDiffusion::typeName(enum sd_type_t type) {
	sd_type_name(type);
}

//--------------------------------------------------------------
void ofxStableDiffusion::setLogCallback(sd_log_cb_t sd_log_cb, void* data) {
	sd_set_log_callback(sd_log_cb, data);
}

//--------------------------------------------------------------
void ofxStableDiffusion::setProgressCallback(sd_progress_cb_t cb, void* data) {
	sd_set_progress_callback(cb, data);
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
void ofxStableDiffusion::txt2img(std::string prompt,
	std::string negativePrompt,
	int clipSkip,
	float cfgScale,
	int width,
	int height,
	enum sample_method_t sampleMethod,
	int sampleSteps,
	int64_t seed,
	int batchCount,
	sd_image_t* controlCond,
	float controlStrength,
	float styleStrength,
	bool normalizeInput,
	std::string inputIdImagesPath) {
	isTextToImage = true;
	this->prompt = prompt;
	this->negativePrompt = negativePrompt;
	this->clipSkip = clipSkip;
	this->cfgScale = cfgScale;
	this->width = width;
	this->height = height;
	this->sampleMethodEnum = sampleMethod;
	this->sampleSteps = sampleSteps;
	this->seed = seed;
	this->batchCount = batchCount;
	this->controlCond = controlCond;
	this->controlStrength = controlStrength;
	this->styleStrength = styleStrength;
	this->normalizeInput = normalizeInput;
	this->inputIdImagesPath = inputIdImagesPath;
	if (!thread.isThreadRunning()) {
		thread.userData = this;
		thread.startThread();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusion::img2img(sd_image_t initImage,
	std::string prompt,
	std::string negativePrompt,
	int clipSkip,
	float cfgScale,
	int width,
	int height,
	enum sample_method_t sample_method,
	int sample_steps,
	float strength,
	int64_t seed,
	int batch_count) const {
}

//--------------------------------------------------------------
void ofxStableDiffusion::img2vid(sd_image_t init_image,
	int width,
	int height,
	int video_frames,
	int motion_bucket_id,
	int fps,
	float augmentation_level,
	float min_cfg,
	float cfg_scale,
	enum sample_method_t sample_method,
	int sample_steps,
	float strength,
	int64_t seed) const {
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
void ofxStableDiffusion::upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor) {
	upscale(upscaler_ctx, input_image, upscale_factor);
}

//--------------------------------------------------------------
bool ofxStableDiffusion::convert(const char* input_path, const char* vae_path, const char* output_path, sd_type_t output_type) {
	return convert(input_path, vae_path, output_path, output_type);
}

//--------------------------------------------------------------
void ofxStableDiffusion::preprocessCanny(uint8_t* img,
	int width,
	int height,
	float high_threshold,
	float low_threshold,
	float weak,
	float strong,
	bool inverse) {
	preprocess_canny(img,
		width,
		height,
		high_threshold,
		low_threshold,
		weak,
		strong,
		inverse);
}