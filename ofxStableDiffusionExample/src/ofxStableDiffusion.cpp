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
void ofxStableDiffusion::loadImage(ofPixels pixels) {
	inputImage = { (uint32_t)width,
		(uint32_t)height,
		3,
		pixels.getData() };
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
sd_ctx_t* ofxStableDiffusion::newSdCtx(std::string modelPath,
	std::string vaePath,
	std::string taesdPath,
	std::string controlNetPathCStr,
	std::string loraModelDir,
	std::string embedDirCStr,
	std::string stackedIdEmbedDirCStr,
					bool vaeDecodeOnly,
					bool vaeTiling,
					bool freeParamsImmediately,
					int nThreads,
					enum sd_type_t wType,
					enum rng_type_t rngType,
					enum schedule_t schedule,
					bool keepClipOnCpu,
					bool keepControlNetCpu,
					bool keepVaeOnCpu) {

		if (!thread.isThreadRunning()) {
			isModelLoading = true;
			thread.userData = this;
			thread.startThread();
		}
	return new_sd_ctx(&modelPath[0],
		&vaePath[0],
		&taesdPath[0],
		&controlNetPathCStr[0],
		&loraModelDir[0],
		&embedDirCStr[0],
		&stackedIdEmbedDirCStr[0],
		vaeDecodeOnly,
		vaeTiling,
		freeParamsImmediately,
		nThreads,
		wType,
		rngType,
		schedule,
		keepClipOnCpu,
		keepControlNetCpu,
		keepVaeOnCpu);
}

//--------------------------------------------------------------
void ofxStableDiffusion::freeSdCtx(sd_ctx_t* sdCtx) {
	free_sd_ctx(sdCtx);
}

//--------------------------------------------------------------
sd_image_t* ofxStableDiffusion::txt2img(sd_ctx_t* sdCtx,
					std::string prompt,
					std::string negativePrompt,
					int clipSkip,
					float cfgScale,
					int width,
					int height,
					enum sample_method_t sampleMethod,
					int sampleSteps,
					int64_t seed,
					int batchCount,
					const sd_image_t* controlCond,
					float controlStrength,
					float styleStrength,
					bool normalizeInput,
					std::string inputIdImagesPath) {
	return txt2img(sdCtx,
		&prompt[0],
		&negativePrompt[0],
		clipSkip,
		cfgScale,
		width,
		height,
		sampleMethod,
		sampleSteps,
		seed,
		batchCount,
		controlCond,
		controlStrength,
		styleStrength,
		normalizeInput,
		&inputIdImagesPath[0]);
}

//--------------------------------------------------------------
sd_image_t* ofxStableDiffusion::img2img(sd_ctx_t* sdCtx,
					sd_image_t initImage,
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
					int batch_count) {
	return img2img(sdCtx,
		initImage,
		&prompt[0],
		&negativePrompt[0],
		clipSkip, 
		cfgScale,
		width,
		height,
		sample_method,
		sample_steps,
		strength,
		seed,
		batch_count);
}

//--------------------------------------------------------------
sd_image_t* ofxStableDiffusion::img2vid(sd_ctx_t* sdCtx,
					sd_image_t init_image,
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
					int64_t seed) {
	return img2vid(sdCtx,
		init_image,
		width,
		height,
		video_frames,
		motion_bucket_id,
		fps,
		augmentation_level,
		min_cfg,
		cfg_scale,
		sample_method,
		sample_steps,
		strength,
		seed);
}

//--------------------------------------------------------------
void ofxStableDiffusion::newUpscalerCtx(const char* esrganPath,
	int nThreads,
	enum sd_type_t wType) {
	new_upscaler_ctx(esrganPath, nThreads, wType);
}

//--------------------------------------------------------------
void ofxStableDiffusion::freeUpscalerCtx(upscaler_ctx_t* upscaler_ctx) {
	free_upscaler_ctx(upscaler_ctx);
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