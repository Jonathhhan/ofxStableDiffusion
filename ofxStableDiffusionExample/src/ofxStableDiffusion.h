#pragma once

#include "ofMain.h"
#include "stableDiffusionThread.h"
#include "../libs/stable-diffusion/include/stable-diffusion.h"

class ofxStableDiffusion {
public:
	void loadImage(ofPixels pixels);
	void typeName(enum sd_type_t type);
	void setLogCallback(sd_log_cb_t sd_log_cb, void* data);
	void setProgressCallback(sd_progress_cb_t cb, void* data);
	int32_t getNumPhysicalCores();
	const char* getSystemInfo();
	sd_ctx_t* newSdCtx(std::string model_path,
		std::string vae_path,
		std::string taesd_path,
		std::string control_net_path_c_str,
		std::string lora_model_dir,
		std::string embed_dir_c_str,
		std::string stacked_id_embed_dir_c_str,
		bool vae_decode_only,
		bool vae_tiling,
		bool free_params_immediately,
		int n_threads,
		enum sd_type_t wtype,
		enum rng_type_t rng_type,
		enum schedule_t s,
		bool keep_clip_on_cpu,
		bool keep_control_net_cpu,
		bool keep_vae_on_cpu);
	void freeSdCtx(sd_ctx_t* sdCtx);
	sd_image_t* txt2img(sd_ctx_t* sd_ctx,
		std::string prompt,
		std::string negative_prompt,
		int clip_skip,
		float cfg_scale,
		int width,
		int height,
		enum sample_method_t sample_method,
		int sample_steps,
		int64_t seed,
		int batch_count,
		const sd_image_t* control_cond,
		float control_strength,
		float style_strength,
		bool normalize_input,
		std::string input_id_images_path);
	sd_image_t* img2img(sd_ctx_t* sd_ctx,
		sd_image_t init_image,
		std::string prompt,
		std::string negative_prompt,
		int clip_skip,
		float cfg_scale,
		int width,
		int height,
		enum sample_method_t sample_method,
		int sample_steps,
		float strength,
		int64_t seed,
		int batch_count);
	sd_image_t* ofxStableDiffusion::img2vid(sd_ctx_t* sd_ctx,
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
		int64_t seed);
	void newUpscalerCtx(const char* esrgan_path,
		int n_threads,
		enum sd_type_t wtype);
	void freeUpscalerCtx(upscaler_ctx_t* upscaler_ctx);
	void upscale(upscaler_ctx_t* upscaler_ctx,
		sd_image_t input_image,
		uint32_t upscale_factor);
	bool convert(const char* input_path,
		const char* vae_path,
		const char* output_path,
		sd_type_t output_type);
	void preprocessCanny(uint8_t* img,
		int width,
		int height,
		float high_threshold,
		float low_threshold,
		float weak,
		float strong,
		bool inverse);

	std::string prompt;
	std::string negativePrompt;
	int width;
	int height;
	float cfgScale;
	int batchSize;
	float strength;
	int seed;
	int clipSkipLayers;
	const char* sampleMethod;
	std::string modelPath;
	std::string modelName;
	std::string taesdPath;
	std::string controlNetPath;
	std::string embedDir;
	std::string loraModelDir;
	std::string vaePath;
	std::string esrganPath;
	std::string stackedIdEmbedDir;
	std::string inputIdImagesPath;
	sample_method_t sampleMethodEnum;
	int sampleSteps;
	bool isVaeDecodeOnly;
	bool isVaeTiling;
	bool isFreeParamsImmediatly;
	bool isFullScreen;
	bool isTAESD;
	bool isESRGAN;
	bool keepClipOnCpu;
	bool keepControlNetCpu;
	bool keepVaeOnCpu;
	float styleStrength;
	bool normalizeInput;
	int numThreads;
	int esrganMultiplier;
	sd_type_t sdType;
	schedule_t schedule;
	rng_type_t rngType;
	std::string controlImagePath;
	float controlStrength;

	sd_image_t inputImage;
	sd_image_t* outputImages;
	sd_image_t* controlImage;
	stableDiffusionThread thread;
	bool isTextToImage;
	bool isModelLoading;
	bool diffused;
};