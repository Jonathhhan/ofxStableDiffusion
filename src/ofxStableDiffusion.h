#pragma once

#include "ofMain.h"
#include "stableDiffusionThread.h"
#include "../libs/ofxStableDiffusion/include/stable-diffusion.h"
#include <thread>
class ofxStableDiffusion {
public:
	ofxStableDiffusion();
	void loadImage(ofPixels pixels);
	bool isDiffused();
	void setDiffused(bool diffused);
	sd_image_t* returnImages();
	void typeName(enum sd_type_t type);
	void setLogCallback(sd_log_cb_t sd_log_cb, void* data);
	void setProgressCallback(sd_progress_cb_t cb, void* data);
	int32_t getNumPhysicalCores();
	const char* getSystemInfo();
	void newSdCtx(std::string model_path,
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
	void freeSdCtx();
	void txt2img(std::string prompt,
		std::string negativePrompt,
		int clipSkip,
		float cfgScale,
		int width,
		int height,
		sample_method_t sampleMethod,
		int sampleSteps,
		int64_t seed,
		int batchCount,
		sd_image_t* controlCond,
		float controlStrength,
		float styleStrength,
		bool normalizeInput,
		std::string inputIdImagesPath);
	void img2img(sd_image_t init_image,
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
		int batch_count) const;
	void img2vid(sd_image_t init_image,
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
		int64_t seed) const;
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
	int batchCount;
	float strength;
	int seed;
	int clipSkip;
	char* sampleMethod;
	std::string modelPath;
	std::string modelName;
	std::string taesdPath;
	std::string controlNetPathCStr;
	std::string embedDirCStr;
	std::string loraModelDir;
	std::string vaePath;
	std::string esrganPath;
	std::string stackedIdEmbedDirCStr;
	std::string inputIdImagesPath;
	sample_method_t sampleMethodEnum;
	int sampleSteps;
	bool vaeDecodeOnly;
	bool vaeTiling;
	bool freeParamsImmediately;
	bool isFullScreen;
	bool isTAESD;
	bool isESRGAN;
	bool keepClipOnCpu;
	bool keepControlNetCpu;
	bool keepVaeOnCpu;
	float styleStrength;
	bool normalizeInput;
	int nThreads;
	int esrganMultiplier;
	sd_type_t wType;
	schedule_t schedule;
	rng_type_t rngType;
	std::string controlImagePath;
	float controlStrength;

	sd_image_t inputImage;
	sd_image_t* outputImages;
	sd_image_t* controlCond;
	stableDiffusionThread thread;
	bool isTextToImage;
	bool isModelLoading;
	bool diffused;
};