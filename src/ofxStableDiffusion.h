#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionThread.h"
#include "../../libs/stable-diffusion/include/stable-diffusion.h"

class ofxStableDiffusion {
public:
	ofxStableDiffusion();
	virtual ~ofxStableDiffusion();
	void loadImage(ofPixels pixels);
	bool isDiffused() const;
	void setDiffused(bool diffused);
	sd_image_t* returnImages() const;
	void typeName(enum sd_type_t type);
	int32_t getNumPhysicalCores();
	const char* getSystemInfo();
	void newSdCtx(std::string modelPath,
		std::string clipLPath,
		std::string clipGPath,
		std::string t5xxlPath,
		std::string diffusionModelPath,
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
		enum schedule_t s,
		bool keepClipOnCpu,
		bool keepControlNetCpu,
		bool keepVaeOnCpu,
		bool diffusionFlashAttn);
	void freeSdCtx();
	void txt2img(std::string prompt,
		std::string negativePrompt,
		int clipSkip,
		float cfgScale,
		float guidance,
		float eta,
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
		std::string inputIdImagesPath,
		int * skip_layers,
		size_t skip_layers_count,
		float slg_scale,
		float skip_layer_start,
		float skip_layer_end);
	void img2img(sd_image_t initImage,
		sd_image_t maskImage,
		std::string prompt,
		std::string negativePrompt,
		int clipSkip,
		float cfgScale,
		float guidance,
		float eta,
		int width,
		int height,
		enum sample_method_t sampleMethod,
		int sampleSteps,
		float strength,
		int64_t seed,
		int batchCount,
		sd_image_t* controlCond,
		float controlStrength,
		float styleStrength,
		bool normalizeInput,
		std::string inputIdImagesPath,
		int * skip_layers,
		size_t skip_layers_count,
		float slg_scale,
		float skip_layer_start,
		float skip_layer_end);
	void img2vid(sd_image_t init_image,
		int width,
		int height,
		int videoFrames,
		int motionBucketId,
		int fps,
		float augmentationLevel,
		float minCfg,
		float cfgScale,
		enum sample_method_t sampleMethod,
		int sampleSteps,
		float strength,
		int64_t seed);
	void newUpscalerCtx(const char* esrganPath,
		int nThreads);
	void freeUpscalerCtx();
	void upscale(upscaler_ctx_t* upscalerCtx,
		sd_image_t inputImage,
		uint32_t upscaleFactor);
	bool convert(const char* inputPath,
		const char* vaePath,
		const char* outputPath,
		sd_type_t outputType);
	void preprocessCanny(uint8_t* img,
		int width,
		int height,
		float highThreshold,
		float lowThreshold,
		float weak,
		float strong,
		bool inverse);

	std::string modelPath;
	std::string modelName;
	std::string clipLPath;
	std::string clipGPath;
	std::string t5xxlPath;
	std::string diffusionModelPath;
	std::string vaePath;
	std::string taesdPath;
	std::string controlNetPathCStr;
	std::string embedDirCStr;
	std::string loraModelDir;
	std::string esrganPath;
	std::string stackedIdEmbedDirCStr;
	bool vaeDecodeOnly;
	bool vaeTiling;
	bool freeParamsImmediately;
	int nThreads;
	sd_type_t wType;
	rng_type_t rngType;
	schedule_t schedule;
	bool keepClipOnCpu;
	bool keepControlNetCpu;
	bool keepVaeOnCpu;
	bool diffusionFlashAttn;
	std::string prompt;
	std::string negativePrompt;
	int clipSkip;
	float cfgScale;
	float guidance;
	float eta;
	int width;
	int height;
	char * sampleMethod;
	sample_method_t sampleMethodEnum;
	int sampleSteps;
	int seed;
	int batchCount;
	sd_image_t * controlCond;
	float controlStrength;
	float styleStrength;
	bool normalizeInput;
	std::string inputIdImagesPath;
	int * skipLayers;
	size_t skipLayersCount;
	float slgScale;
	float skipLayerStart;
	float skipLayerEnd;
	sd_image_t initImage;
	sd_image_t maskImage;
	int videoFrames;
	int motionBucketId;
	int fps;
	float augmentationLevel;
	float minCfg;
	float strength;

	bool isFullScreen;
	bool isTAESD;
	bool isESRGAN;
	int esrganMultiplier;

	//std::string controlImagePath;
	sd_image_t* outputImages;
	
	stableDiffusionThread thread;
	bool isTextToImage = false;
	bool isModelLoading = false;
	bool diffused = false;
};
