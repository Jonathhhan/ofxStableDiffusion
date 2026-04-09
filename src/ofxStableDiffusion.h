#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionThread.h"
#include "../../libs/stable-diffusion/include/stable-diffusion.h"

/// Callback type for generation progress reporting.
/// @param step Current step number.
/// @param steps Total number of steps.
/// @param time Time elapsed in seconds.
typedef std::function<void(int step, int steps, float time)> ofxSdProgressCallback;

class ofxStableDiffusion {
public:
	ofxStableDiffusion();
	virtual ~ofxStableDiffusion();

	/// Load an input image from ofPixels. The pixels must remain valid for the
	/// lifetime of any subsequent img2img / img2vid call.
	void loadImage(const ofPixels& pixels);

	bool isDiffused() const;
	void setDiffused(bool diffused);
	sd_image_t* returnImages() const;

	/// Return the human-readable name for an sd_type_t enum value.
	const char* typeName(enum sd_type_t type);

	int32_t getNumPhysicalCores();
	const char* getSystemInfo();

	/// Set an optional progress callback that fires on each diffusion step.
	void setProgressCallback(ofxSdProgressCallback cb);

	void newSdCtx(const std::string& modelPath,
		const std::string& vaePath,
		const std::string& taesdPath,
		const std::string& controlNetPathCStr,
		const std::string& loraModelDir,
		const std::string& embedDirCStr,
		const std::string& stackedIdEmbedDirCStr,
		bool vaeDecodeOnly,
		bool vaeTiling,
		bool freeParamsImmediately,
		int nThreads,
		enum sd_type_t wType,
		enum rng_type_t rngType,
		enum schedule_t s,
		bool keepClipOnCpu,
		bool keepControlNetCpu,
		bool keepVaeOnCpu);
	void freeSdCtx();

	void txt2img(const std::string& prompt,
		const std::string& negativePrompt,
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
		const std::string& inputIdImagesPath);

	void img2img(sd_image_t initImage,
		const std::string& prompt,
		const std::string& negativePrompt,
		int clipSkip,
		float cfgScale,
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
		const std::string& inputIdImagesPath);

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

	void newUpscalerCtx(const std::string& esrganPath,
		int nThreads,
		enum sd_type_t wType);
	void freeUpscalerCtx();

	sd_image_t upscaleImage(sd_image_t inputImage,
		uint32_t upscaleFactor);

	bool convert(const char* inputPath,
		const char* vaePath,
		const char* outputPath,
		sd_type_t outputType);

	/// Run Canny edge detection on an image buffer in-place.
	/// Returns a pointer to the processed data (allocated by the library).
	uint8_t* preprocessCanny(uint8_t* img,
		int width,
		int height,
		float highThreshold,
		float lowThreshold,
		float weak,
		float strong,
		bool inverse);

	/// Returns true if generation is currently in progress.
	bool isGenerating() const;

	std::string prompt;
	std::string negativePrompt;
	int width = 512;
	int height = 512;
	float cfgScale = 7.0f;
	int batchCount = 1;
	float strength = 0.5f;
	int seed = -1;
	int clipSkip = -1;
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
	sample_method_t sampleMethodEnum = EULER_A;
	int sampleSteps = 20;
	int videoFrames = 6;
	int motionBucketId = 127;
	int fps = 6;
	float augmentationLevel = 0.0f;
	float minCfg = 1.0f;
	bool vaeDecodeOnly = false;
	bool vaeTiling = false;
	bool freeParamsImmediately = false;
	bool isFullScreen = false;
	bool isTAESD = false;
	bool isESRGAN = false;
	bool keepClipOnCpu = false;
	bool keepControlNetCpu = false;
	bool keepVaeOnCpu = false;
	float styleStrength = 20.0f;
	bool normalizeInput = true;
	int nThreads = -1;
	int esrganMultiplier = 4;
	sd_type_t wType = SD_TYPE_COUNT;
	schedule_t schedule = DEFAULT;
	rng_type_t rngType = STD_DEFAULT_RNG;
	std::string controlImagePath;
	float controlStrength = 0.9f;

	sd_image_t inputImage = {0, 0, 0, nullptr};
	sd_image_t* outputImages = nullptr;
	sd_image_t* controlCond = nullptr;
	stableDiffusionThread thread;
	bool isTextToImage = false;
	bool isImageToVideo = false;
	bool isModelLoading = false;
	bool diffused = false;

	ofxSdProgressCallback progressCallback;
};