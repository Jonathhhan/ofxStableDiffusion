#pragma once

#include "ofMain.h"
#include "bridges/ofxStableDiffusionHoloscanBridge.h"
#include "core/ofxStableDiffusionParameterTuningHelpers.h"
#include "core/ofxStableDiffusionTypes.h"
#include "video/ofxStableDiffusionLongVideoManifest.h"
#include "video/ofxStableDiffusionLongVideoWorkflow.h"
#include "video/ofxStableDiffusionVideoWorkflowHelpers.h"
#include "ofxStableDiffusionThread.h"
#include "stable-diffusion.h"

#include <deque>
#include <functional>
#include <mutex>
#include <vector>

/// Callback type for generation progress reporting.
/// @param step Current step number.
/// @param steps Total number of steps.
/// @param time Time elapsed in seconds.
using ofxSdProgressCallback = std::function<void(int step, int steps, float time)>;
using ofxSdImageRankCallback = std::function<std::vector<ofxStableDiffusionImageScore>(
	const ofxStableDiffusionImageRequest& request,
	const std::vector<ofxStableDiffusionImageFrame>& images)>;

class ofxStableDiffusion {
public:
	ofxStableDiffusion();
	virtual ~ofxStableDiffusion();

	void configureContext(const ofxStableDiffusionContextSettings& settings);
	void generate(const ofxStableDiffusionImageRequest& request);
	void generateVideo(const ofxStableDiffusionVideoRequest& request);
	void setUpscalerSettings(const ofxStableDiffusionUpscalerSettings& settings);
	ofxStableDiffusionContextSettings getContextSettings() const;
	ofxStableDiffusionUpscalerSettings getUpscalerSettings() const;
	ofxStableDiffusionCapabilities getCapabilities() const;
	ofxStableDiffusionResult getLastResult() const;
	std::vector<ofxStableDiffusionImageFrame> getImages() const;
	ofxStableDiffusionVideoClip getVideoClip() const;
	bool hasImageResult() const;
	bool hasVideoResult() const;
	int getOutputCount() const;
	std::string getLastError() const;
	ofxStableDiffusionErrorCode getLastErrorCode() const;
	ofxStableDiffusionError getLastErrorInfo() const;
	std::vector<ofxStableDiffusionError> getErrorHistory() const;
	void clearErrorHistory();
	int getVideoFrameIndexForTime(float seconds) const;
	const ofPixels* getVideoFramePixels(int index) const;
	bool saveVideoFrames(const std::string& directory, const std::string& prefix = "frame") const;
	bool saveVideoMetadata(const std::string& path) const;
	bool saveVideoFramesWithMetadata(
		const std::string& directory,
		const std::string& prefix = "frame",
		const std::string& metadataFilename = "metadata.json") const;
	bool saveVideoWebm(const std::string& path, int quality = 90) const;
	void setVideoGenerationMode(ofxStableDiffusionVideoMode mode);
	ofxStableDiffusionVideoMode getVideoGenerationMode() const;
	void setImageGenerationMode(ofxStableDiffusionImageMode mode);
	ofxStableDiffusionImageMode getImageGenerationMode() const;
	void setImageSelectionMode(ofxStableDiffusionImageSelectionMode mode);
	ofxStableDiffusionImageSelectionMode getImageSelectionMode() const;
	void setImageRankCallback(ofxSdImageRankCallback cb);
	int getSelectedImageIndex() const;

	/// Load an input image from ofPixels. The pixels must remain valid for the
	/// lifetime of any subsequent img2img / img2vid call.
	void loadImage(const ofPixels& pixels);

	/// Replace the active LoRA/LoCon stack for subsequent generations.
	void setLoras(const std::vector<ofxStableDiffusionLora>& loras_);
	std::vector<ofxStableDiffusionLora> getLoras() const;

	/// Reload embeddings by rebuilding the context with the current (or provided) embed directory.
	void reloadEmbeddings(const std::string& embedDir = "");

	/// Query current embeddings on disk (resolved from embedDir); loads names and paths.
	std::vector<std::pair<std::string, std::string>> listEmbeddings() const;

	bool isDiffused() const;
	void setDiffused(bool diffused);
	/// Returned buffers are owned by the addon; they become invalid after a new
	/// generation starts, output is cleared, or the addon is destroyed.
	sd_image_t* returnImages() const;

	/// Return the human-readable name for an sd_type_t enum value.
	const char* typeName(enum sd_type_t type);

	int32_t getNumPhysicalCores();
	const char* getSystemInfo();

	/// Set an optional progress callback that fires on each diffusion step.
	void setProgressCallback(ofxSdProgressCallback cb);

	void newSdCtx(const ofxStableDiffusionContextSettings& settings);
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
	void instructImage(sd_image_t initImage,
		const std::string& instruction,
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
		bool normalizeInput);

	void img2vid(sd_image_t init_image,
		int width,
		int height,
		int videoFrames,
		int fps,
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

	/// Returns the actual seed used in the last generation (auto-generated if seed was -1).
	int64_t getLastUsedSeed() const;

	/// Returns the seed history from recent generations (up to 20 most recent).
	std::vector<int64_t> getSeedHistory() const;

	/// Clear the seed history.
	void clearSeedHistory();

	/// Hash a string to a deterministic seed value for reproducibility.
	static int64_t hashStringToSeed(const std::string& text);

	std::string prompt;
	std::string instruction;
	std::string negativePrompt;
	int width = 512;
	int height = 512;
	float cfgScale = 7.0f;
	int batchCount = 1;
	float strength = 0.5f;
	int64_t seed = -1;
	int clipSkip = -1;
	// Context model paths
	std::string modelPath;
	std::string modelName;
	std::string diffusionModelPath;
	std::string clipLPath;
	std::string clipGPath;
	std::string t5xxlPath;
	std::string taesdPath;
	std::string controlNetPathCStr;
	std::string embedDirCStr;
	std::string loraModelDir;
	std::string vaePath;
	std::string esrganPath;
	std::string stackedIdEmbedDirCStr;
	std::string inputIdImagesPath;
	sample_method_t sampleMethodEnum = EULER_A_SAMPLE_METHOD;
	int sampleSteps = 20;
	int videoFrames = 6;
	int fps = 6;
	float vaceStrength = 1.0f;
	bool vaeDecodeOnly = false;
	bool vaeTiling = false;
	bool freeParamsImmediately = false;
	bool isFullScreen = false;
	bool isTAESD = false;
	bool isESRGAN = false;
	bool keepClipOnCpu = false;
	bool keepControlNetCpu = false;
	bool keepVaeOnCpu = false;
	bool offloadParamsToCpu = false;
	bool flashAttn = false;
	bool enableMmap = true;
	float styleStrength = 20.0f;
	bool normalizeInput = true;
	int nThreads = -1;
	int esrganMultiplier = 4;
	sd_type_t wType = SD_TYPE_COUNT;
	sd_backend_t backend = SD_BACKEND_CUDA;
	scheduler_t schedule = SCHEDULER_COUNT;
	rng_type_t rngType = STD_DEFAULT_RNG;
	prediction_t prediction = EPS_PRED;
	lora_apply_mode_t loraApplyMode = LORA_APPLY_AUTO;
	std::string controlImagePath;
	float controlStrength = 0.9f;
	std::vector<ofxStableDiffusionLora> loras;

	sd_image_t inputImage = {0, 0, 0, nullptr};
	sd_image_t maskImage = {0, 0, 0, nullptr};
	sd_image_t endImage = {0, 0, 0, nullptr};
	sd_image_t* outputImages = nullptr;
	sd_image_t* controlCond = nullptr;
	stableDiffusionThread thread;
	bool isTextToImage = false;
	bool isImageToVideo = false;
	bool isModelLoading = false;
	bool diffused = false;

	ofxSdProgressCallback progressCallback;
	ofxSdImageRankCallback imageRankCallback;

private:
	friend class stableDiffusionThread;

	bool beginBackgroundTask(ofxStableDiffusionTask task);
	void applyContextSettings(const ofxStableDiffusionContextSettings& settings);
	void applyImageRequest(const ofxStableDiffusionImageRequest& request);
	void applyVideoRequest(const ofxStableDiffusionVideoRequest& request);
	bool validateImageRequestAndSetError(const ofxStableDiffusionImageRequest& request, ofxStableDiffusionTask task);
	bool validateVideoRequestAndSetError(const ofxStableDiffusionVideoRequest& request);
	void clearOutputState();
	ofxStableDiffusionContextSettings captureContextSettingsNoLock() const;
	ofxStableDiffusionUpscalerSettings captureUpscalerSettingsNoLock() const;
	void setLastError(const std::string& errorMessage, ofxStableDiffusionErrorCode code = ofxStableDiffusionErrorCode::Unknown);
	void setLastError(ofxStableDiffusionErrorCode code, const std::string& errorMessage);
	void clearLastError();
	void captureImageResults(
		sd_image_t* images,
		int count,
		int64_t seedValue,
		float elapsedMs,
		ofxStableDiffusionTask task,
		const ofxStableDiffusionImageRequest& request,
		const ofxSdImageRankCallback& rankCallback);
	void captureVideoResults(
		sd_image_t* images,
		int count,
		int64_t seedValue,
		const std::vector<int64_t>& frameSeeds,
		const std::vector<ofxStableDiffusionGenerationParameters>& frameGeneration,
		float elapsedMs,
		ofxStableDiffusionTask task,
		const ofxStableDiffusionVideoRequest& request);
	void applyImageRanking(
		std::vector<ofxStableDiffusionImageFrame>& frames,
		ofxStableDiffusionResult& result,
		const ofxStableDiffusionImageRequest& request,
		const ofxSdImageRankCallback& rankCallback);
	std::vector<sd_image_t> buildOutputImageViews(const ofxStableDiffusionResult& result) const;
	ofPixels makePixelsCopy(const sd_image_t& image) const;

	ofxStableDiffusionTask activeTask = ofxStableDiffusionTask::None;
	ofxStableDiffusionImageMode imageMode = ofxStableDiffusionImageMode::TextToImage;
	ofxStableDiffusionImageSelectionMode imageSelectionMode =
		ofxStableDiffusionImageSelectionMode::KeepOrder;
	ofxStableDiffusionVideoMode videoMode = ofxStableDiffusionVideoMode::Standard;
	ofxStableDiffusionResult lastResult;
	std::vector<sd_image_t> outputImageViews;
	std::string lastError;
	ofxStableDiffusionError lastErrorInfo;
	std::deque<ofxStableDiffusionError> errorHistory;
	static constexpr std::size_t maxErrorHistorySize = 10;
	std::deque<int64_t> seedHistory;
	static constexpr std::size_t maxSeedHistorySize = 20;
	uint64_t taskStartMicros = 0;
	stableDiffusionThread::OwnedImage loadedInputImage;
	mutable std::mutex stateMutex;
};


