#include "ofxStableDiffusion.h"

#include <algorithm>

//--------------------------------------------------------------
static void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
	if (level <= SD_LOG_INFO) {
		fputs(log, stdout);
		fflush(stdout);
	} else {
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
	thread.userData = this;
}

//--------------------------------------------------------------
ofxStableDiffusion::~ofxStableDiffusion() {
	if (thread.isThreadRunning()) {
		thread.waitForThread(true);
	}
	thread.clearContexts();
}

//--------------------------------------------------------------
static ofxStableDiffusionErrorCode validateDimensions(int width, int height, std::string& errorMsg) {
	if (width <= 0 || height <= 0) {
		errorMsg = "Width and height must be positive values";
		return ofxStableDiffusionErrorCode::InvalidDimensions;
	}
	if (width > 2048 || height > 2048) {
		errorMsg = "Width and height must not exceed 2048 pixels";
		return ofxStableDiffusionErrorCode::InvalidDimensions;
	}
	if (width % 64 != 0) {
		errorMsg = "Width must be a multiple of 64 (recommended: 512, 768, 1024)";
		return ofxStableDiffusionErrorCode::InvalidDimensions;
	}
	if (height % 64 != 0) {
		errorMsg = "Height must be a multiple of 64 (recommended: 512, 768, 1024)";
		return ofxStableDiffusionErrorCode::InvalidDimensions;
	}
	return ofxStableDiffusionErrorCode::None;
}

//--------------------------------------------------------------
static ofxStableDiffusionErrorCode validateBatchCount(int batchCount, std::string& errorMsg) {
	if (batchCount <= 0) {
		errorMsg = "Batch count must be positive";
		return ofxStableDiffusionErrorCode::InvalidBatchCount;
	}
	if (batchCount > 16) {
		errorMsg = "Batch count exceeds maximum of 16 (risk of out-of-memory)";
		return ofxStableDiffusionErrorCode::InvalidBatchCount;
	}
	return ofxStableDiffusionErrorCode::None;
}

//--------------------------------------------------------------
void ofxStableDiffusion::configureContext(const ofxStableDiffusionContextSettings& settings) {
	applyContextSettings(settings);
}

//--------------------------------------------------------------
void ofxStableDiffusion::generate(const ofxStableDiffusionImageRequest& request) {
	std::string validationError;
	ofxStableDiffusionErrorCode errorCode;

	// Validate dimensions
	errorCode = validateDimensions(request.width, request.height, validationError);
	if (errorCode != ofxStableDiffusionErrorCode::None) {
		imageMode = request.mode;
		activeTask = ofxStableDiffusionTaskForImageMode(request.mode);
		setLastError(errorCode, validationError);
		return;
	}

	// Validate batch count
	errorCode = validateBatchCount(request.batchCount, validationError);
	if (errorCode != ofxStableDiffusionErrorCode::None) {
		imageMode = request.mode;
		activeTask = ofxStableDiffusionTaskForImageMode(request.mode);
		setLastError(errorCode, validationError);
		return;
	}

	const sd_image_t candidateInputImage =
		request.initImage.data != nullptr ? request.initImage : inputImage;
	if (ofxStableDiffusionImageModeUsesInputImage(request.mode) && candidateInputImage.data == nullptr) {
		imageMode = request.mode;
		activeTask = ofxStableDiffusionTaskForImageMode(request.mode);
		setLastError(ofxStableDiffusionErrorCode::MissingInputImage, "Selected image mode requires an input image");
		return;
	}

	if (!beginBackgroundTask(ofxStableDiffusionTaskForImageMode(request.mode))) {
		return;
	}
	applyImageRequest(request);
	thread.startThread();
}

//--------------------------------------------------------------
void ofxStableDiffusion::generateVideo(const ofxStableDiffusionVideoRequest& request) {
	std::string validationError;
	ofxStableDiffusionErrorCode errorCode;

	// Validate dimensions
	errorCode = validateDimensions(request.width, request.height, validationError);
	if (errorCode != ofxStableDiffusionErrorCode::None) {
		activeTask = ofxStableDiffusionTask::ImageToVideo;
		setLastError(errorCode, validationError);
		return;
	}

	// Validate frame count
	if (request.frameCount <= 0 || request.frameCount > 100) {
		activeTask = ofxStableDiffusionTask::ImageToVideo;
		setLastError(ofxStableDiffusionErrorCode::InvalidFrameCount, "Frame count must be between 1 and 100");
		return;
	}

	if (!beginBackgroundTask(ofxStableDiffusionTask::ImageToVideo)) {
		return;
	}
	applyVideoRequest(request);
	thread.startThread();
}

//--------------------------------------------------------------
void ofxStableDiffusion::setUpscalerSettings(const ofxStableDiffusionUpscalerSettings& settings) {
	esrganPath = settings.modelPath;
	nThreads = settings.nThreads;
	wType = settings.weightType;
	esrganMultiplier = settings.multiplier;
	isESRGAN = settings.enabled;
}

//--------------------------------------------------------------
ofxStableDiffusionContextSettings ofxStableDiffusion::getContextSettings() const {
	return {
		modelPath,
		vaePath,
		taesdPath,
		controlNetPathCStr,
		loraModelDir,
		embedDirCStr,
		stackedIdEmbedDirCStr,
		vaeDecodeOnly,
		vaeTiling,
		freeParamsImmediately,
		nThreads,
		wType,
		rngType,
		schedule,
		keepClipOnCpu,
		keepControlNetCpu,
		keepVaeOnCpu
	};
}

//--------------------------------------------------------------
ofxStableDiffusionUpscalerSettings ofxStableDiffusion::getUpscalerSettings() const {
	return {esrganPath, nThreads, wType, esrganMultiplier, isESRGAN};
}

//--------------------------------------------------------------
const ofxStableDiffusionResult& ofxStableDiffusion::getLastResult() const {
	return lastResult;
}

//--------------------------------------------------------------
const std::vector<ofxStableDiffusionImageFrame>& ofxStableDiffusion::getImages() const {
	return lastResult.images;
}

//--------------------------------------------------------------
const ofxStableDiffusionVideoClip& ofxStableDiffusion::getVideoClip() const {
	return lastResult.video;
}

//--------------------------------------------------------------
bool ofxStableDiffusion::hasImageResult() const {
	return lastResult.hasImages();
}

//--------------------------------------------------------------
bool ofxStableDiffusion::hasVideoResult() const {
	return lastResult.hasVideo();
}

//--------------------------------------------------------------
int ofxStableDiffusion::getOutputCount() const {
	if (lastResult.hasVideo()) {
		return static_cast<int>(lastResult.video.frames.size());
	}
	return static_cast<int>(lastResult.images.size());
}

//--------------------------------------------------------------
const std::string& ofxStableDiffusion::getLastError() const {
	return lastError;
}

//--------------------------------------------------------------
ofxStableDiffusionErrorCode ofxStableDiffusion::getLastErrorCode() const {
	return lastErrorInfo.code;
}

//--------------------------------------------------------------
const ofxStableDiffusionError& ofxStableDiffusion::getLastErrorInfo() const {
	return lastErrorInfo;
}

//--------------------------------------------------------------
const std::vector<ofxStableDiffusionError>& ofxStableDiffusion::getErrorHistory() const {
	return errorHistory;
}

//--------------------------------------------------------------
void ofxStableDiffusion::clearErrorHistory() {
	errorHistory.clear();
}

//--------------------------------------------------------------
int ofxStableDiffusion::getVideoFrameIndexForTime(float seconds) const {
	return lastResult.video.frameIndexForTime(seconds);
}

//--------------------------------------------------------------
const ofPixels* ofxStableDiffusion::getVideoFramePixels(int index) const {
	if (index < 0 || index >= static_cast<int>(lastResult.video.frames.size())) {
		return nullptr;
	}
	return &lastResult.video.frames[static_cast<std::size_t>(index)].pixels;
}

//--------------------------------------------------------------
bool ofxStableDiffusion::saveVideoFrames(const std::string& directory, const std::string& prefix) const {
	return lastResult.video.saveFrameSequence(directory, prefix);
}

//--------------------------------------------------------------
void ofxStableDiffusion::setVideoGenerationMode(ofxStableDiffusionVideoMode mode) {
	videoMode = mode;
}

//--------------------------------------------------------------
ofxStableDiffusionVideoMode ofxStableDiffusion::getVideoGenerationMode() const {
	return videoMode;
}

//--------------------------------------------------------------
void ofxStableDiffusion::setImageGenerationMode(ofxStableDiffusionImageMode mode) {
	imageMode = mode;
	isTextToImage = (mode == ofxStableDiffusionImageMode::TextToImage);
	isImageToVideo = false;
}

//--------------------------------------------------------------
ofxStableDiffusionImageMode ofxStableDiffusion::getImageGenerationMode() const {
	return imageMode;
}

//--------------------------------------------------------------
void ofxStableDiffusion::setImageSelectionMode(ofxStableDiffusionImageSelectionMode mode) {
	imageSelectionMode = mode;
}

//--------------------------------------------------------------
ofxStableDiffusionImageSelectionMode ofxStableDiffusion::getImageSelectionMode() const {
	return imageSelectionMode;
}

//--------------------------------------------------------------
void ofxStableDiffusion::setImageRankCallback(ofxSdImageRankCallback cb) {
	imageRankCallback = cb;
}

//--------------------------------------------------------------
int ofxStableDiffusion::getSelectedImageIndex() const {
	return lastResult.selectedImageIndex;
}

//--------------------------------------------------------------
void ofxStableDiffusion::loadImage(const ofPixels& pixels) {
	inputImage = {
		static_cast<uint32_t>(pixels.getWidth()),
		static_cast<uint32_t>(pixels.getHeight()),
		static_cast<uint32_t>(pixels.getNumChannels()),
		const_cast<unsigned char*>(pixels.getData())
	};
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
	return sd_get_num_physical_cores();
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
int64_t ofxStableDiffusion::getLastUsedSeed() const {
	return lastResult.actualSeedUsed;
}

//--------------------------------------------------------------
const std::vector<int64_t>& ofxStableDiffusion::getSeedHistory() const {
	return seedHistory;
}

//--------------------------------------------------------------
void ofxStableDiffusion::clearSeedHistory() {
	seedHistory.clear();
}

//--------------------------------------------------------------
int64_t ofxStableDiffusion::hashStringToSeed(const std::string& text) {
	return ofxStableDiffusionHashStringToSeed(text);
}

//--------------------------------------------------------------
bool ofxStableDiffusion::beginBackgroundTask(ofxStableDiffusionTask task) {
	if (thread.isThreadRunning()) {
		ofLogWarning("ofxStableDiffusion") << "Ignoring request while another task is still running";
		return false;
	}

	activeTask = task;
	taskStartMicros = ofGetElapsedTimeMicros();
	isModelLoading = (task == ofxStableDiffusionTask::LoadModel);
	isTextToImage = (task == ofxStableDiffusionTask::TextToImage);
	isImageToVideo = (task == ofxStableDiffusionTask::ImageToVideo);
	diffused = false;
	clearLastError();
	clearOutputState();
	thread.userData = this;
	return true;
}

//--------------------------------------------------------------
void ofxStableDiffusion::applyContextSettings(const ofxStableDiffusionContextSettings& settings) {
	modelPath = settings.modelPath;
	modelName = ofFilePath::getFileName(modelPath);
	vaePath = settings.vaePath;
	taesdPath = settings.taesdPath;
	controlNetPathCStr = settings.controlNetPath;
	loraModelDir = settings.loraModelDir;
	embedDirCStr = settings.embedDir;
	stackedIdEmbedDirCStr = settings.stackedIdEmbedDir;
	vaeDecodeOnly = settings.vaeDecodeOnly;
	vaeTiling = settings.vaeTiling;
	freeParamsImmediately = settings.freeParamsImmediately;
	nThreads = settings.nThreads;
	wType = settings.weightType;
	rngType = settings.rngType;
	schedule = settings.schedule;
	keepClipOnCpu = settings.keepClipOnCpu;
	keepControlNetCpu = settings.keepControlNetCpu;
	keepVaeOnCpu = settings.keepVaeOnCpu;
}

//--------------------------------------------------------------
void ofxStableDiffusion::applyImageRequest(const ofxStableDiffusionImageRequest& request) {
	currentImageRequest = request;
	imageMode = request.mode;
	imageSelectionMode = request.selectionMode;
	if (request.initImage.data != nullptr) {
		inputImage = request.initImage;
	} else if (!ofxStableDiffusionImageModeUsesInputImage(request.mode)) {
		inputImage = {0, 0, 0, nullptr};
	}
	instruction = request.instruction;
	prompt = request.prompt;
	negativePrompt = request.negativePrompt;
	clipSkip = request.clipSkip;
	cfgScale = request.cfgScale;
	width = request.width;
	height = request.height;
	sampleMethodEnum = request.sampleMethod;
	sampleSteps = request.sampleSteps;
	strength = request.strength;
	seed = static_cast<int>(request.seed);
	batchCount = request.batchCount;
	controlCond = request.controlCond;
	controlStrength = request.controlStrength;
	styleStrength = request.styleStrength;
	normalizeInput = request.normalizeInput;
	inputIdImagesPath = request.inputIdImagesPath;
}

//--------------------------------------------------------------
void ofxStableDiffusion::applyVideoRequest(const ofxStableDiffusionVideoRequest& request) {
	inputImage = request.initImage;
	width = request.width;
	height = request.height;
	videoFrames = request.frameCount;
	motionBucketId = request.motionBucketId;
	fps = request.fps;
	augmentationLevel = request.augmentationLevel;
	minCfg = request.minCfg;
	cfgScale = request.cfgScale;
	sampleMethodEnum = request.sampleMethod;
	sampleSteps = request.sampleSteps;
	strength = request.strength;
	seed = static_cast<int>(request.seed);
	videoMode = request.mode;
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
	enum scheduler_t schedule_,
	bool keepClipOnCpu_,
	bool keepControlNetCpu_,
	bool keepVaeOnCpu_) {
	if (!beginBackgroundTask(ofxStableDiffusionTask::LoadModel)) {
		return;
	}

	applyContextSettings({
		modelPath_,
		vaePath_,
		taesdPath_,
		controlNetPathCStr_,
		loraModelDir_,
		embedDirCStr_,
		stackedIdEmbedDirCStr_,
		vaeDecodeOnly_,
		vaeTiling_,
		freeParamsImmediately_,
		nThreads_,
		wType_,
		rngType_,
		schedule_,
		keepClipOnCpu_,
		keepControlNetCpu_,
		keepVaeOnCpu_
	});
	thread.startThread();
}

//--------------------------------------------------------------
void ofxStableDiffusion::freeSdCtx() {
	if (thread.isThreadRunning()) {
		thread.waitForThread(true);
	}
	thread.clearContexts();
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
	if (!beginBackgroundTask(ofxStableDiffusionTask::TextToImage)) {
		return;
	}

	applyImageRequest({
		ofxStableDiffusionImageMode::TextToImage,
		imageSelectionMode,
		{0, 0, 0, nullptr},
		prompt_,
		"",
		negativePrompt_,
		clipSkip_,
		cfgScale_,
		width_,
		height_,
		sampleMethod_,
		sampleSteps_,
		0.5f,
		seed_,
		batchCount_,
		controlCond_,
		controlStrength_,
		styleStrength_,
		normalizeInput_,
		inputIdImagesPath_
	});
	thread.startThread();
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
	if (!beginBackgroundTask(ofxStableDiffusionTask::ImageToImage)) {
		return;
	}

	inputImage = initImage_;
	applyImageRequest({
		ofxStableDiffusionImageMode::ImageToImage,
		imageSelectionMode,
		initImage_,
		prompt_,
		"",
		negativePrompt_,
		clipSkip_,
		cfgScale_,
		width_,
		height_,
		sampleMethod_,
		sampleSteps_,
		strength_,
		seed_,
		batchCount_,
		controlCond_,
		controlStrength_,
		styleStrength_,
		normalizeInput_,
		inputIdImagesPath_
	});
	thread.startThread();
}

//--------------------------------------------------------------
void ofxStableDiffusion::instructImage(sd_image_t initImage_,
	const std::string& instruction_,
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
	bool normalizeInput_) {
	if (!beginBackgroundTask(ofxStableDiffusionTask::InstructImage)) {
		return;
	}

	applyImageRequest({
		ofxStableDiffusionImageMode::InstructImage,
		imageSelectionMode,
		initImage_,
		instruction_,
		instruction_,
		negativePrompt_,
		clipSkip_,
		cfgScale_,
		width_,
		height_,
		sampleMethod_,
		sampleSteps_,
		strength_,
		seed_,
		batchCount_,
		controlCond_,
		controlStrength_,
		styleStrength,
		normalizeInput_,
		""
	});
	thread.startThread();
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
	if (!beginBackgroundTask(ofxStableDiffusionTask::ImageToVideo)) {
		return;
	}

	applyVideoRequest({
		initImage_,
		width_,
		height_,
		videoFrames_,
		motionBucketId_,
		fps_,
		augmentationLevel_,
		minCfg_,
		cfgScale_,
		sampleMethod_,
		sampleSteps_,
		strength_,
		seed_,
		videoMode
	});
	thread.startThread();
}

//--------------------------------------------------------------
void ofxStableDiffusion::newUpscalerCtx(const std::string& esrganPath_,
	int nThreads_,
	enum sd_type_t wType_) {
	if (thread.isThreadRunning()) {
		ofLogWarning("ofxStableDiffusion") << "Cannot rebuild the upscaler while a task is still running";
		return;
	}
	setUpscalerSettings({esrganPath_, nThreads_, wType_, esrganMultiplier, true});
	if (thread.upscalerCtx) {
		free_upscaler_ctx(thread.upscalerCtx);
		thread.upscalerCtx = nullptr;
	}
	thread.upscalerCtx = new_upscaler_ctx(esrganPath_.c_str(), false, false, nThreads_, 0);
}

//--------------------------------------------------------------
void ofxStableDiffusion::freeUpscalerCtx() {
	if (thread.isThreadRunning()) {
		thread.waitForThread(true);
	}
	if (thread.upscalerCtx) {
		free_upscaler_ctx(thread.upscalerCtx);
		thread.upscalerCtx = nullptr;
	}
	isESRGAN = false;
}

//--------------------------------------------------------------
sd_image_t ofxStableDiffusion::upscaleImage(sd_image_t inputImage_, uint32_t upscaleFactor) {
	return upscale(thread.upscalerCtx, inputImage_, upscaleFactor);
}

//--------------------------------------------------------------
bool ofxStableDiffusion::convert(const char* inputPath_, const char* vaePath_, const char* outputPath_, sd_type_t outputType_) {
	return ::convert(inputPath_, vaePath_, outputPath_, outputType_, nullptr, false);
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
	sd_image_t image{
		static_cast<uint32_t>(width_),
		static_cast<uint32_t>(height_),
		1u,
		img
	};
	return ::preprocess_canny(image,
		highThreshold,
		lowThreshold,
		weak,
		strong,
		inverse) ? img : nullptr;
}

//--------------------------------------------------------------
void ofxStableDiffusion::clearOutputState() {
	outputImages = nullptr;
	outputImageViews.clear();
	lastResult = {};
}

//--------------------------------------------------------------
void ofxStableDiffusion::setLastError(const std::string& errorMessage, ofxStableDiffusionErrorCode code) {
	lastError = errorMessage;
	lastErrorInfo.code = code;
	lastErrorInfo.message = errorMessage;
	lastErrorInfo.suggestion = ofxStableDiffusionErrorCodeSuggestion(code);
	lastErrorInfo.timestampMicros = ofGetElapsedTimeMicros();

	// Add to error history
	errorHistory.push_back(lastErrorInfo);
	if (errorHistory.size() > maxErrorHistorySize) {
		errorHistory.erase(errorHistory.begin());
	}

	lastResult.success = false;
	lastResult.task = activeTask;
	lastResult.imageMode = imageMode;
	lastResult.selectionMode = imageSelectionMode;
	lastResult.error = errorMessage;
	diffused = false;
}

//--------------------------------------------------------------
void ofxStableDiffusion::setLastError(ofxStableDiffusionErrorCode code, const std::string& errorMessage) {
	setLastError(errorMessage, code);
}

//--------------------------------------------------------------
void ofxStableDiffusion::clearLastError() {
	lastError.clear();
	lastErrorInfo = ofxStableDiffusionError();
	lastResult.error.clear();
}

//--------------------------------------------------------------
ofPixels ofxStableDiffusion::makePixelsCopy(const sd_image_t& image) const {
	ofPixels pixels;
	if (!image.data || image.width == 0 || image.height == 0) {
		return pixels;
	}

	ofImageType type = OF_IMAGE_COLOR;
	switch (image.channel) {
	case 1: type = OF_IMAGE_GRAYSCALE; break;
	case 4: type = OF_IMAGE_COLOR_ALPHA; break;
	case 3:
	default:
		type = OF_IMAGE_COLOR;
		break;
	}

	pixels.setFromPixels(
		image.data,
		static_cast<int>(image.width),
		static_cast<int>(image.height),
		type);
	return pixels;
}

//--------------------------------------------------------------
void ofxStableDiffusion::captureImageResults(sd_image_t* images, int count, int seedValue, float elapsedMs) {
	lastResult = {};
	lastResult.success = true;
	lastResult.task = activeTask;
	lastResult.imageMode = imageMode;
	lastResult.selectionMode = imageSelectionMode;
	lastResult.elapsedMs = elapsedMs;
	lastResult.actualSeedUsed = seedValue;
	lastResult.images.reserve(std::max(0, count));

	// Add seed to history
	seedHistory.push_back(seedValue);
	if (seedHistory.size() > maxSeedHistorySize) {
		seedHistory.erase(seedHistory.begin());
	}

	for (int i = 0; i < count; ++i) {
		ofxStableDiffusionImageFrame frame;
		frame.index = i;
		frame.sourceIndex = i;
		frame.seed = seedValue;
		frame.pixels = makePixelsCopy(images[i]);
		lastResult.images.push_back(std::move(frame));
	}

	applyImageRanking(lastResult.images);
	rebuildLegacyOutputViews();
	outputImages = outputImageViews.empty() ? nullptr : outputImageViews.data();
	diffused = true;
}

//--------------------------------------------------------------
void ofxStableDiffusion::captureVideoResults(sd_image_t* images, int count, int seedValue, float elapsedMs) {
	lastResult = {};
	lastResult.success = true;
	lastResult.task = activeTask;
	lastResult.elapsedMs = elapsedMs;
	lastResult.actualSeedUsed = seedValue;
	lastResult.video.fps = fps;
	lastResult.video.sourceFrameCount = count;
	lastResult.video.mode = videoMode;

	// Add seed to history
	seedHistory.push_back(seedValue);
	if (seedHistory.size() > maxSeedHistorySize) {
		seedHistory.erase(seedHistory.begin());
	}

	std::vector<ofxStableDiffusionImageFrame> sourceFrames;
	sourceFrames.reserve(std::max(0, count));
	for (int i = 0; i < count; ++i) {
		ofxStableDiffusionImageFrame frame;
		frame.index = i;
		frame.seed = seedValue;
		frame.pixels = makePixelsCopy(images[i]);
		sourceFrames.push_back(std::move(frame));
	}

	lastResult.video.frames = ofxStableDiffusionBuildVideoFrames(sourceFrames, videoMode);
	rebuildLegacyOutputViews();
	outputImages = outputImageViews.empty() ? nullptr : outputImageViews.data();
	diffused = true;
}

//--------------------------------------------------------------
void ofxStableDiffusion::rebuildLegacyOutputViews() {
	outputImageViews.clear();

	const auto appendViews = [this](const std::vector<ofxStableDiffusionImageFrame>& frames) {
		outputImageViews.reserve(frames.size());
		for (const auto& frame : frames) {
			if (!frame.isAllocated()) {
				continue;
			}
			outputImageViews.push_back({
				static_cast<uint32_t>(frame.pixels.getWidth()),
				static_cast<uint32_t>(frame.pixels.getHeight()),
				static_cast<uint32_t>(frame.pixels.getNumChannels()),
				const_cast<unsigned char*>(frame.pixels.getData())
			});
		}
	};

	if (lastResult.hasVideo()) {
		appendViews(lastResult.video.frames);
	} else {
		appendViews(lastResult.images);
	}
}

//--------------------------------------------------------------
void ofxStableDiffusion::applyImageRanking(std::vector<ofxStableDiffusionImageFrame>& frames) {
	lastResult.rankingApplied = false;
	lastResult.selectedImageIndex = frames.empty() ? -1 : 0;
	if (frames.empty()) {
		return;
	}

	for (auto& frame : frames) {
		frame.isSelected = false;
	}

	if (!imageRankCallback) {
		frames.front().isSelected = true;
		return;
	}

	const std::vector<ofxStableDiffusionImageScore> scores =
		imageRankCallback(currentImageRequest, frames);
	if (scores.size() != frames.size()) {
		frames.front().isSelected = true;
		return;
	}

	for (std::size_t i = 0; i < frames.size(); ++i) {
		frames[i].score = scores[i];
	}

	const std::vector<std::size_t> order = ofxStableDiffusionBuildRankedImageOrder(scores);
	if (order.empty()) {
		frames.front().isSelected = true;
		return;
	}

	lastResult.rankingApplied = true;
	const int bestSourceIndex = static_cast<int>(order.front());
	if (imageSelectionMode == ofxStableDiffusionImageSelectionMode::Rerank ||
		imageSelectionMode == ofxStableDiffusionImageSelectionMode::BestOnly) {
		std::vector<ofxStableDiffusionImageFrame> ranked;
		ranked.reserve(frames.size());
		for (const std::size_t index : order) {
			ranked.push_back(frames[index]);
		}
		frames = std::move(ranked);
		if (imageSelectionMode == ofxStableDiffusionImageSelectionMode::BestOnly && !frames.empty()) {
			frames.resize(1);
		}
	}

	lastResult.selectedImageIndex = -1;
	for (std::size_t i = 0; i < frames.size(); ++i) {
		frames[i].index = static_cast<int>(i);
		if (frames[i].sourceIndex == bestSourceIndex && lastResult.selectedImageIndex < 0) {
			frames[i].isSelected = true;
			lastResult.selectedImageIndex = static_cast<int>(i);
		}
	}

	if (lastResult.selectedImageIndex < 0 && !frames.empty()) {
		frames.front().isSelected = true;
		lastResult.selectedImageIndex = 0;
	}
}
