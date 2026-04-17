#include "ofxStableDiffusion.h"
#include "core/ofxStableDiffusionMemoryHelpers.h"

#include <algorithm>
#include <mutex>

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
struct ValidationResult {
	ofxStableDiffusionErrorCode code = ofxStableDiffusionErrorCode::None;
	std::string message;

	bool ok() const {
		return code == ofxStableDiffusionErrorCode::None;
	}
};

static ValidationResult validateDimensions(int width, int height) {
	if (width <= 0 || height <= 0) {
		return {ofxStableDiffusionErrorCode::InvalidDimensions, "Width and height must be positive values"};
	}
	if (width > 2048 || height > 2048) {
		return {ofxStableDiffusionErrorCode::InvalidDimensions, "Width and height must not exceed 2048 pixels"};
	}
	if (width % 64 != 0) {
		return {ofxStableDiffusionErrorCode::InvalidDimensions, "Width must be a multiple of 64 (recommended: 512, 768, 1024)"};
	}
	if (height % 64 != 0) {
		return {ofxStableDiffusionErrorCode::InvalidDimensions, "Height must be a multiple of 64 (recommended: 512, 768, 1024)"};
	}
	return {};
}

static ValidationResult validateBatchCount(int batchCount) {
	if (batchCount <= 0) {
		return {ofxStableDiffusionErrorCode::InvalidBatchCount, "Batch count must be positive"};
	}
	if (batchCount > 16) {
		return {ofxStableDiffusionErrorCode::InvalidBatchCount, "Batch count exceeds maximum of 16 (risk of out-of-memory)"};
	}
	return {};
}

static ValidationResult validateSampleSteps(int sampleSteps) {
	if (sampleSteps <= 0 || sampleSteps > 200) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Sample steps must be between 1 and 200"};
	}
	return {};
}

static ValidationResult validateCfgScale(float cfgScale) {
	if (cfgScale <= 0.0f || cfgScale > 50.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "CFG scale must be greater than 0 and no more than 50"};
	}
	return {};
}

static ValidationResult validateStrength(float strength) {
	if (strength < 0.0f || strength > 1.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Strength must be between 0.0 and 1.0"};
	}
	return {};
}

static ValidationResult validateClipSkip(int clipSkip) {
	if (clipSkip < -1 || clipSkip > 12) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Clip skip must be -1 (auto) or between 0 and 12"};
	}
	return {};
}

static ValidationResult validateSeed(int64_t seed) {
	if (seed < -1) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Seed must be -1 for randomization or a non-negative value"};
	}
	return {};
}

static ValidationResult validateControlStrength(float controlStrength) {
	if (controlStrength < 0.0f || controlStrength > 2.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Control strength must be between 0.0 and 2.0"};
	}
	return {};
}

static ValidationResult validateStyleStrength(float styleStrength) {
	if (styleStrength < 0.0f || styleStrength > 100.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Style strength must be between 0 and 100"};
	}
	return {};
}

static ValidationResult validateMaskBlur(float maskBlur) {
	if (maskBlur < 0.0f || maskBlur > 256.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Mask blur must be between 0 and 256 pixels"};
	}
	return {};
}

static ValidationResult validateImageRequestNumbers(const ofxStableDiffusionImageRequest& request) {
	const ValidationResult dimResult = validateDimensions(request.width, request.height);
	if (!dimResult.ok()) return dimResult;

	const ValidationResult batchResult = validateBatchCount(request.batchCount);
	if (!batchResult.ok()) return batchResult;

	const ValidationResult stepsResult = validateSampleSteps(request.sampleSteps);
	if (!stepsResult.ok()) return stepsResult;

	const ValidationResult cfgResult = validateCfgScale(request.cfgScale);
	if (!cfgResult.ok()) return cfgResult;

	const ValidationResult strengthResult = validateStrength(request.strength);
	if (!strengthResult.ok()) return strengthResult;

	const ValidationResult clipResult = validateClipSkip(request.clipSkip);
	if (!clipResult.ok()) return clipResult;

	const ValidationResult seedResult = validateSeed(request.seed);
	if (!seedResult.ok()) return seedResult;

	const ValidationResult controlResult = validateControlStrength(request.controlStrength);
	if (!controlResult.ok()) return controlResult;

	const ValidationResult styleResult = validateStyleStrength(request.styleStrength);
	if (!styleResult.ok()) return styleResult;

	return validateMaskBlur(request.maskBlur);
}

static ValidationResult validateVideoRequestNumbers(const ofxStableDiffusionVideoRequest& request) {
	const ValidationResult dimResult = validateDimensions(request.width, request.height);
	if (!dimResult.ok()) return dimResult;

	if (request.frameCount <= 0 || request.frameCount > 100) {
		return {ofxStableDiffusionErrorCode::InvalidFrameCount, "Frame count must be between 1 and 100"};
	}

	if (request.fps <= 0 || request.fps > 120) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "FPS must be between 1 and 120"};
	}

	if (request.motionBucketId < 0 || request.motionBucketId > 255) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Motion bucket ID must be between 0 and 255"};
	}

	if (request.augmentationLevel < 0.0f || request.augmentationLevel > 2.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Augmentation level must be between 0.0 and 2.0"};
	}

	if (request.minCfg <= 0.0f || request.minCfg > 50.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Minimum CFG must be greater than 0 and no more than 50"};
	}

	const ValidationResult cfgResult = validateCfgScale(request.cfgScale);
	if (!cfgResult.ok()) return cfgResult;

	const ValidationResult stepsResult = validateSampleSteps(request.sampleSteps);
	if (!stepsResult.ok()) return stepsResult;

	const ValidationResult strengthResult = validateStrength(request.strength);
	if (!strengthResult.ok()) return strengthResult;

	return validateSeed(request.seed);
}

static ValidationResult validateUpscalerSettings(const ofxStableDiffusionUpscalerSettings& settings) {
	if (settings.nThreads == 0 || settings.nThreads < -1) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Upscaler thread count must be -1 (auto) or a positive value"};
	}
	if (settings.multiplier < 1 || settings.multiplier > 8) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Upscale multiplier must be between 1 and 8"};
	}
	return {};
}

//--------------------------------------------------------------
void ofxStableDiffusion::configureContext(const ofxStableDiffusionContextSettings& settings) {
	if (settings.nThreads == 0 || settings.nThreads < -1) {
		activeTask = ofxStableDiffusionTask::LoadModel;
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, "Thread count must be -1 (auto) or a positive value");
		return;
	}

	applyContextSettings(settings);
}

//--------------------------------------------------------------
void ofxStableDiffusion::generate(const ofxStableDiffusionImageRequest& request) {
	const ofxStableDiffusionTask task = ofxStableDiffusionTaskForImageMode(request.mode);
	if (!validateImageRequestAndSetError(request, task)) {
		return;
	}

	if (!beginBackgroundTask(task)) {
		return;
	}
	applyImageRequest(request);
	thread.startThread();
}

//--------------------------------------------------------------
void ofxStableDiffusion::generateVideo(const ofxStableDiffusionVideoRequest& request) {
	if (!validateVideoRequestAndSetError(request)) {
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
	const ValidationResult validation = validateUpscalerSettings(settings);
	if (!validation.ok()) {
		activeTask = ofxStableDiffusionTask::Upscale;
		setLastError(validation.code, validation.message);
		return;
	}

	std::lock_guard<std::mutex> lock(stateMutex);
	esrganPath = settings.modelPath;
	nThreads = settings.nThreads;
	wType = settings.weightType;
	esrganMultiplier = settings.multiplier;
	isESRGAN = settings.enabled;
}

//--------------------------------------------------------------
ofxStableDiffusionContextSettings ofxStableDiffusion::getContextSettings() const {
	std::lock_guard<std::mutex> lock(stateMutex);
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
	std::lock_guard<std::mutex> lock(stateMutex);
	return {esrganPath, nThreads, wType, esrganMultiplier, isESRGAN};
}

//--------------------------------------------------------------
ofxStableDiffusionResult ofxStableDiffusion::getLastResult() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult;
}

//--------------------------------------------------------------
std::vector<ofxStableDiffusionImageFrame> ofxStableDiffusion::getImages() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.images;
}

//--------------------------------------------------------------
ofxStableDiffusionVideoClip ofxStableDiffusion::getVideoClip() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.video;
}

//--------------------------------------------------------------
bool ofxStableDiffusion::hasImageResult() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.hasImages();
}

//--------------------------------------------------------------
bool ofxStableDiffusion::hasVideoResult() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.hasVideo();
}

//--------------------------------------------------------------
int ofxStableDiffusion::getOutputCount() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (lastResult.hasVideo()) {
		return static_cast<int>(lastResult.video.frames.size());
	}
	return static_cast<int>(lastResult.images.size());
}

//--------------------------------------------------------------
std::string ofxStableDiffusion::getLastError() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastError;
}

//--------------------------------------------------------------
ofxStableDiffusionErrorCode ofxStableDiffusion::getLastErrorCode() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastErrorInfo.code;
}

//--------------------------------------------------------------
ofxStableDiffusionError ofxStableDiffusion::getLastErrorInfo() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastErrorInfo;
}

//--------------------------------------------------------------
std::vector<ofxStableDiffusionError> ofxStableDiffusion::getErrorHistory() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return errorHistory;
}

//--------------------------------------------------------------
void ofxStableDiffusion::clearErrorHistory() {
	std::lock_guard<std::mutex> lock(stateMutex);
	errorHistory.clear();
}

//--------------------------------------------------------------
int ofxStableDiffusion::getVideoFrameIndexForTime(float seconds) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.video.frameIndexForTime(seconds);
}

//--------------------------------------------------------------
const ofPixels* ofxStableDiffusion::getVideoFramePixels(int index) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (index < 0 || index >= static_cast<int>(lastResult.video.frames.size())) {
		return nullptr;
	}
	return &lastResult.video.frames[static_cast<std::size_t>(index)].pixels;
}

//--------------------------------------------------------------
bool ofxStableDiffusion::saveVideoFrames(const std::string& directory, const std::string& prefix) const {
	return getVideoClip().saveFrameSequence(directory, prefix);
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
	std::lock_guard<std::mutex> lock(stateMutex);
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
	std::lock_guard<std::mutex> lock(stateMutex);
	return diffused;
}

sd_image_t* ofxStableDiffusion::returnImages() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return outputImages;
}

void ofxStableDiffusion::setDiffused(bool diffused_) {
	std::lock_guard<std::mutex> lock(stateMutex);
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
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.actualSeedUsed;
}

//--------------------------------------------------------------
std::vector<int64_t> ofxStableDiffusion::getSeedHistory() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return seedHistory;
}

//--------------------------------------------------------------
void ofxStableDiffusion::clearSeedHistory() {
	std::lock_guard<std::mutex> lock(stateMutex);
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
bool ofxStableDiffusion::validateImageRequestAndSetError(const ofxStableDiffusionImageRequest& request, ofxStableDiffusionTask task) {
	const ValidationResult validation = validateImageRequestNumbers(request);
	imageMode = request.mode;
	activeTask = task;

	if (!validation.ok()) {
		setLastError(validation.code, validation.message);
		return false;
	}

	const sd_image_t candidateInputImage =
		request.initImage.data != nullptr ? request.initImage : inputImage;
	if (ofxStableDiffusionImageModeUsesInputImage(request.mode) && candidateInputImage.data == nullptr) {
		setLastError(ofxStableDiffusionErrorCode::MissingInputImage, "Selected image mode requires an input image");
		return false;
	}

	return true;
}

//--------------------------------------------------------------
bool ofxStableDiffusion::validateVideoRequestAndSetError(const ofxStableDiffusionVideoRequest& request) {
	const ValidationResult validation = validateVideoRequestNumbers(request);
	activeTask = ofxStableDiffusionTask::ImageToVideo;
	if (!validation.ok()) {
		setLastError(validation.code, validation.message);
		return false;
	}
	return true;
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
	if (nThreads_ == 0 || nThreads_ < -1) {
		activeTask = ofxStableDiffusionTask::LoadModel;
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, "Thread count must be -1 (auto) or a positive value");
		return;
	}

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
	ofxStableDiffusionImageRequest request;
	request.mode = ofxStableDiffusionImageMode::TextToImage;
	request.selectionMode = imageSelectionMode;
	request.initImage = {0, 0, 0, nullptr};
	request.maskImage = {0, 0, 0, nullptr};
	request.prompt = prompt_;
	request.instruction.clear();
	request.negativePrompt = negativePrompt_;
	request.clipSkip = clipSkip_;
	request.cfgScale = cfgScale_;
	request.width = width_;
	request.height = height_;
	request.sampleMethod = sampleMethod_;
	request.sampleSteps = sampleSteps_;
	request.strength = 0.5f;
	request.seed = seed_;
	request.batchCount = batchCount_;
	request.controlCond = controlCond_;
	request.controlStrength = controlStrength_;
	request.styleStrength = styleStrength_;
	request.normalizeInput = normalizeInput_;
	request.inputIdImagesPath = inputIdImagesPath_;

	if (!validateImageRequestAndSetError(request, ofxStableDiffusionTask::TextToImage)) {
		return;
	}

	if (!beginBackgroundTask(ofxStableDiffusionTask::TextToImage)) {
		return;
	}

	applyImageRequest(request);
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
	ofxStableDiffusionImageRequest request;
	request.mode = ofxStableDiffusionImageMode::ImageToImage;
	request.selectionMode = imageSelectionMode;
	request.initImage = initImage_;
	request.maskImage = {0, 0, 0, nullptr};
	request.prompt = prompt_;
	request.instruction.clear();
	request.negativePrompt = negativePrompt_;
	request.clipSkip = clipSkip_;
	request.cfgScale = cfgScale_;
	request.width = width_;
	request.height = height_;
	request.sampleMethod = sampleMethod_;
	request.sampleSteps = sampleSteps_;
	request.strength = strength_;
	request.seed = seed_;
	request.batchCount = batchCount_;
	request.controlCond = controlCond_;
	request.controlStrength = controlStrength_;
	request.styleStrength = styleStrength_;
	request.normalizeInput = normalizeInput_;
	request.inputIdImagesPath = inputIdImagesPath_;

	if (!validateImageRequestAndSetError(request, ofxStableDiffusionTask::ImageToImage)) {
		return;
	}

	if (!beginBackgroundTask(ofxStableDiffusionTask::ImageToImage)) {
		return;
	}

	inputImage = initImage_;
	applyImageRequest(request);
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
	ofxStableDiffusionImageRequest request;
	request.mode = ofxStableDiffusionImageMode::InstructImage;
	request.selectionMode = imageSelectionMode;
	request.initImage = initImage_;
	request.maskImage = {0, 0, 0, nullptr};
	request.prompt = instruction_;
	request.instruction = instruction_;
	request.negativePrompt = negativePrompt_;
	request.clipSkip = clipSkip_;
	request.cfgScale = cfgScale_;
	request.width = width_;
	request.height = height_;
	request.sampleMethod = sampleMethod_;
	request.sampleSteps = sampleSteps_;
	request.strength = strength_;
	request.seed = seed_;
	request.batchCount = batchCount_;
	request.controlCond = controlCond_;
	request.controlStrength = controlStrength_;
	request.styleStrength = styleStrength;
	request.normalizeInput = normalizeInput_;

	if (!validateImageRequestAndSetError(request, ofxStableDiffusionTask::InstructImage)) {
		return;
	}

	if (!beginBackgroundTask(ofxStableDiffusionTask::InstructImage)) {
		return;
	}

	applyImageRequest(request);
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
	ofxStableDiffusionVideoRequest request;
	request.initImage = initImage_;
	request.width = width_;
	request.height = height_;
	request.frameCount = videoFrames_;
	request.motionBucketId = motionBucketId_;
	request.fps = fps_;
	request.augmentationLevel = augmentationLevel_;
	request.minCfg = minCfg_;
	request.cfgScale = cfgScale_;
	request.sampleMethod = sampleMethod_;
	request.sampleSteps = sampleSteps_;
	request.strength = strength_;
	request.seed = seed_;
	request.mode = videoMode;

	if (!validateVideoRequestAndSetError(request)) {
		return;
	}

	if (!beginBackgroundTask(ofxStableDiffusionTask::ImageToVideo)) {
		return;
	}

	applyVideoRequest(request);
	thread.startThread();
}

//--------------------------------------------------------------
void ofxStableDiffusion::newUpscalerCtx(const std::string& esrganPath_,
	int nThreads_,
	enum sd_type_t wType_) {
	const ofxStableDiffusionUpscalerSettings requested{esrganPath_, nThreads_, wType_, esrganMultiplier, true};
	const ValidationResult validation = validateUpscalerSettings(requested);
	activeTask = ofxStableDiffusionTask::Upscale;
	if (!validation.ok()) {
		setLastError(validation.code, validation.message);
		return;
	}

	if (thread.isThreadRunning()) {
		ofLogWarning("ofxStableDiffusion") << "Cannot rebuild the upscaler while a task is still running";
		return;
	}

	setUpscalerSettings(requested);
	if (thread.upscalerCtx) {
		free_upscaler_ctx(thread.upscalerCtx);
		thread.upscalerCtx = nullptr;
	}
	thread.upscalerCtx = new_upscaler_ctx(esrganPath_.c_str(), false, false, nThreads_, 0);
	if (!thread.upscalerCtx) {
		{
			std::lock_guard<std::mutex> lock(stateMutex);
			isESRGAN = false;
		}
		setLastError(ofxStableDiffusionErrorCode::UpscaleFailed, "Failed to create upscaler context");
		return;
	}

	{
		std::lock_guard<std::mutex> lock(stateMutex);
		isESRGAN = true;
	}
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
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		isESRGAN = false;
	}
}

//--------------------------------------------------------------
sd_image_t ofxStableDiffusion::upscaleImage(sd_image_t inputImage_, uint32_t upscaleFactor) {
	activeTask = ofxStableDiffusionTask::Upscale;
	if (upscaleFactor == 0) {
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, "Upscale factor must be at least 1");
		return {0, 0, 0, nullptr};
	}

	if (!thread.upscalerCtx) {
		setLastError(ofxStableDiffusionErrorCode::UpscaleFailed, "Upscaler context is not initialized");
		return {0, 0, 0, nullptr};
	}

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
	std::lock_guard<std::mutex> lock(stateMutex);
	outputImages = nullptr;
	outputImageViews.clear();
	lastResult = {};
	diffused = false;
}

//--------------------------------------------------------------
void ofxStableDiffusion::setLastError(const std::string& errorMessage, ofxStableDiffusionErrorCode code) {
	std::lock_guard<std::mutex> lock(stateMutex);
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

	lastResult = {};
	lastResult.success = false;
	lastResult.task = activeTask;
	lastResult.imageMode = imageMode;
	lastResult.selectionMode = imageSelectionMode;
	lastResult.error = errorMessage;
	outputImageViews.clear();
	outputImages = nullptr;
	diffused = false;
}

//--------------------------------------------------------------
void ofxStableDiffusion::setLastError(ofxStableDiffusionErrorCode code, const std::string& errorMessage) {
	setLastError(errorMessage, code);
}

//--------------------------------------------------------------
void ofxStableDiffusion::clearLastError() {
	std::lock_guard<std::mutex> lock(stateMutex);
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
	ofxStableDiffusionResult result;
	result.success = true;
	result.task = activeTask;
	result.imageMode = imageMode;
	result.selectionMode = imageSelectionMode;
	result.elapsedMs = elapsedMs;
	result.actualSeedUsed = seedValue;
	result.images.reserve(std::max(0, count));

	for (int i = 0; i < count; ++i) {
		ofxStableDiffusionImageFrame frame;
		frame.index = i;
		frame.sourceIndex = i;
		frame.seed = seedValue;
		frame.pixels = makePixelsCopy(images[i]);
		result.images.push_back(std::move(frame));
	}

	applyImageRanking(result.images, result);

	{
		std::lock_guard<std::mutex> lock(stateMutex);
		seedHistory.push_back(seedValue);
		if (seedHistory.size() > maxSeedHistorySize) {
			seedHistory.erase(seedHistory.begin());
		}

		lastResult = std::move(result);
		outputImageViews = buildOutputImageViews(lastResult);
		outputImages = outputImageViews.empty() ? nullptr : outputImageViews.data();
		diffused = true;
	}

	ofxSdReleaseImageArray(images, count);
}

//--------------------------------------------------------------
void ofxStableDiffusion::captureVideoResults(sd_image_t* images, int count, int seedValue, float elapsedMs) {
	ofxStableDiffusionResult result;
	result.success = true;
	result.task = activeTask;
	result.elapsedMs = elapsedMs;
	result.actualSeedUsed = seedValue;
	result.video.fps = fps;
	result.video.sourceFrameCount = count;
	result.video.mode = videoMode;

	std::vector<ofxStableDiffusionImageFrame> sourceFrames;
	sourceFrames.reserve(std::max(0, count));
	for (int i = 0; i < count; ++i) {
		ofxStableDiffusionImageFrame frame;
		frame.index = i;
		frame.seed = seedValue;
		frame.pixels = makePixelsCopy(images[i]);
		sourceFrames.push_back(std::move(frame));
	}

	result.video.frames = ofxStableDiffusionBuildVideoFrames(sourceFrames, videoMode);

	{
		std::lock_guard<std::mutex> lock(stateMutex);
		seedHistory.push_back(seedValue);
		if (seedHistory.size() > maxSeedHistorySize) {
			seedHistory.erase(seedHistory.begin());
		}

		lastResult = std::move(result);
		outputImageViews = buildOutputImageViews(lastResult);
		outputImages = outputImageViews.empty() ? nullptr : outputImageViews.data();
		diffused = true;
	}

	ofxSdReleaseImageArray(images, count);
}

//--------------------------------------------------------------
std::vector<sd_image_t> ofxStableDiffusion::buildOutputImageViews(const ofxStableDiffusionResult& result) const {
	std::vector<sd_image_t> views;
	const auto appendViews = [&views](const std::vector<ofxStableDiffusionImageFrame>& frames) {
		views.reserve(frames.size());
		for (const auto& frame : frames) {
			if (!frame.isAllocated()) {
				continue;
			}
			views.push_back({
				static_cast<uint32_t>(frame.pixels.getWidth()),
				static_cast<uint32_t>(frame.pixels.getHeight()),
				static_cast<uint32_t>(frame.pixels.getNumChannels()),
				const_cast<unsigned char*>(frame.pixels.getData())
			});
		}
	};

	if (result.hasVideo()) {
		appendViews(result.video.frames);
	} else {
		appendViews(result.images);
	}

	return views;
}

//--------------------------------------------------------------
void ofxStableDiffusion::applyImageRanking(std::vector<ofxStableDiffusionImageFrame>& frames, ofxStableDiffusionResult& result) {
	result.rankingApplied = false;
	result.selectedImageIndex = frames.empty() ? -1 : 0;
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

	result.rankingApplied = true;
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

	result.selectedImageIndex = -1;
	for (std::size_t i = 0; i < frames.size(); ++i) {
		frames[i].index = static_cast<int>(i);
		if (frames[i].sourceIndex == bestSourceIndex && result.selectedImageIndex < 0) {
			frames[i].isSelected = true;
			result.selectedImageIndex = static_cast<int>(i);
		}
	}

	if (result.selectedImageIndex < 0 && !frames.empty()) {
		frames.front().isSelected = true;
		result.selectedImageIndex = 0;
	}
}
