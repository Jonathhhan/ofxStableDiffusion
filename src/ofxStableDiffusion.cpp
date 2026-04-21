#include "ofxStableDiffusion.h"
#include "core/ofxStableDiffusionCapabilityHelpers.h"
#include "core/ofxStableDiffusionMemoryHelpers.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <mutex>

namespace {

namespace fs = std::filesystem;

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
	(void)data;
	if (level <= SD_LOG_INFO) {
		fputs(log, stdout);
		fflush(stdout);
	} else {
		fputs(log, stderr);
		fflush(stderr);
	}
}

struct ValidationResult {
	ofxStableDiffusionErrorCode code = ofxStableDiffusionErrorCode::None;
	std::string message;

	bool ok() const {
		return code == ofxStableDiffusionErrorCode::None;
	}
};

ValidationResult validateDimensions(int width, int height) {
	if (width <= 0 || height <= 0) {
		return {ofxStableDiffusionErrorCode::InvalidDimensions, "Width and height must be positive values"};
	}
	if (width > 2048 || height > 2048) {
		return {ofxStableDiffusionErrorCode::InvalidDimensions, "Width and height must not exceed 2048 pixels"};
	}
	if (width % 64 != 0 || height % 64 != 0) {
		return {ofxStableDiffusionErrorCode::InvalidDimensions, "Width and height must both be multiples of 64"};
	}
	return {};
}

ValidationResult validateBatchCount(int batchCount) {
	if (batchCount <= 0) {
		return {ofxStableDiffusionErrorCode::InvalidBatchCount, "Batch count must be positive"};
	}
	if (batchCount > 16) {
		return {ofxStableDiffusionErrorCode::InvalidBatchCount, "Batch count exceeds maximum of 16"};
	}
	return {};
}

ValidationResult validateSampleSteps(int sampleSteps) {
	if (sampleSteps <= 0 || sampleSteps > 200) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Sample steps must be between 1 and 200"};
	}
	return {};
}

ValidationResult validateCfgScale(float cfgScale) {
	if (cfgScale <= 0.0f || cfgScale > 50.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "CFG scale must be greater than 0 and no more than 50"};
	}
	return {};
}

ValidationResult validateStrength(float strength) {
	if (strength < 0.0f || strength > 1.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Strength must be between 0.0 and 1.0"};
	}
	return {};
}

ValidationResult validateClipSkip(int clipSkip) {
	if (clipSkip < -1 || clipSkip > 12) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Clip skip must be -1 (auto) or between 0 and 12"};
	}
	return {};
}

ValidationResult validateSeed(int64_t seed) {
	if (seed < -1) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Seed must be -1 for randomization or a non-negative value"};
	}
	return {};
}

ValidationResult validateControlStrength(float controlStrength) {
	if (controlStrength < 0.0f || controlStrength > 2.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Control strength must be between 0.0 and 2.0"};
	}
	return {};
}

ValidationResult validateStyleStrength(float styleStrength) {
	if (styleStrength < 0.0f || styleStrength > 100.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Style strength must be between 0 and 100"};
	}
	return {};
}

ValidationResult validateVaceStrength(float vaceStrength) {
	if (vaceStrength < 0.0f || vaceStrength > 1.0f) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "VACE strength must be between 0.0 and 1.0"};
	}
	return {};
}

ValidationResult validateImageRequestNumbers(const ofxStableDiffusionImageRequest& request) {
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

	return {};
}

ValidationResult validateVideoRequestNumbers(const ofxStableDiffusionVideoRequest& request) {
	const ValidationResult dimResult = validateDimensions(request.width, request.height);
	if (!dimResult.ok()) return dimResult;

	if (request.frameCount <= 0 || request.frameCount > 100) {
		return {ofxStableDiffusionErrorCode::InvalidFrameCount, "Frame count must be between 1 and 100"};
	}

	if (request.fps <= 0 || request.fps > 120) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "FPS must be between 1 and 120"};
	}

	const ValidationResult clipResult = validateClipSkip(request.clipSkip);
	if (!clipResult.ok()) return clipResult;

	const ValidationResult cfgResult = validateCfgScale(request.cfgScale);
	if (!cfgResult.ok()) return cfgResult;

	const ValidationResult stepsResult = validateSampleSteps(request.sampleSteps);
	if (!stepsResult.ok()) return stepsResult;

	const ValidationResult strengthResult = validateStrength(request.strength);
	if (!strengthResult.ok()) return strengthResult;

	const ValidationResult seedResult = validateSeed(request.seed);
	if (!seedResult.ok()) return seedResult;

	return validateVaceStrength(request.vaceStrength);
}

ValidationResult validateUpscalerSettings(const ofxStableDiffusionUpscalerSettings& settings) {
	if (settings.nThreads == 0 || settings.nThreads < -1) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Upscaler thread count must be -1 (auto) or a positive value"};
	}
	if (settings.multiplier < 1 || settings.multiplier > 8) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Upscale multiplier must be between 1 and 8"};
	}
	return {};
}

bool isResolvableModelFile(const fs::path& path) {
	if (!fs::is_regular_file(path)) {
		return false;
	}

	std::string ext = ofToLower(path.extension().string());
	return ext == ".gguf" ||
		ext == ".safetensors" ||
		ext == ".ckpt" ||
		ext == ".bin" ||
		ext == ".pth";
}

void appendUniqueSearchRoot(std::vector<fs::path>& roots, const fs::path& root) {
	if (root.empty()) {
		return;
	}

	const fs::path normalized = root.lexically_normal();
	if (std::find(roots.begin(), roots.end(), normalized) == roots.end()) {
		roots.push_back(normalized);
	}
}

void appendSearchRootsForModelPath(std::vector<fs::path>& roots, const std::string& modelPath) {
	if (modelPath.empty()) {
		return;
	}

	const fs::path path(modelPath);
	const fs::path dir = path.has_parent_path() ? path.parent_path() : fs::path();
	if (dir.empty()) {
		return;
	}

	appendUniqueSearchRoot(roots, dir);
	if (dir.has_parent_path()) {
		appendUniqueSearchRoot(roots, dir.parent_path());
	}
}

std::string resolveTextEncoderPathFromSubfolders(const ofxStableDiffusionContextSettings& settings) {
	std::vector<fs::path> roots;
	appendSearchRootsForModelPath(roots, settings.diffusionModelPath);
	appendSearchRootsForModelPath(roots, settings.modelPath);
	if (roots.empty()) {
		return "";
	}

	const std::vector<std::string> subfolders = {"umt5", "t5xxl", "text_encoders"};
	const std::vector<std::string> preferredNameParts = {"umt5", "t5xxl", "encoder"};

	for (const auto& root : roots) {
		for (const auto& subfolder : subfolders) {
			const fs::path candidateDir = root / subfolder;
			if (!fs::exists(candidateDir) || !fs::is_directory(candidateDir)) {
				continue;
			}

			std::vector<fs::path> preferredFiles;
			std::vector<fs::path> fallbackFiles;
			for (const auto& entry : fs::directory_iterator(candidateDir)) {
				const fs::path candidatePath = entry.path();
				if (!isResolvableModelFile(candidatePath)) {
					continue;
				}

				const std::string filename = ofToLower(candidatePath.filename().string());
				const bool preferred = std::any_of(
					preferredNameParts.begin(),
					preferredNameParts.end(),
					[&filename](const std::string& part) {
						return filename.find(part) != std::string::npos;
					});

				if (preferred) {
					preferredFiles.push_back(candidatePath);
				} else {
					fallbackFiles.push_back(candidatePath);
				}
			}

			auto byFilename = [](const fs::path& a, const fs::path& b) {
				return ofToLower(a.filename().string()) < ofToLower(b.filename().string());
			};
			std::sort(preferredFiles.begin(), preferredFiles.end(), byFilename);
			std::sort(fallbackFiles.begin(), fallbackFiles.end(), byFilename);

			if (!preferredFiles.empty()) {
				return preferredFiles.front().string();
			}
			if (!fallbackFiles.empty()) {
				return fallbackFiles.front().string();
			}
		}
	}

	return "";
}

ofxStableDiffusionContextSettings resolveContextModelPaths(
	const ofxStableDiffusionContextSettings& requestedSettings) {
	ofxStableDiffusionContextSettings resolvedSettings = requestedSettings;

	if (resolvedSettings.t5xxlPath.empty()) {
		resolvedSettings.t5xxlPath = resolveTextEncoderPathFromSubfolders(resolvedSettings);
	}

	return resolvedSettings;
}

} // namespace

ofxStableDiffusion::ofxStableDiffusion() {
	sd_set_log_callback(sd_log_cb, nullptr);
	thread.userData = this;
}

ofxStableDiffusion::~ofxStableDiffusion() {
	if (thread.isThreadRunning()) {
		thread.waitForThread(true);
	}
	thread.clearContexts();
}

void ofxStableDiffusion::configureContext(const ofxStableDiffusionContextSettings& settings) {
	if (settings.nThreads == 0 || settings.nThreads < -1) {
		activeTask = ofxStableDiffusionTask::LoadModel;
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, "Thread count must be -1 (auto) or a positive value");
		return;
	}
	applyContextSettings(settings);
}

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

ofxStableDiffusionContextSettings ofxStableDiffusion::getContextSettings() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return captureContextSettingsNoLock();
}

ofxStableDiffusionUpscalerSettings ofxStableDiffusion::getUpscalerSettings() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return captureUpscalerSettingsNoLock();
}

ofxStableDiffusionCapabilities ofxStableDiffusion::getCapabilities() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return ofxStableDiffusionCapabilityHelpers::resolveCapabilities(
		captureContextSettingsNoLock(),
		captureUpscalerSettingsNoLock());
}

ofxStableDiffusionResult ofxStableDiffusion::getLastResult() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult;
}

std::vector<ofxStableDiffusionImageFrame> ofxStableDiffusion::getImages() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.images;
}

ofxStableDiffusionVideoClip ofxStableDiffusion::getVideoClip() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.video;
}

bool ofxStableDiffusion::hasImageResult() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.hasImages();
}

bool ofxStableDiffusion::hasVideoResult() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.hasVideo();
}

int ofxStableDiffusion::getOutputCount() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (lastResult.hasVideo()) {
		return static_cast<int>(lastResult.video.frames.size());
	}
	return static_cast<int>(lastResult.images.size());
}

std::string ofxStableDiffusion::getLastError() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastError;
}

ofxStableDiffusionErrorCode ofxStableDiffusion::getLastErrorCode() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastErrorInfo.code;
}

ofxStableDiffusionError ofxStableDiffusion::getLastErrorInfo() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastErrorInfo;
}

std::vector<ofxStableDiffusionError> ofxStableDiffusion::getErrorHistory() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return std::vector<ofxStableDiffusionError>(errorHistory.begin(), errorHistory.end());
}

void ofxStableDiffusion::clearErrorHistory() {
	std::lock_guard<std::mutex> lock(stateMutex);
	errorHistory.clear();
}

int ofxStableDiffusion::getVideoFrameIndexForTime(float seconds) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.video.frameIndexForTime(seconds);
}

const ofPixels* ofxStableDiffusion::getVideoFramePixels(int index) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (index < 0 || index >= static_cast<int>(lastResult.video.frames.size())) {
		return nullptr;
	}
	return &lastResult.video.frames[static_cast<std::size_t>(index)].pixels;
}

bool ofxStableDiffusion::saveVideoFrames(const std::string& directory, const std::string& prefix) const {
	return getVideoClip().saveFrameSequence(directory, prefix);
}

bool ofxStableDiffusion::saveVideoMetadata(const std::string& path) const {
	return getVideoClip().saveMetadataJson(path);
}

bool ofxStableDiffusion::saveVideoFramesWithMetadata(
	const std::string& directory,
	const std::string& prefix,
	const std::string& metadataFilename) const {
	return getVideoClip().saveFrameSequenceWithMetadata(directory, prefix, metadataFilename);
}

bool ofxStableDiffusion::saveVideoWebm(const std::string& path, int quality) const {
	return getVideoClip().saveWebm(path, quality);
}

void ofxStableDiffusion::setVideoGenerationMode(ofxStableDiffusionVideoMode mode) {
	std::lock_guard<std::mutex> lock(stateMutex);
	videoMode = mode;
}

ofxStableDiffusionVideoMode ofxStableDiffusion::getVideoGenerationMode() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return videoMode;
}

void ofxStableDiffusion::setImageGenerationMode(ofxStableDiffusionImageMode mode) {
	std::lock_guard<std::mutex> lock(stateMutex);
	imageMode = mode;
	isTextToImage = (mode == ofxStableDiffusionImageMode::TextToImage);
	isImageToVideo = false;
}

ofxStableDiffusionImageMode ofxStableDiffusion::getImageGenerationMode() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return imageMode;
}

void ofxStableDiffusion::setImageSelectionMode(ofxStableDiffusionImageSelectionMode mode) {
	std::lock_guard<std::mutex> lock(stateMutex);
	imageSelectionMode = mode;
}

ofxStableDiffusionImageSelectionMode ofxStableDiffusion::getImageSelectionMode() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return imageSelectionMode;
}

void ofxStableDiffusion::setImageRankCallback(ofxSdImageRankCallback cb) {
	std::lock_guard<std::mutex> lock(stateMutex);
	imageRankCallback = cb;
}

int ofxStableDiffusion::getSelectedImageIndex() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.selectedImageIndex;
}

void ofxStableDiffusion::loadImage(const ofPixels& pixels) {
	std::lock_guard<std::mutex> lock(stateMutex);
	loadedInputImage.assign({
		static_cast<uint32_t>(pixels.getWidth()),
		static_cast<uint32_t>(pixels.getHeight()),
		static_cast<uint32_t>(pixels.getNumChannels()),
		const_cast<unsigned char*>(pixels.getData())
	});
	inputImage = loadedInputImage.image;
}

void ofxStableDiffusion::setLoras(const std::vector<ofxStableDiffusionLora>& loras_) {
	std::lock_guard<std::mutex> lock(stateMutex);
	loras = loras_;
}

std::vector<ofxStableDiffusionLora> ofxStableDiffusion::getLoras() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return loras;
}

std::vector<std::pair<std::string, std::string>> ofxStableDiffusion::listLoras() const {
	std::vector<std::pair<std::string, std::string>> results;
	const std::string targetDir = loraModelDir;
	if (targetDir.empty()) {
		return results;
	}

	ofDirectory dir(targetDir);
	if (!dir.exists()) {
		return results;
	}
	dir.allowExt("safetensors");
	dir.allowExt("ckpt");
	dir.allowExt("pt");
	dir.allowExt("bin");
	dir.listDir();

	for (std::size_t i = 0; i < dir.size(); ++i) {
		const ofFile& file = dir.getFile(static_cast<int>(i));
		if (file.isFile()) {
			results.emplace_back(file.getBaseName(), file.getAbsolutePath());
		}
	}
	return results;
}

void ofxStableDiffusion::reloadEmbeddings(const std::string& embedDir) {
	ofxStableDiffusionContextSettings settings = getContextSettings();
	if (!embedDir.empty()) {
		settings.embedDir = embedDir;
	}
	newSdCtx(settings);
}

std::vector<std::pair<std::string, std::string>> ofxStableDiffusion::listEmbeddings() const {
	std::vector<std::pair<std::string, std::string>> results;
	const std::string targetDir = embedDirCStr;
	if (targetDir.empty()) {
		return results;
	}

	ofDirectory dir(targetDir);
	if (!dir.exists()) {
		return results;
	}
	dir.allowExt("pt");
	dir.allowExt("ckpt");
	dir.allowExt("safetensors");
	dir.allowExt("bin");
	dir.allowExt("gguf");
	dir.listDir();

	for (std::size_t i = 0; i < dir.size(); ++i) {
		const ofFile& file = dir.getFile(static_cast<int>(i));
		if (file.isFile()) {
			results.emplace_back(file.getBaseName(), file.getAbsolutePath());
		}
	}
	return results;
}

bool ofxStableDiffusion::isDiffused() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return diffused;
}

void ofxStableDiffusion::setDiffused(bool diffused_) {
	std::lock_guard<std::mutex> lock(stateMutex);
	diffused = diffused_;
}

sd_image_t* ofxStableDiffusion::returnImages() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return outputImages;
}

const char* ofxStableDiffusion::typeName(enum sd_type_t type) {
	return sd_type_name(type);
}

int32_t ofxStableDiffusion::getNumPhysicalCores() {
	return sd_get_num_physical_cores();
}

const char* ofxStableDiffusion::getSystemInfo() {
	return sd_get_system_info();
}

void ofxStableDiffusion::setProgressCallback(ofxSdProgressCallback cb) {
	std::lock_guard<std::mutex> lock(stateMutex);
	progressCallback = cb;
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
		setLastError(ofxStableDiffusionErrorCode::ThreadBusy, "A task is already running");
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
void ofxStableDiffusion::newSdCtx(const ofxStableDiffusionContextSettings& settings) {
	if (settings.nThreads == 0 || settings.nThreads < -1) {
		activeTask = ofxStableDiffusionTask::LoadModel;
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, "Thread count must be -1 (auto) or a positive value");
		return;
	}

	if (!beginBackgroundTask(ofxStableDiffusionTask::LoadModel)) {
		return;
	}

	applyContextSettings(settings);
	stableDiffusionThread::ContextTaskData taskData;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		taskData.contextSettings = captureContextSettingsNoLock();
		taskData.upscalerSettings = captureUpscalerSettingsNoLock();
	}
	thread.prepareContextTask(taskData);
	thread.startThread();
}

void ofxStableDiffusion::freeSdCtx() {
	if (thread.isThreadRunning()) {
		thread.waitForThread(true);
	}
	thread.clearContexts();
}

void ofxStableDiffusion::txt2img(const std::string& prompt_,
	const std::string& negativePrompt_,
	int clipSkip_,
	float cfgScale_,
	int width_,
	int height_,
	sample_method_t sampleMethod_,
	int sampleSteps_,
	int64_t seed_,
	int batchCount_,
	sd_image_t* controlCond_,
	float controlStrength_,
	float styleStrength_,
	bool normalizeInput_,
	const std::string& inputIdImagesPath_) {
	if (thread.isThreadRunning()) {
		setLastError(ofxStableDiffusionErrorCode::ThreadBusy, "A task is already running");
		return;
	}

	const ValidationResult dimResult = validateDimensions(width_, height_);
	if (!dimResult.ok()) {
		imageMode = ofxStableDiffusionImageMode::TextToImage;
		activeTask = ofxStableDiffusionTask::TextToImage;
		setLastError(dimResult.code, dimResult.message);
		return;
	}
	const ValidationResult batchResult = validateBatchCount(batchCount_);
	if (!batchResult.ok()) {
		imageMode = ofxStableDiffusionImageMode::TextToImage;
		activeTask = ofxStableDiffusionTask::TextToImage;
		setLastError(batchResult.code, batchResult.message);
		return;
	}

	ofxStableDiffusionImageRequest request;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		request.selectionMode = imageSelectionMode;
		request.loras = loras;
	}
	request.mode = ofxStableDiffusionImageMode::TextToImage;
	request.prompt = prompt_;
	request.negativePrompt = negativePrompt_;
	request.clipSkip = clipSkip_;
	request.cfgScale = cfgScale_;
	request.width = width_;
	request.height = height_;
	request.sampleMethod = sampleMethod_;
	request.sampleSteps = sampleSteps_;
	request.seed = seed_;
	request.batchCount = batchCount_;
	request.controlCond = controlCond_;
	request.controlStrength = controlStrength_;
	request.styleStrength = styleStrength_;
	request.normalizeInput = normalizeInput_;
	request.inputIdImagesPath = inputIdImagesPath_;
	generate(request);
}

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
	if (thread.isThreadRunning()) {
		setLastError(ofxStableDiffusionErrorCode::ThreadBusy, "A task is already running");
		return;
	}

	if (initImage_.data == nullptr) {
		imageMode = ofxStableDiffusionImageMode::ImageToImage;
		activeTask = ofxStableDiffusionTask::ImageToImage;
		setLastError(ofxStableDiffusionErrorCode::MissingInputImage, "Image-to-image requires an input image");
		return;
	}

	const ValidationResult dimResult = validateDimensions(width_, height_);
	if (!dimResult.ok()) {
		imageMode = ofxStableDiffusionImageMode::ImageToImage;
		activeTask = ofxStableDiffusionTask::ImageToImage;
		setLastError(dimResult.code, dimResult.message);
		return;
	}
	const ValidationResult batchResult = validateBatchCount(batchCount_);
	if (!batchResult.ok()) {
		imageMode = ofxStableDiffusionImageMode::ImageToImage;
		activeTask = ofxStableDiffusionTask::ImageToImage;
		setLastError(batchResult.code, batchResult.message);
		return;
	}

	ofxStableDiffusionImageRequest request;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		request.selectionMode = imageSelectionMode;
		request.loras = loras;
	}
	request.mode = ofxStableDiffusionImageMode::ImageToImage;
	request.initImage = initImage_;
	request.prompt = prompt_;
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
	generate(request);
}

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
	if (thread.isThreadRunning()) {
		setLastError(ofxStableDiffusionErrorCode::ThreadBusy, "A task is already running");
		return;
	}

	if (initImage_.data == nullptr) {
		imageMode = ofxStableDiffusionImageMode::InstructImage;
		activeTask = ofxStableDiffusionTask::InstructImage;
		setLastError(ofxStableDiffusionErrorCode::MissingInputImage, "Instruct image requires an input image");
		return;
	}

	const ValidationResult dimResult = validateDimensions(width_, height_);
	if (!dimResult.ok()) {
		imageMode = ofxStableDiffusionImageMode::InstructImage;
		activeTask = ofxStableDiffusionTask::InstructImage;
		setLastError(dimResult.code, dimResult.message);
		return;
	}
	const ValidationResult batchResult = validateBatchCount(batchCount_);
	if (!batchResult.ok()) {
		imageMode = ofxStableDiffusionImageMode::InstructImage;
		activeTask = ofxStableDiffusionTask::InstructImage;
		setLastError(batchResult.code, batchResult.message);
		return;
	}

	ofxStableDiffusionImageRequest request;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		request.selectionMode = imageSelectionMode;
		request.styleStrength = styleStrength;
		request.loras = loras;
	}
	request.mode = ofxStableDiffusionImageMode::InstructImage;
	request.initImage = initImage_;
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
	request.normalizeInput = normalizeInput_;
	generate(request);
}

void ofxStableDiffusion::img2vid(sd_image_t initImage_,
	int width_,
	int height_,
	int videoFrames_,
	int fps_,
	float cfgScale_,
	enum sample_method_t sampleMethod_,
	int sampleSteps_,
	float strength_,
	int64_t seed_) {
	if (thread.isThreadRunning()) {
		setLastError(ofxStableDiffusionErrorCode::ThreadBusy, "A task is already running");
		return;
	}

	if (initImage_.data == nullptr) {
		activeTask = ofxStableDiffusionTask::ImageToVideo;
		setLastError(ofxStableDiffusionErrorCode::MissingInputImage, "Image-to-video requires an input image");
		return;
	}

	const ValidationResult dimResult = validateDimensions(width_, height_);
	if (!dimResult.ok()) {
		activeTask = ofxStableDiffusionTask::ImageToVideo;
		setLastError(dimResult.code, dimResult.message);
		return;
	}
	if (videoFrames_ <= 0 || videoFrames_ > 100) {
		activeTask = ofxStableDiffusionTask::ImageToVideo;
		setLastError(ofxStableDiffusionErrorCode::InvalidFrameCount, "Frame count must be between 1 and 100");
		return;
	}

	ofxStableDiffusionVideoRequest request;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		request.prompt = prompt;
		request.negativePrompt = negativePrompt;
		request.clipSkip = clipSkip;
		request.vaceStrength = vaceStrength;
		request.mode = videoMode;
		request.loras = loras;
	}
	request.initImage = initImage_;
	request.width = width_;
	request.height = height_;
	request.frameCount = videoFrames_;
	request.fps = fps_;
	request.cfgScale = cfgScale_;
	request.sampleMethod = sampleMethod_;
	request.sampleSteps = sampleSteps_;
	request.strength = strength_;
	request.seed = seed_;
	generateVideo(request);
}

void ofxStableDiffusion::newUpscalerCtx(const std::string& esrganPath_,
	int nThreads_,
	enum sd_type_t wType_) {
	int requestedMultiplier = 4;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		requestedMultiplier = esrganMultiplier;
	}
	const ofxStableDiffusionUpscalerSettings requested{
		esrganPath_,
		nThreads_,
		wType_,
		requestedMultiplier,
		true
	};

	const ValidationResult validation = validateUpscalerSettings(requested);
	activeTask = ofxStableDiffusionTask::Upscale;
	if (!validation.ok()) {
		setLastError(validation.code, validation.message);
		return;
	}

	if (thread.isThreadRunning()) {
		setLastError(ofxStableDiffusionErrorCode::ThreadBusy, "Cannot rebuild the upscaler while another task is running");
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

void ofxStableDiffusion::freeUpscalerCtx() {
	if (thread.isThreadRunning()) {
		thread.waitForThread(true);
	}
	if (thread.upscalerCtx) {
		free_upscaler_ctx(thread.upscalerCtx);
		thread.upscalerCtx = nullptr;
	}
	std::lock_guard<std::mutex> lock(stateMutex);
	isESRGAN = false;
}

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

bool ofxStableDiffusion::convert(const char* inputPath_, const char* vaePath_, const char* outputPath_, sd_type_t outputType_) {
	return ::convert(inputPath_, vaePath_, outputPath_, outputType_, nullptr, false);
}

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
	return ::preprocess_canny(image, highThreshold, lowThreshold, weak, strong, inverse) ? img : nullptr;
}

bool ofxStableDiffusion::isGenerating() const {
	return thread.isThreadRunning();
}

int64_t ofxStableDiffusion::getLastUsedSeed() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResult.actualSeedUsed;
}

std::vector<int64_t> ofxStableDiffusion::getSeedHistory() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return std::vector<int64_t>(seedHistory.begin(), seedHistory.end());
}

void ofxStableDiffusion::clearSeedHistory() {
	std::lock_guard<std::mutex> lock(stateMutex);
	seedHistory.clear();
}

int64_t ofxStableDiffusion::hashStringToSeed(const std::string& text) {
	return ofxStableDiffusionHashStringToSeed(text);
}

bool ofxStableDiffusion::beginBackgroundTask(ofxStableDiffusionTask task) {
	if (thread.isThreadRunning()) {
		activeTask = task;
		setLastError(ofxStableDiffusionErrorCode::ThreadBusy, "Another task is still running");
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

ofxStableDiffusionContextSettings ofxStableDiffusion::captureContextSettingsNoLock() const {
	ofxStableDiffusionContextSettings settings;
	settings.modelPath = modelPath;
	settings.diffusionModelPath = diffusionModelPath;
	settings.clipLPath = clipLPath;
	settings.clipGPath = clipGPath;
	settings.t5xxlPath = t5xxlPath;
	settings.vaePath = vaePath;
	settings.taesdPath = taesdPath;
	settings.controlNetPath = controlNetPathCStr;
	settings.loraModelDir = loraModelDir;
	settings.embedDir = embedDirCStr;
	settings.stackedIdEmbedDir = stackedIdEmbedDirCStr;
	settings.vaeDecodeOnly = vaeDecodeOnly;
	settings.vaeTiling = vaeTiling;
	settings.freeParamsImmediately = freeParamsImmediately;
	settings.nThreads = nThreads;
	settings.weightType = wType;
	settings.backend = backend;
	settings.rngType = rngType;
	settings.schedule = schedule;
	settings.prediction = prediction;
	settings.loraApplyMode = loraApplyMode;
	settings.keepClipOnCpu = keepClipOnCpu;
	settings.keepControlNetCpu = keepControlNetCpu;
	settings.keepVaeOnCpu = keepVaeOnCpu;
	settings.offloadParamsToCpu = offloadParamsToCpu;
	settings.flashAttn = flashAttn;
	settings.enableMmap = enableMmap;
	return settings;
}

ofxStableDiffusionUpscalerSettings ofxStableDiffusion::captureUpscalerSettingsNoLock() const {
	return {esrganPath, nThreads, wType, esrganMultiplier, isESRGAN};
}

void ofxStableDiffusion::applyContextSettings(const ofxStableDiffusionContextSettings& settings) {
	const ofxStableDiffusionContextSettings resolvedSettings = resolveContextModelPaths(settings);
	std::lock_guard<std::mutex> lock(stateMutex);
	modelPath = resolvedSettings.modelPath;
	diffusionModelPath = resolvedSettings.diffusionModelPath;
	clipLPath = resolvedSettings.clipLPath;
	clipGPath = resolvedSettings.clipGPath;
	t5xxlPath = resolvedSettings.t5xxlPath;
	modelName = ofFilePath::getFileName(modelPath.empty() ? diffusionModelPath : modelPath);
	vaePath = resolvedSettings.vaePath;
	taesdPath = resolvedSettings.taesdPath;
	controlNetPathCStr = resolvedSettings.controlNetPath;
	loraModelDir = resolvedSettings.loraModelDir;
	embedDirCStr = resolvedSettings.embedDir;
	stackedIdEmbedDirCStr = resolvedSettings.stackedIdEmbedDir;
	vaeDecodeOnly = resolvedSettings.vaeDecodeOnly;
	vaeTiling = resolvedSettings.vaeTiling;
	freeParamsImmediately = resolvedSettings.freeParamsImmediately;
	nThreads = resolvedSettings.nThreads;
	wType = resolvedSettings.weightType;
	backend = resolvedSettings.backend;
	rngType = resolvedSettings.rngType;
	schedule = resolvedSettings.schedule;
	prediction = resolvedSettings.prediction;
	loraApplyMode = resolvedSettings.loraApplyMode;
	keepClipOnCpu = resolvedSettings.keepClipOnCpu;
	keepControlNetCpu = resolvedSettings.keepControlNetCpu;
	keepVaeOnCpu = resolvedSettings.keepVaeOnCpu;
	offloadParamsToCpu = resolvedSettings.offloadParamsToCpu;
	flashAttn = resolvedSettings.flashAttn;
	enableMmap = resolvedSettings.enableMmap;
}

void ofxStableDiffusion::applyImageRequest(const ofxStableDiffusionImageRequest& request) {
	stableDiffusionThread::ImageTaskData taskData;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		taskData.task = activeTask;
		taskData.contextSettings = captureContextSettingsNoLock();
		taskData.upscalerSettings = captureUpscalerSettingsNoLock();
		taskData.request = request;
		if (request.initImage.data != nullptr) {
			loadedInputImage.assign(request.initImage);
			inputImage = loadedInputImage.image;
		} else if (!ofxStableDiffusionImageModeUsesInputImage(request.mode)) {
			inputImage = {0, 0, 0, nullptr};
			loadedInputImage.clear();
		}

		if (ofxStableDiffusionImageModeUsesInputImage(request.mode) && inputImage.data != nullptr) {
			taskData.initImage.assign(inputImage);
		}
		taskData.maskImage.assign(request.maskImage);
		if (request.controlCond != nullptr) {
			taskData.controlImage.assign(*request.controlCond);
		}
		taskData.syncViews();
		taskData.progressCallback = progressCallback;
		taskData.imageRankCallback = imageRankCallback;

		imageMode = request.mode;
		imageSelectionMode = request.selectionMode;
		maskImage = request.maskImage;
		endImage = {0, 0, 0, nullptr};
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
		seed = request.seed;
		batchCount = request.batchCount;
		controlCond = request.controlCond;
		controlStrength = request.controlStrength;
		styleStrength = request.styleStrength;
		normalizeInput = request.normalizeInput;
		inputIdImagesPath = request.inputIdImagesPath;
		loras = request.loras;
	}
	thread.prepareImageTask(taskData);
}

void ofxStableDiffusion::applyVideoRequest(const ofxStableDiffusionVideoRequest& request) {
	stableDiffusionThread::VideoTaskData taskData;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		taskData.task = activeTask;
		taskData.contextSettings = captureContextSettingsNoLock();
		taskData.upscalerSettings = captureUpscalerSettingsNoLock();
		taskData.request = request;
		loadedInputImage.assign(request.initImage);
		inputImage = loadedInputImage.image;
		taskData.initImage.assign(inputImage);
		taskData.endImage.assign(request.endImage);
		taskData.syncViews();
		taskData.progressCallback = progressCallback;

		endImage = request.endImage;
		prompt = request.prompt;
		negativePrompt = request.negativePrompt;
		clipSkip = request.clipSkip;
		width = request.width;
		height = request.height;
		videoFrames = request.frameCount;
		fps = request.fps;
		cfgScale = request.cfgScale;
		sampleMethodEnum = request.sampleMethod;
		sampleSteps = request.sampleSteps;
		strength = request.strength;
		seed = request.seed;
		vaceStrength = request.vaceStrength;
		videoMode = request.mode;
		loras = request.loras;
	}
	thread.prepareVideoTask(taskData);
}

bool ofxStableDiffusion::validateImageRequestAndSetError(const ofxStableDiffusionImageRequest& request, ofxStableDiffusionTask task) {
	const ValidationResult validation = validateImageRequestNumbers(request);
	imageMode = request.mode;
	activeTask = task;
	if (!validation.ok()) {
		setLastError(validation.code, validation.message);
		return false;
	}

	sd_image_t candidateInputImage = request.initImage;
	if (candidateInputImage.data == nullptr) {
		std::lock_guard<std::mutex> lock(stateMutex);
		candidateInputImage = inputImage;
	}
	if (ofxStableDiffusionImageModeUsesInputImage(request.mode) && candidateInputImage.data == nullptr) {
		setLastError(ofxStableDiffusionErrorCode::MissingInputImage, "Selected image mode requires an input image");
		return false;
	}

	if (request.mode == ofxStableDiffusionImageMode::Inpainting && request.maskImage.data == nullptr) {
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, "Inpainting requires a mask image");
		return false;
	}

	// Validate mask dimensions match init image if both are provided
	if (request.mode == ofxStableDiffusionImageMode::Inpainting &&
		request.maskImage.data != nullptr &&
		candidateInputImage.data != nullptr) {
		if (request.maskImage.width != candidateInputImage.width ||
			request.maskImage.height != candidateInputImage.height) {
			setLastError(ofxStableDiffusionErrorCode::InvalidDimensions,
				"Inpainting mask dimensions must match input image dimensions");
			return false;
		}
	}

	return true;
}

bool ofxStableDiffusion::validateVideoRequestAndSetError(const ofxStableDiffusionVideoRequest& request) {
	const ValidationResult validation = validateVideoRequestNumbers(request);
	activeTask = ofxStableDiffusionTask::ImageToVideo;
	if (!validation.ok()) {
		setLastError(validation.code, validation.message);
		return false;
	}
	if (request.initImage.data == nullptr) {
		setLastError(ofxStableDiffusionErrorCode::MissingInputImage, "Video generation requires an input image");
		return false;
	}
	return true;
}

void ofxStableDiffusion::clearOutputState() {
	std::lock_guard<std::mutex> lock(stateMutex);
	outputImages = nullptr;
	outputImageViews.clear();
	lastResult = {};
	diffused = false;
}

void ofxStableDiffusion::setLastError(const std::string& errorMessage, ofxStableDiffusionErrorCode code) {
	std::lock_guard<std::mutex> lock(stateMutex);
	lastError = errorMessage;
	lastErrorInfo.code = code;
	lastErrorInfo.message = errorMessage;
	lastErrorInfo.suggestion = ofxStableDiffusionErrorCodeSuggestion(code);
	lastErrorInfo.timestampMicros = ofGetElapsedTimeMicros();
	errorHistory.push_back(lastErrorInfo);
	if (errorHistory.size() > maxErrorHistorySize) {
		errorHistory.pop_front();
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

void ofxStableDiffusion::setLastError(ofxStableDiffusionErrorCode code, const std::string& errorMessage) {
	setLastError(errorMessage, code);
}

void ofxStableDiffusion::clearLastError() {
	std::lock_guard<std::mutex> lock(stateMutex);
	lastError.clear();
	lastErrorInfo = ofxStableDiffusionError();
	lastResult.error.clear();
}

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

void ofxStableDiffusion::captureImageResults(
	sd_image_t* images,
	int count,
	int64_t seedValue,
	float elapsedMs,
	ofxStableDiffusionTask task,
	const ofxStableDiffusionImageRequest& request,
	const ofxSdImageRankCallback& rankCallback) {
	ofxStableDiffusionResult result;
	result.success = true;
	result.task = task;
	result.imageMode = request.mode;
	result.selectionMode = request.selectionMode;
	result.elapsedMs = elapsedMs;
	result.actualSeedUsed = seedValue;
	result.images.reserve(std::max(0, count));

	for (int i = 0; i < count; ++i) {
		ofxStableDiffusionImageFrame frame;
		frame.index = i;
		frame.sourceIndex = i;
		frame.seed = seedValue;
		frame.generation.prompt = request.prompt;
		frame.generation.negativePrompt = request.negativePrompt;
		frame.generation.cfgScale = request.cfgScale;
		frame.generation.strength = request.strength;
		frame.pixels = makePixelsCopy(images[i]);
		result.images.push_back(std::move(frame));
	}

	applyImageRanking(result.images, result, request, rankCallback);

	{
		std::lock_guard<std::mutex> lock(stateMutex);
		seedHistory.push_back(seedValue);
		if (seedHistory.size() > maxSeedHistorySize) {
			seedHistory.pop_front();
		}
		lastResult = std::move(result);
		outputImageViews = buildOutputImageViews(lastResult);
		outputImages = outputImageViews.empty() ? nullptr : outputImageViews.data();
		diffused = true;
	}

	ofxSdReleaseImageArray(images, count);
}

void ofxStableDiffusion::captureVideoResults(
	sd_image_t* images,
	int count,
	int64_t seedValue,
	const std::vector<int64_t>& frameSeeds,
	const std::vector<ofxStableDiffusionGenerationParameters>& frameGeneration,
	float elapsedMs,
	ofxStableDiffusionTask task,
	const ofxStableDiffusionVideoRequest& request) {
	ofxStableDiffusionResult result;
	result.success = true;
	result.task = task;
	result.elapsedMs = elapsedMs;
	result.actualSeedUsed = seedValue;
	result.video.fps = request.fps;
	result.video.sourceFrameCount = count;
	result.video.mode = request.mode;

	std::vector<ofxStableDiffusionImageFrame> sourceFrames;
	sourceFrames.reserve(std::max(0, count));
	for (int i = 0; i < count; ++i) {
		ofxStableDiffusionImageFrame frame;
		frame.index = i;
		frame.sourceIndex = i;
		frame.seed =
			static_cast<std::size_t>(i) < frameSeeds.size() ?
				frameSeeds[static_cast<std::size_t>(i)] :
				seedValue;
		frame.generation =
			static_cast<std::size_t>(i) < frameGeneration.size() ?
				frameGeneration[static_cast<std::size_t>(i)] :
				ofxStableDiffusionGenerationParameters{
					request.prompt,
					request.negativePrompt,
					request.cfgScale,
					request.strength
				};
		frame.pixels = makePixelsCopy(images[i]);
		sourceFrames.push_back(std::move(frame));
	}

	result.video.frames = ofxStableDiffusionBuildVideoFrames(sourceFrames, request.mode);

	{
		std::lock_guard<std::mutex> lock(stateMutex);
		seedHistory.push_back(seedValue);
		if (seedHistory.size() > maxSeedHistorySize) {
			seedHistory.pop_front();
		}
		lastResult = std::move(result);
		outputImageViews = buildOutputImageViews(lastResult);
		outputImages = outputImageViews.empty() ? nullptr : outputImageViews.data();
		diffused = true;
	}

	ofxSdReleaseImageArray(images, count);
}

void ofxStableDiffusion::applyImageRanking(
	std::vector<ofxStableDiffusionImageFrame>& frames,
	ofxStableDiffusionResult& result,
	const ofxStableDiffusionImageRequest& request,
	const ofxSdImageRankCallback& rankCallback) {
	result.rankingApplied = false;
	result.selectedImageIndex = frames.empty() ? -1 : 0;
	if (frames.empty()) {
		return;
	}

	for (auto& frame : frames) {
		frame.isSelected = false;
	}

	if (!rankCallback) {
		frames.front().isSelected = true;
		return;
	}

	const std::vector<ofxStableDiffusionImageScore> scores = rankCallback(request, frames);
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
	if (request.selectionMode == ofxStableDiffusionImageSelectionMode::Rerank ||
		request.selectionMode == ofxStableDiffusionImageSelectionMode::BestOnly) {
		std::vector<ofxStableDiffusionImageFrame> ranked;
		ranked.reserve(frames.size());
		for (const std::size_t index : order) {
			ranked.push_back(frames[index]);
		}
		frames = std::move(ranked);
		if (request.selectionMode == ofxStableDiffusionImageSelectionMode::BestOnly && !frames.empty()) {
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

