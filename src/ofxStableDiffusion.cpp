#include "ofxStableDiffusion.h"
#include "core/ofxStableDiffusionCapabilityHelpers.h"
#include "core/ofxStableDiffusionLimits.h"
#include "core/ofxStableDiffusionMemoryHelpers.h"
#include "core/ofxStableDiffusionNativeAdapter.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <mutex>

namespace {

namespace fs = std::filesystem;
std::atomic<bool> g_sdLoggingEnabled{true};
std::atomic<int> g_sdMinLogLevel{SD_LOG_DEBUG};
bool isProgressLikeSdLog(const char* log) {
	if (log == nullptr) {
		return false;
	}
	if (std::strchr(log, '\r') != nullptr) {
		return true;
	}
	const char* p = log;
	while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') {
		++p;
	}
	return *p == '|';
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
	(void)data;
	if (!g_sdLoggingEnabled.load()) {
		return;
	}
	if (level < g_sdMinLogLevel.load()) {
		return;
	}
	if (log == nullptr || log[0] == '\0') {
		return;
	}
	if (isProgressLikeSdLog(log)) {
		FILE* stream = (level <= SD_LOG_INFO) ? stdout : stderr;
		std::string prefixed;
		prefixed.reserve(std::strlen(log) + 16);
		bool atLineStart = true;
		for (const char* p = log; *p != '\0'; ++p) {
			if (*p == '\r' || *p == '\n') {
				prefixed.push_back(*p);
				atLineStart = true;
				continue;
			}
			if (atLineStart) {
				prefixed += "[notice ] ";
				atLineStart = false;
			}
			prefixed.push_back(*p);
		}
		fputs(prefixed.c_str(), stream);
		fflush(stream);
		return;
	}

	std::string message(log);
	while (!message.empty() && (message.back() == '\n' || message.back() == '\r')) {
		message.pop_back();
	}
	if (message.empty()) {
		return;
	}

	switch (level) {
	case SD_LOG_DEBUG:
		ofLogVerbose("stable-diffusion") << message;
		break;
	case SD_LOG_INFO:
		ofLogNotice("stable-diffusion") << message;
		break;
	case SD_LOG_WARN:
		ofLogWarning("stable-diffusion") << message;
		break;
	case SD_LOG_ERROR:
	default:
		ofLogError("stable-diffusion") << message;
		break;
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
	using namespace ofxStableDiffusionLimits;
	if (width <= 0 || height <= 0) {
		return {ofxStableDiffusionErrorCode::InvalidDimensions, "Width and height must be positive values"};
	}
	if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
		return {ofxStableDiffusionErrorCode::InvalidDimensions,
			"Width and height must not exceed " + std::to_string(MAX_DIMENSION) + " pixels"};
	}
	return {};
}

ValidationResult validateBatchCount(int batchCount) {
	using namespace ofxStableDiffusionLimits;
	if (batchCount <= 0) {
		return {ofxStableDiffusionErrorCode::InvalidBatchCount, "Batch count must be positive"};
	}
	if (batchCount > MAX_BATCH_COUNT) {
		return {ofxStableDiffusionErrorCode::InvalidBatchCount,
			"Batch count exceeds maximum of " + std::to_string(MAX_BATCH_COUNT)};
	}
	return {};
}

ValidationResult validateSampleSteps(int sampleSteps) {
	using namespace ofxStableDiffusionLimits;
	if (sampleSteps <= 0 || sampleSteps > MAX_SAMPLE_STEPS) {
		return {ofxStableDiffusionErrorCode::InvalidParameter,
			"Sample steps must be between 1 and " + std::to_string(MAX_SAMPLE_STEPS)};
	}
	return {};
}

ValidationResult validateCfgScale(float cfgScale) {
	using namespace ofxStableDiffusionLimits;
	if (cfgScale <= MIN_CFG_SCALE || cfgScale > MAX_CFG_SCALE) {
		return {ofxStableDiffusionErrorCode::InvalidParameter,
			"CFG scale must be greater than 0 and no more than " + std::to_string(static_cast<int>(MAX_CFG_SCALE))};
	}
	return {};
}

ValidationResult validateStrength(float strength) {
	using namespace ofxStableDiffusionLimits;
	if (!isValidUnitInterval(strength)) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "Strength must be between 0.0 and 1.0"};
	}
	return {};
}

ValidationResult validateClipSkip(int clipSkip) {
	using namespace ofxStableDiffusionLimits;
	if (!isValidClipSkip(clipSkip)) {
		return {ofxStableDiffusionErrorCode::InvalidParameter,
			"Clip skip must be -1 (auto) or between 0 and " + std::to_string(MAX_CLIP_SKIP)};
	}
	return {};
}

ValidationResult validateSeed(int64_t seed) {
	using namespace ofxStableDiffusionLimits;
	if (!isValidSeed(seed)) {
		return {ofxStableDiffusionErrorCode::InvalidParameter,
			"Seed must be -1 for randomization or a non-negative value"};
	}
	return {};
}

ValidationResult validateControlStrength(float controlStrength) {
	using namespace ofxStableDiffusionLimits;
	if (!isValidControlStrength(controlStrength)) {
		return {ofxStableDiffusionErrorCode::InvalidParameter,
			"Control strength must be between 0.0 and " + std::to_string(static_cast<int>(MAX_CONTROL_STRENGTH))};
	}
	return {};
}

ValidationResult validateStyleStrength(float styleStrength) {
	using namespace ofxStableDiffusionLimits;
	if (!isValidStyleStrength(styleStrength)) {
		return {ofxStableDiffusionErrorCode::InvalidParameter,
			"Style strength must be between 0 and " + std::to_string(static_cast<int>(MAX_STYLE_STRENGTH))};
	}
	return {};
}

ValidationResult validateVaceStrength(float vaceStrength) {
	using namespace ofxStableDiffusionLimits;
	if (!isValidUnitInterval(vaceStrength)) {
		return {ofxStableDiffusionErrorCode::InvalidParameter, "VACE strength must be between 0.0 and 1.0"};
	}
	return {};
}

ValidationResult validateUnitInterval(float value, const std::string& label) {
	if (value < 0.0f || value > 1.0f) {
		return {
			ofxStableDiffusionErrorCode::InvalidParameter,
			label + " must be between 0.0 and 1.0"
		};
	}
	return {};
}

bool isNativeWanVideoFamily(ofxStableDiffusionModelFamily family) {
	return family == ofxStableDiffusionModelFamily::WAN ||
		family == ofxStableDiffusionModelFamily::WANI2V ||
		family == ofxStableDiffusionModelFamily::WANTI2V ||
		family == ofxStableDiffusionModelFamily::WANFLF2V ||
		family == ofxStableDiffusionModelFamily::WANVACE;
}

ValidationResult validateControlNets(
	const std::vector<ofxStableDiffusionControlNet>& controlNets,
	int width,
	int height) {
	if (controlNets.empty()) {
		return {};
	}

	int expectedWidth = -1;
	int expectedHeight = -1;
	int expectedChannels = -1;
	for (const auto& controlNet : controlNets) {
		if (controlNet.conditionImage.data == nullptr) {
			return {ofxStableDiffusionErrorCode::InvalidParameter, "ControlNet image is missing"};
		}
		if (controlNet.strength < 0.0f || controlNet.strength > 2.0f) {
			return {ofxStableDiffusionErrorCode::InvalidParameter, "ControlNet strength must be between 0.0 and 2.0"};
		}

		const int controlWidth = static_cast<int>(controlNet.conditionImage.width);
		const int controlHeight = static_cast<int>(controlNet.conditionImage.height);
		const int controlChannels = static_cast<int>(controlNet.conditionImage.channel);

		if (expectedWidth < 0) {
			expectedWidth = controlWidth;
			expectedHeight = controlHeight;
			expectedChannels = controlChannels;
		} else if (controlWidth != expectedWidth ||
			controlHeight != expectedHeight ||
			controlChannels != expectedChannels) {
			return {ofxStableDiffusionErrorCode::InvalidDimensions, "All ControlNet images must share width, height, and channel count"};
		}

		if ((width > 0 && controlWidth != width) || (height > 0 && controlHeight != height)) {
			return {ofxStableDiffusionErrorCode::InvalidDimensions, "ControlNet image dimensions must match the request dimensions"};
		}
	}

	return {};
}

ValidationResult validateImageRequestNumbers(const ofxStableDiffusionImageRequest& request) {
	const ValidationResult dimResult = validateDimensions(request.width, request.height);
	if (!dimResult.ok()) return dimResult;

	const ValidationResult batchResult = validateBatchCount(request.batchCount);
	if (!batchResult.ok()) return batchResult;

	if (request.sampleSteps > 0) {
		const ValidationResult stepsResult = validateSampleSteps(request.sampleSteps);
		if (!stepsResult.ok()) return stepsResult;
	}

	if (std::isfinite(request.cfgScale)) {
		const ValidationResult cfgResult = validateCfgScale(request.cfgScale);
		if (!cfgResult.ok()) return cfgResult;
	}

	if (std::isfinite(request.strength)) {
		const ValidationResult strengthResult = validateStrength(request.strength);
		if (!strengthResult.ok()) return strengthResult;
	}

	const ValidationResult clipResult = validateClipSkip(request.clipSkip);
	if (!clipResult.ok()) return clipResult;

	const ValidationResult seedResult = validateSeed(request.seed);
	if (!seedResult.ok()) return seedResult;

	const ValidationResult controlResult = validateControlStrength(request.controlStrength);
	if (!controlResult.ok()) return controlResult;

	const ValidationResult styleResult = validateStyleStrength(request.styleStrength);
	if (!styleResult.ok()) return styleResult;

	const ValidationResult controlNetResult =
		validateControlNets(request.controlNets, request.width, request.height);
	if (!controlNetResult.ok()) return controlNetResult;

	return {};
}

	ValidationResult validateVideoRequestNumbers(const ofxStableDiffusionVideoRequest& request) {
		const ValidationResult dimResult = validateDimensions(request.width, request.height);
		if (!dimResult.ok()) return dimResult;

		if (!ofxStableDiffusionLimits::isValidFrameCount(request.frameCount)) {
			return {ofxStableDiffusionErrorCode::InvalidFrameCount,
				"Frame count must be between " +
				std::to_string(ofxStableDiffusionLimits::MIN_FRAME_COUNT) + " and " +
				std::to_string(ofxStableDiffusionLimits::MAX_FRAME_COUNT)};
		}

		if (!ofxStableDiffusionLimits::isValidFps(request.fps)) {
			return {ofxStableDiffusionErrorCode::InvalidParameter,
				"FPS must be between " +
				std::to_string(ofxStableDiffusionLimits::MIN_FPS) + " and " +
				std::to_string(ofxStableDiffusionLimits::MAX_FPS)};
		}

	const ValidationResult clipResult = validateClipSkip(request.clipSkip);
	if (!clipResult.ok()) return clipResult;

	if (std::isfinite(request.cfgScale)) {
		const ValidationResult cfgResult = validateCfgScale(request.cfgScale);
		if (!cfgResult.ok()) return cfgResult;
	}

	if (request.sampleSteps > 0) {
		const ValidationResult stepsResult = validateSampleSteps(request.sampleSteps);
		if (!stepsResult.ok()) return stepsResult;
	}

	if (std::isfinite(request.strength)) {
		const ValidationResult strengthResult = validateStrength(request.strength);
		if (!strengthResult.ok()) return strengthResult;
	}

	const ValidationResult seedResult = validateSeed(request.seed);
	if (!seedResult.ok()) return seedResult;

	for (std::size_t i = 0; i < request.controlFrames.size(); ++i) {
		const auto& frame = request.controlFrames[i];
		if (frame.data == nullptr || frame.width == 0 || frame.height == 0 || frame.channel == 0) {
			return {
				ofxStableDiffusionErrorCode::InvalidParameter,
				"Control frame " + ofToString(static_cast<int>(i)) + " is not allocated"
			};
		}
	}

	if (request.cache.mode == SD_CACHE_EASYCACHE || request.cache.mode == SD_CACHE_UCACHE) {
		if (request.cache.reuse_threshold < 0.0f) {
			return {
				ofxStableDiffusionErrorCode::InvalidParameter,
				"Video cache reuse threshold must be non-negative"
			};
		}
		if (request.cache.start_percent < 0.0f ||
			request.cache.start_percent >= 1.0f ||
			request.cache.end_percent <= 0.0f ||
			request.cache.end_percent > 1.0f ||
			request.cache.start_percent >= request.cache.end_percent) {
			return {
				ofxStableDiffusionErrorCode::InvalidParameter,
				"Video cache start/end percents must satisfy 0.0 <= start < end <= 1.0"
			};
		}
	}

	if (std::isfinite(request.moeBoundary)) {
		const ValidationResult moeBoundaryResult =
			validateUnitInterval(request.moeBoundary, "MoE boundary");
		if (!moeBoundaryResult.ok()) return moeBoundaryResult;
	}

	if (std::isfinite(request.vaceStrength)) {
		const ValidationResult vaceResult = validateVaceStrength(request.vaceStrength);
		if (!vaceResult.ok()) return vaceResult;
	}

	if (request.hasAnimation()) {
		std::string animationError;
		if (!ofxStableDiffusionValidateAnimationKeyframes(
				request.animationSettings,
				request.frameCount,
				animationError)) {
			return {ofxStableDiffusionErrorCode::InvalidParameter, animationError};
		}
	}

	return {};
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

bool mergeControlNets(
	const std::vector<ofxStableDiffusionControlNet>& controlNets,
	stableDiffusionThread::OwnedImage& output,
	float& resolvedStrength,
	std::string& errorMessage) {
	if (controlNets.empty()) {
		return false;
	}

	const sd_image_t& reference = controlNets.front().conditionImage;
	const std::size_t byteCount =
		static_cast<std::size_t>(reference.width) *
		static_cast<std::size_t>(reference.height) *
		static_cast<std::size_t>(reference.channel);

	std::vector<double> accum(byteCount, 0.0);
	double totalStrength = 0.0;

	for (std::size_t idx = 0; idx < controlNets.size(); ++idx) {
		const auto& controlNet = controlNets[idx];
		if (controlNet.conditionImage.data == nullptr) {
			errorMessage = "ControlNet image is missing (index " + std::to_string(idx) + ")";
			return false;
		}
		if (controlNet.strength < 0.0f || controlNet.strength > 2.0f) {
			errorMessage = "ControlNet strength must be between 0.0 and 2.0 (index " + std::to_string(idx) + ")";
			return false;
		}

		const double weight = static_cast<double>(controlNet.strength);
		totalStrength += weight;
		const uint8_t* data = controlNet.conditionImage.data;
		for (std::size_t i = 0; i < byteCount; ++i) {
			accum[i] += weight * static_cast<double>(data[i]);
		}
	}

	if (totalStrength <= 0.0) {
		errorMessage = "Combined ControlNet strength must be greater than zero";
		return false;
	}

	output.storage.resize(byteCount);
	for (std::size_t i = 0; i < byteCount; ++i) {
		const double value = accum[i] / totalStrength;
		output.storage[i] = static_cast<uint8_t>(std::clamp(
			std::llround(value),
			0ll,
			255ll));
	}

	output.image = {
		reference.width,
		reference.height,
		reference.channel,
		output.storage.data()
	};

	resolvedStrength = static_cast<float>(std::min(
		totalStrength / static_cast<double>(controlNets.size()),
		2.0));
	return true;
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

bool contextSettingsEquivalent(
	const ofxStableDiffusionContextSettings& lhs,
	const ofxStableDiffusionContextSettings& rhs) {
	return lhs.modelPath == rhs.modelPath &&
		lhs.diffusionModelPath == rhs.diffusionModelPath &&
		lhs.clipLPath == rhs.clipLPath &&
		lhs.clipGPath == rhs.clipGPath &&
		lhs.t5xxlPath == rhs.t5xxlPath &&
		lhs.vaePath == rhs.vaePath &&
		lhs.taesdPath == rhs.taesdPath &&
		lhs.controlNetPath == rhs.controlNetPath &&
		lhs.loraModelDir == rhs.loraModelDir &&
		lhs.embedDir == rhs.embedDir &&
		lhs.stackedIdEmbedDir == rhs.stackedIdEmbedDir &&
		lhs.vaeDecodeOnly == rhs.vaeDecodeOnly &&
		lhs.vaeTiling == rhs.vaeTiling &&
		lhs.freeParamsImmediately == rhs.freeParamsImmediately &&
		lhs.nThreads == rhs.nThreads &&
		lhs.weightType == rhs.weightType &&
		lhs.rngType == rhs.rngType &&
		lhs.schedule == rhs.schedule &&
		lhs.prediction == rhs.prediction &&
		lhs.loraApplyMode == rhs.loraApplyMode &&
		lhs.keepClipOnCpu == rhs.keepClipOnCpu &&
		lhs.keepControlNetCpu == rhs.keepControlNetCpu &&
		lhs.keepVaeOnCpu == rhs.keepVaeOnCpu &&
		lhs.offloadParamsToCpu == rhs.offloadParamsToCpu &&
		lhs.flashAttn == rhs.flashAttn &&
		lhs.diffusionFlashAttn == rhs.diffusionFlashAttn &&
		lhs.enableMmap == rhs.enableMmap;
}

} // namespace

ofxStableDiffusion::ofxStableDiffusion() {
	sd_set_log_callback(sd_log_cb, nullptr);
	thread.userData = this;
	cachedResolvedSchedulersBySampleMethod.assign(static_cast<std::size_t>(SAMPLE_METHOD_COUNT), SCHEDULER_COUNT);
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

void ofxStableDiffusion::generate(const ofxStableDiffusionImageRequest& request) {
	const ofxStableDiffusionTask task = ofxStableDiffusionTaskForImageMode(request.mode);
	if (!validateImageRequestAndSetError(request, task)) {
		return;
	}
	if (!beginBackgroundTask(task)) {
		return;
	}
	if (!applyImageRequest(request)) {
		return;
	}
	thread.startThread();
}

void ofxStableDiffusion::generateVideo(const ofxStableDiffusionVideoRequest& request) {
	if (!validateVideoRequestAndSetError(request)) {
		return;
	}
	const ofxStableDiffusionCapabilities capabilities = getCapabilities();
	if (!capabilities.contextConfigured) {
		setLastError(ofxStableDiffusionErrorCode::ModelNotFound, "Video generation requires a loaded model");
		return;
	}
	if (!capabilities.imageToVideo) {
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, "Current model does not support image-to-video generation");
		return;
	}
	if (request.endImage.data != nullptr && !capabilities.videoEndFrame) {
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, "This model does not support providing an end frame");
		return;
	}
	if (request.hasAnimation() && !capabilities.videoAnimation) {
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, "Animated video generation is not supported by the current model");
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
	cachedUpscalerSettings = settings;
	esrganPath = settings.modelPath;
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

bool ofxStableDiffusion::hasLoadedContext() const {
	return thread.sdCtx != nullptr;
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

std::string ofxStableDiffusion::getLastResolvedVideoRequestSummary() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResolvedVideoRequestSummary;
}

std::string ofxStableDiffusion::getLastResolvedVideoCliCommand() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastResolvedVideoCliCommand;
}

sample_method_t ofxStableDiffusion::getResolvedSampleMethod(sample_method_t requested) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (requested != SAMPLE_METHOD_COUNT) {
		return requested;
	}
	return cachedResolvedDefaultSampleMethod;
}

scheduler_t ofxStableDiffusion::getResolvedScheduler(
	sample_method_t requestedSampleMethod,
	scheduler_t requestedSchedule) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (requestedSchedule != SCHEDULER_COUNT) {
		return requestedSchedule;
	}
	if (requestedSampleMethod == SAMPLE_METHOD_COUNT) {
		return cachedResolvedDefaultScheduler;
	}
	const int sampleMethodIndex = static_cast<int>(requestedSampleMethod);
	if (sampleMethodIndex < 0 ||
		sampleMethodIndex >= static_cast<int>(cachedResolvedSchedulersBySampleMethod.size())) {
		return SCHEDULER_COUNT;
	}
	return cachedResolvedSchedulersBySampleMethod[static_cast<std::size_t>(sampleMethodIndex)];
}

std::string ofxStableDiffusion::getResolvedSampleMethodName(sample_method_t requested) const {
	const sample_method_t resolved = getResolvedSampleMethod(requested);
	if (resolved == SAMPLE_METHOD_COUNT) {
		return "MODEL_DEFAULT";
	}
	return sd_sample_method_name(resolved);
}

std::string ofxStableDiffusion::getResolvedSchedulerName(
	sample_method_t requestedSampleMethod,
	scheduler_t requestedSchedule) const {
	const scheduler_t resolved = getResolvedScheduler(requestedSampleMethod, requestedSchedule);
	if (resolved == SCHEDULER_COUNT) {
		return "MODEL_DEFAULT";
	}
	return sd_scheduler_name(resolved);
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

const ofPixels* ofxStableDiffusion::getImagePixels(int index) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (index < 0 || index >= static_cast<int>(lastResult.images.size())) {
		return nullptr;
	}
	return &lastResult.images[static_cast<std::size_t>(index)].pixels;
}

bool ofxStableDiffusion::copyImagePixels(int index, ofPixels& pixels) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (index < 0 || index >= static_cast<int>(lastResult.images.size())) {
		return false;
	}
	const auto& storedPixels = lastResult.images[static_cast<std::size_t>(index)].pixels;
	if (!storedPixels.isAllocated()) {
		return false;
	}
	pixels = storedPixels;
	return true;
}

bool ofxStableDiffusion::getImageFrameMetadata(
	int index,
	ofxStableDiffusionImageScore& score,
	bool& isSelected) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (index < 0 || index >= static_cast<int>(lastResult.images.size())) {
		return false;
	}
	const auto& frame = lastResult.images[static_cast<std::size_t>(index)];
	score = frame.score;
	isSelected = frame.isSelected;
	return true;
}

const ofPixels* ofxStableDiffusion::getVideoFramePixels(int index) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (index < 0 || index >= static_cast<int>(lastResult.video.frames.size())) {
		return nullptr;
	}
	return &lastResult.video.frames[static_cast<std::size_t>(index)].pixels;
}

bool ofxStableDiffusion::copyVideoFramePixels(int index, ofPixels& pixels) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (index < 0 || index >= static_cast<int>(lastResult.video.frames.size())) {
		return false;
	}
	const auto& storedPixels = lastResult.video.frames[static_cast<std::size_t>(index)].pixels;
	if (!storedPixels.isAllocated()) {
		return false;
	}
	pixels = storedPixels;
	return true;
}

bool ofxStableDiffusion::getVideoFrameMetadata(
	int index,
	int64_t& seed,
	ofxStableDiffusionGenerationParameters& generation) const {
	std::lock_guard<std::mutex> lock(stateMutex);
	if (index < 0 || index >= static_cast<int>(lastResult.video.frames.size())) {
		return false;
	}
	const auto& frame = lastResult.video.frames[static_cast<std::size_t>(index)];
	seed = frame.seed;
	generation = frame.generation;
	return true;
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
	isTextToImage.store(mode == ofxStableDiffusionImageMode::TextToImage, std::memory_order_relaxed);
	isImageToVideo.store(false, std::memory_order_relaxed);
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
	// Create a temporary sd_image_t with the pixel data, then copy via assign()
	sd_image_t tempImage{
		static_cast<uint32_t>(pixels.getWidth()),
		static_cast<uint32_t>(pixels.getHeight()),
		static_cast<uint32_t>(pixels.getNumChannels()),
		pixels.getData()
	};
	loadedInputImage.assign(tempImage);
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
	std::string targetDir;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		targetDir = loraModelDir;
	}
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

std::vector<ofxStableDiffusionModelInfo> ofxStableDiffusion::scanModels(const std::string& directory) {
	return modelManager.scanModelsInDirectory(directory);
}

ofxStableDiffusionModelInfo ofxStableDiffusion::getModelInfo(const std::string& modelPath) {
	return modelManager.extractModelInfo(modelPath);
}

std::vector<ofxStableDiffusionModelInfo> ofxStableDiffusion::getCachedModels() const {
	return modelManager.getCachedModels();
}

bool ofxStableDiffusion::preloadModel(const std::string& modelPath, std::string& errorMessage) {
	ofxStableDiffusionModelInfo info = modelManager.extractModelInfo(modelPath);
	return modelManager.preloadModel(info, errorMessage);
}

void ofxStableDiffusion::clearModelCache() {
	modelManager.clearCache();
}

void ofxStableDiffusion::setModelCacheSize(uint64_t maxBytes) {
	modelManager.setMaxCacheSize(maxBytes);
}

void ofxStableDiffusion::setMaxCachedModels(int count) {
	modelManager.setMaxCachedModels(count);
}

void ofxStableDiffusion::setProfilingEnabled(bool enabled) {
	performanceProfiler.setEnabled(enabled);
}

bool ofxStableDiffusion::isProfilingEnabled() const {
	return performanceProfiler.isEnabled();
}

ofxStableDiffusionPerformanceStats ofxStableDiffusion::getPerformanceStats() const {
	return performanceProfiler.getStats();
}

ofxStableDiffusionProfileEntry ofxStableDiffusion::getPerformanceEntry(const std::string& name) const {
	return performanceProfiler.getEntry(name);
}

void ofxStableDiffusion::resetProfiling() {
	performanceProfiler.reset();
}

void ofxStableDiffusion::printPerformanceSummary() const {
	performanceProfiler.printSummary();
}

std::vector<std::string> ofxStableDiffusion::getPerformanceBottlenecks(float thresholdPercent) const {
	return performanceProfiler.getBottlenecks(thresholdPercent);
}

std::string ofxStableDiffusion::exportPerformanceJSON() const {
	return performanceProfiler.toJSON();
}

std::string ofxStableDiffusion::exportPerformanceCSV() const {
	return performanceProfiler.toCSV();
}

void ofxStableDiffusion::addControlNet(const ofxStableDiffusionControlNet& controlNet) {
	std::lock_guard<std::mutex> lock(stateMutex);
	controlNets.push_back(controlNet);
}

void ofxStableDiffusion::clearControlNets() {
	std::lock_guard<std::mutex> lock(stateMutex);
	controlNets.clear();
}

std::vector<ofxStableDiffusionControlNet> ofxStableDiffusion::getControlNets() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return controlNets;
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
	std::string targetDir;
	{
		std::lock_guard<std::mutex> lock(stateMutex);
		targetDir = embedDirCStr;
	}
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

void ofxStableDiffusion::setNativeLoggingEnabled(bool enabled) {
	g_sdLoggingEnabled.store(enabled);
}

bool ofxStableDiffusion::isNativeLoggingEnabled() const {
	return g_sdLoggingEnabled.load();
}

void ofxStableDiffusion::setNativeLogLevel(sd_log_level_t level) {
	g_sdMinLogLevel.store(static_cast<int>(level));
}

sd_log_level_t ofxStableDiffusion::getNativeLogLevel() const {
	return static_cast<sd_log_level_t>(g_sdMinLogLevel.load());
}

//--------------------------------------------------------------
void ofxStableDiffusion::newSdCtx(const ofxStableDiffusionContextSettings& settings) {
	configureContext(settings);
}

void ofxStableDiffusion::freeSdCtx() {
	if (thread.isThreadRunning()) {
		thread.waitForThread(true);
	}
	thread.clearContexts();
	std::lock_guard<std::mutex> lock(stateMutex);
	clearResolvedDefaultCachesNoLock();
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
	if (videoFrames_ <= 0) {
		activeTask = ofxStableDiffusionTask::ImageToVideo;
		setLastError(ofxStableDiffusionErrorCode::InvalidFrameCount, "Frame count must be positive");
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
	bool inverse,
	int channels) {
	sd_image_t image{
		static_cast<uint32_t>(width_),
		static_cast<uint32_t>(height_),
		static_cast<uint32_t>(channels > 0 ? channels : 1),
		img
	};
	return ::preprocess_canny(image, highThreshold, lowThreshold, weak, strong, inverse) ? img : nullptr;
}

bool ofxStableDiffusion::isGenerating() const {
	return thread.isThreadRunning();
}

bool ofxStableDiffusion::isBusy() const {
	return isGenerating() || isModelLoading;
}

bool ofxStableDiffusion::requestCancellation() {
	if (!isGenerating()) {
		return false;
	}
	thread.requestCancellation();
	return true;
}

bool ofxStableDiffusion::isCancellationRequested() const {
	return thread.isCancellationRequested();
}

bool ofxStableDiffusion::wasCancelled() const {
	std::lock_guard<std::mutex> lock(stateMutex);
	return lastOperationCancelled;
}

bool ofxStableDiffusion::matchesContextSettings(
	const ofxStableDiffusionContextSettings& settings) const {
	const ofxStableDiffusionContextSettings resolvedSettings =
		resolveContextModelPaths(settings);
	std::lock_guard<std::mutex> lock(stateMutex);
	return contextSettingsEquivalent(
		captureContextSettingsNoLock(),
		resolvedSettings);
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
	isTextToImage.store(task == ofxStableDiffusionTask::TextToImage, std::memory_order_relaxed);
	isImageToVideo.store(task == ofxStableDiffusionTask::ImageToVideo, std::memory_order_relaxed);
	clearLastError();
	clearOutputState();
	thread.userData = this;
	thread.resetCancellation();  // Reset cancellation flag for new task
	lastOperationCancelled = false;
	return true;
}

void ofxStableDiffusion::finishBackgroundTask(bool cancelled, const std::string& cancelMessage) {
	if (cancelled) {
		setLastError(
			ofxStableDiffusionErrorCode::Cancelled,
			cancelMessage.empty() ? "Operation cancelled" : cancelMessage);
		std::lock_guard<std::mutex> lock(stateMutex);
		lastOperationCancelled = true;
	}

	isModelLoading = false;
	isTextToImage.store(false, std::memory_order_relaxed);
	isImageToVideo.store(false, std::memory_order_relaxed);
	activeTask = ofxStableDiffusionTask::None;
}

void ofxStableDiffusion::clearResolvedDefaultCachesNoLock() {
	cachedResolvedDefaultSampleMethod = SAMPLE_METHOD_COUNT;
	cachedResolvedDefaultScheduler = SCHEDULER_COUNT;
	if (cachedResolvedSchedulersBySampleMethod.size() != static_cast<std::size_t>(SAMPLE_METHOD_COUNT)) {
		cachedResolvedSchedulersBySampleMethod.assign(static_cast<std::size_t>(SAMPLE_METHOD_COUNT), SCHEDULER_COUNT);
	} else {
		std::fill(cachedResolvedSchedulersBySampleMethod.begin(), cachedResolvedSchedulersBySampleMethod.end(), SCHEDULER_COUNT);
	}
}

void ofxStableDiffusion::refreshResolvedDefaultCachesNoLock(sd_ctx_t* ctx) {
	clearResolvedDefaultCachesNoLock();
	if (ctx == nullptr) {
		return;
	}
	cachedResolvedDefaultSampleMethod =
		ofxStableDiffusionNativeAdapter::resolveSampleMethod(ctx, SAMPLE_METHOD_COUNT);
	cachedResolvedDefaultScheduler =
		ofxStableDiffusionNativeAdapter::resolveScheduler(
			ctx,
			cachedResolvedDefaultSampleMethod,
			SCHEDULER_COUNT);
	for (int i = 0; i < static_cast<int>(SAMPLE_METHOD_COUNT); ++i) {
		const auto sampleMethod = static_cast<sample_method_t>(i);
		cachedResolvedSchedulersBySampleMethod[static_cast<std::size_t>(i)] =
			ofxStableDiffusionNativeAdapter::resolveScheduler(
				ctx,
				sampleMethod,
				SCHEDULER_COUNT);
	}
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
	settings.rngType = rngType;
	settings.schedule = schedule;
	settings.prediction = prediction;
	settings.loraApplyMode = loraApplyMode;
	settings.keepClipOnCpu = keepClipOnCpu;
	settings.keepControlNetCpu = keepControlNetCpu;
	settings.keepVaeOnCpu = keepVaeOnCpu;
	settings.offloadParamsToCpu = offloadParamsToCpu;
	settings.flashAttn = flashAttn;
	settings.diffusionFlashAttn = diffusionFlashAttn;
	settings.enableMmap = enableMmap;
	return settings;
}

ofxStableDiffusionUpscalerSettings ofxStableDiffusion::captureUpscalerSettingsNoLock() const {
	return {esrganPath, cachedUpscalerSettings.nThreads, cachedUpscalerSettings.weightType, esrganMultiplier, isESRGAN};
}

void ofxStableDiffusion::applyContextSettings(const ofxStableDiffusionContextSettings& settings) {
	const ofxStableDiffusionContextSettings resolvedSettings = resolveContextModelPaths(settings);
	std::lock_guard<std::mutex> lock(stateMutex);
	clearResolvedDefaultCachesNoLock();
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
	rngType = resolvedSettings.rngType;
	schedule = resolvedSettings.schedule;
	prediction = resolvedSettings.prediction;
	loraApplyMode = resolvedSettings.loraApplyMode;
	keepClipOnCpu = resolvedSettings.keepClipOnCpu;
	keepControlNetCpu = resolvedSettings.keepControlNetCpu;
	keepVaeOnCpu = resolvedSettings.keepVaeOnCpu;
	offloadParamsToCpu = resolvedSettings.offloadParamsToCpu;
	flashAttn = resolvedSettings.flashAttn;
	diffusionFlashAttn = resolvedSettings.diffusionFlashAttn;
	enableMmap = resolvedSettings.enableMmap;
}

bool ofxStableDiffusion::applyImageRequest(const ofxStableDiffusionImageRequest& request) {
	stableDiffusionThread::ImageTaskData taskData;
	bool mergeFailed = false;
	std::string mergeError;
	float mergedStrength = request.controlStrength;

	try {
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

		bool hasControl = false;
		if (!request.controlNets.empty()) {
			if (!mergeControlNets(request.controlNets, taskData.controlImage, mergedStrength, mergeError)) {
				mergeFailed = true;
			} else {
				taskData.request.controlCond = &taskData.controlImage.image;
				taskData.request.controlStrength = mergedStrength;
				hasControl = true;
			}
		}

		if (!hasControl && request.controlCond != nullptr) {
			taskData.controlImage.assign(*request.controlCond);
			taskData.request.controlCond = &taskData.controlImage.image;
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
		controlCond = nullptr;
		controlStrength = taskData.request.controlStrength;
		styleStrength = request.styleStrength;
		normalizeInput = request.normalizeInput;
		inputIdImagesPath = request.inputIdImagesPath;
		loras = request.loras;
	} catch (const std::exception& e) {
		setLastError(ofxStableDiffusionErrorCode::Unknown,
			std::string("Exception while preparing image request: ") + e.what());
		return false;
	} catch (...) {
		setLastError(ofxStableDiffusionErrorCode::Unknown,
			"Unknown exception while preparing image request");
		return false;
	}

	if (mergeFailed) {
		const std::string message = mergeError.empty() ?
			"Failed to merge ControlNet inputs" :
			mergeError;
		setLastError(ofxStableDiffusionErrorCode::InvalidParameter, message);
		return false;
	}

	try {
		thread.prepareImageTask(taskData);
	} catch (const std::exception& e) {
		setLastError(ofxStableDiffusionErrorCode::Unknown,
			std::string("Exception while starting image task: ") + e.what());
		return false;
	} catch (...) {
		setLastError(ofxStableDiffusionErrorCode::Unknown,
			"Unknown exception while starting image task");
		return false;
	}

	return true;
}

void ofxStableDiffusion::applyVideoRequest(const ofxStableDiffusionVideoRequest& request) {
	stableDiffusionThread::VideoTaskData taskData;

	try {
		std::lock_guard<std::mutex> lock(stateMutex);
		taskData.task = activeTask;
		taskData.contextSettings = captureContextSettingsNoLock();
		taskData.upscalerSettings = captureUpscalerSettingsNoLock();
		taskData.request = request;
		loadedInputImage.assign(request.initImage);
		inputImage = loadedInputImage.image;
		taskData.initImage.assign(inputImage);
		taskData.endImage.assign(request.endImage);
		taskData.controlFrames.clear();
		taskData.controlFrames.reserve(request.controlFrames.size());
		for (const auto& frame : request.controlFrames) {
			stableDiffusionThread::OwnedImage ownedFrame;
			if (ownedFrame.assign(frame)) {
				taskData.controlFrames.push_back(std::move(ownedFrame));
			}
		}
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
	} catch (const std::exception& e) {
		setLastError(ofxStableDiffusionErrorCode::Unknown,
			std::string("Exception while preparing video request: ") + e.what());
		return;
	} catch (...) {
		setLastError(ofxStableDiffusionErrorCode::Unknown,
			"Unknown exception while preparing video request");
		return;
	}

	try {
		thread.prepareVideoTask(taskData);
	} catch (const std::exception& e) {
		setLastError(ofxStableDiffusionErrorCode::Unknown,
			std::string("Exception while starting video task: ") + e.what());
	} catch (...) {
		setLastError(ofxStableDiffusionErrorCode::Unknown,
			"Unknown exception while starting video task");
	}
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
	const ofxStableDiffusionCapabilities capabilities = getCapabilities();
	if (capabilities.videoRequiresInputImage && request.initImage.data == nullptr) {
		setLastError(ofxStableDiffusionErrorCode::MissingInputImage, "Video generation requires an input image");
		return false;
	}
	if (request.hasAnimation() && isNativeWanVideoFamily(capabilities.modelFamily)) {
		setLastError(
			ofxStableDiffusionErrorCode::InvalidParameter,
			"Wrapper image-sequence animation is not supported for native Wan video models. Use native video diffusion settings instead.");
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
	lastResolvedVideoRequestSummary.clear();
	lastResolvedVideoCliCommand.clear();
}

void ofxStableDiffusion::setLastError(const std::string& errorMessage, ofxStableDiffusionErrorCode code) {
	{
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

	if (!errorMessage.empty()) {
		ofLogError("ofxStableDiffusion") << errorMessage;
	}
}

void ofxStableDiffusion::setLastError(ofxStableDiffusionErrorCode code, const std::string& errorMessage) {
	setLastError(errorMessage, code);
}

void ofxStableDiffusion::setLastResolvedVideoRequestSummary(const std::string& summary) {
	std::lock_guard<std::mutex> lock(stateMutex);
	lastResolvedVideoRequestSummary = summary;
}

void ofxStableDiffusion::setLastResolvedVideoCliCommand(const std::string& command) {
	std::lock_guard<std::mutex> lock(stateMutex);
	lastResolvedVideoCliCommand = command;
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

	std::vector<ofxStableDiffusionImageScore> scores;
	try {
		scores = rankCallback(request, frames);
	} catch (const std::exception& e) {
		ofLogWarning("ofxStableDiffusion") << "Image rank callback threw: " << e.what();
	} catch (...) {
		ofLogWarning("ofxStableDiffusion") << "Image rank callback threw an unknown exception";
	}
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
			// NOTE: const_cast required for sd_image_t compatibility; this creates read-only views
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
