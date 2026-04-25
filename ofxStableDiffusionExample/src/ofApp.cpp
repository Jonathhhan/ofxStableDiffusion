#include "ofApp.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace {

const char* nativeLogLevelLabels[5] = {"Debug", "Info", "Warn", "Error", "Off"};
std::string lowerCopy(const std::string& value) {
	std::string lowered = value;
	std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	return lowered;
}

bool containsKeyword(const std::string& haystack, const std::string& needle) {
	return haystack.find(needle) != std::string::npos;
}

std::string quoteCliArg(const std::string& value) {
	std::string escaped = "\"";
	for (char c : value) {
		if (c == '"') {
			escaped += "\\\"";
		} else {
			escaped += c;
		}
	}
	escaped += "\"";
	return escaped;
}

std::string formatCliFloat(float value) {
	std::ostringstream stream;
	stream.setf(std::ios::fixed);
	stream.precision(3);
	stream << value;
	std::string formatted = stream.str();
	while (!formatted.empty() && formatted.back() == '0') {
		formatted.pop_back();
	}
	if (!formatted.empty() && formatted.back() == '.') {
		formatted.pop_back();
	}
	return formatted.empty() ? "0" : formatted;
}

const char* sampleMethodToCliName(sample_method_t method) {
	switch (method) {
	case EULER_A_SAMPLE_METHOD: return "euler_a";
	case EULER_SAMPLE_METHOD: return "euler";
	case HEUN_SAMPLE_METHOD: return "heun";
	case DPM2_SAMPLE_METHOD: return "dpm2";
	case DPMPP2S_A_SAMPLE_METHOD: return "dpm++2s_a";
	case DPMPP2M_SAMPLE_METHOD: return "dpm++2m";
	case DPMPP2Mv2_SAMPLE_METHOD: return "dpm++2mv2";
	case LCM_SAMPLE_METHOD: return "lcm";
	default:
		return "euler";
	}
}

const char* schedulerToCliName(scheduler_t scheduler) {
	switch (scheduler) {
	case DISCRETE_SCHEDULER: return "discrete";
	case KARRAS_SCHEDULER: return "karras";
	case EXPONENTIAL_SCHEDULER: return "exponential";
	case AYS_SCHEDULER: return "ays";
	case GITS_SCHEDULER: return "gits";
	case SGM_UNIFORM_SCHEDULER: return "sgm_uniform";
	case SIMPLE_SCHEDULER: return "simple";
	case SMOOTHSTEP_SCHEDULER: return "smoothstep";
	case KL_OPTIMAL_SCHEDULER: return "kl_optimal";
	case LCM_SCHEDULER: return "lcm";
	case BONG_TANGENT_SCHEDULER: return "bong_tangent";
	case SCHEDULER_COUNT:
	default:
		return "MODEL_DEFAULT";
	}
}

const rng_type_t rngTypeOptions[3] = {
	STD_DEFAULT_RNG,
	CUDA_RNG,
	CPU_RNG
};

sd_cache_mode_t cacheModeFromLabel(const std::string& label) {
	if (label == "easycache") return SD_CACHE_EASYCACHE;
	if (label == "ucache") return SD_CACHE_UCACHE;
	if (label == "dbcache") return SD_CACHE_DBCACHE;
	if (label == "taylorseer") return SD_CACHE_TAYLORSEER;
	if (label == "cache-dit") return SD_CACHE_CACHE_DIT;
	if (label == "spectrum") return SD_CACHE_SPECTRUM;
	return SD_CACHE_DISABLED;
}

bool cacheModeUsesThresholdWindow(sd_cache_mode_t mode) {
	return mode == SD_CACHE_EASYCACHE || mode == SD_CACHE_UCACHE;
}

std::string defaultControlFramesReadmeText() {
	return
		"Default control-frame folder for ofxStableDiffusion native video diffusion.\n"
		"\n"
		"Put an ordered image sequence here, for example:\n"
		"  control_0000.png\n"
		"  control_0001.png\n"
		"  control_0002.png\n"
		"\n"
		"Supported formats: png, jpg, jpeg, bmp, webp\n"
		"\n"
		"The example app sorts filenames and uses them as per-frame native control inputs.\n"
		"You can also fill this folder from the GUI with the current generated video clip.\n";
}

bool loadImageFile(
	const std::string& path,
	int width,
	int height,
	ofImage& targetImage,
	ofPixels& targetPixels,
	sd_image_t& targetSdImage) {
	ofFile file(path);
	const std::string extension = ofToUpper(file.getExtension());
	if (extension != "JPG" &&
		extension != "JPEG" &&
		extension != "PNG" &&
		extension != "BMP" &&
		extension != "WEBP") {
		return false;
	}

	if (!targetImage.load(path)) {
		return false;
	}
	targetImage.resize(width, height);
	targetPixels = targetImage.getPixels();
	targetSdImage = {
		static_cast<uint32_t>(targetPixels.getWidth()),
		static_cast<uint32_t>(targetPixels.getHeight()),
		static_cast<uint32_t>(targetPixels.getNumChannels()),
		targetPixels.getData()
	};
	return true;
}

float computeBrightnessScore(const ofPixels& pixels) {
	if (!pixels.isAllocated() || pixels.getNumChannels() < 3) {
		return 0.0f;
	}
	const int width = pixels.getWidth();
	const int height = pixels.getHeight();
	const std::size_t pixelCount =
		static_cast<std::size_t>(width) *
		static_cast<std::size_t>(height);
	if (pixelCount == 0) {
		return 0.0f;
	}
	const unsigned char* data = pixels.getData();
	const std::size_t channels =
		static_cast<std::size_t>(pixels.getNumChannels());
	double brightnessSum = 0.0;
	double colorfulnessSum = 0.0;
	std::size_t samples = 0;
	const std::size_t targetSamples = 4096;
	const std::size_t stride = std::max<std::size_t>(
		1,
		static_cast<std::size_t>(std::sqrt(
			std::max<double>(
				1.0,
				static_cast<double>(pixelCount) /
					static_cast<double>(targetSamples)))));
	for (int y = 0; y < height; y += static_cast<int>(stride)) {
		const std::size_t rowOffset =
			static_cast<std::size_t>(y) *
			static_cast<std::size_t>(width) *
			channels;
		for (int x = 0; x < width; x += static_cast<int>(stride)) {
			const std::size_t offset =
				rowOffset +
				static_cast<std::size_t>(x) * channels;
			const float r = static_cast<float>(data[offset + 0]) / 255.0f;
			const float g = static_cast<float>(data[offset + 1]) / 255.0f;
			const float b = static_cast<float>(data[offset + 2]) / 255.0f;
			const float maxChannel = std::max(r, std::max(g, b));
			const float minChannel = std::min(r, std::min(g, b));
			brightnessSum += (r + g + b) / 3.0f;
			colorfulnessSum += (maxChannel - minChannel);
			++samples;
		}
	}
	if (samples == 0) {
		return 0.0f;
	}
	const float meanBrightness =
		static_cast<float>(brightnessSum / static_cast<double>(samples));
	const float meanColorfulness =
		static_cast<float>(colorfulnessSum / static_cast<double>(samples));
	return 0.65f * meanBrightness + 0.35f * meanColorfulness;
}

float computeHeuristicPromptScore(const std::string& rankingPrompt, const ofPixels& pixels) {
	float score = computeBrightnessScore(pixels);
	const std::string lowered = lowerCopy(rankingPrompt);
	if (lowered.empty()) {
		return score;
	}

	if (containsKeyword(lowered, "bright") || containsKeyword(lowered, "light")) {
		score += 0.15f;
	}
	if (containsKeyword(lowered, "dark") || containsKeyword(lowered, "moody")) {
		score -= 0.15f;
	}
	if (containsKeyword(lowered, "color") || containsKeyword(lowered, "vivid") ||
		containsKeyword(lowered, "neon") || containsKeyword(lowered, "futur")) {
		score += 0.1f;
	}
	if (containsKeyword(lowered, "soft") || containsKeyword(lowered, "muted")) {
		score -= 0.05f;
	}
	return score;
}

int InputTextResizeCallback(ImGuiInputTextCallbackData* data) {
	if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
		std::string* str = static_cast<std::string*>(data->UserData);
		IM_ASSERT(data->Buf == str->c_str());
		str->resize(data->BufTextLen);
		data->Buf = str->data();
	}
	return 0;
}

bool InputTextMultilineString(
	const char* label,
	std::string* value,
	const ImVec2& size = ImVec2(0, 0),
	ImGuiInputTextFlags flags = 0) {
	IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
	return ImGui::InputTextMultiline(
		label,
		(char*)value->c_str(),
		value->capacity() + 1,
		size,
		flags | ImGuiInputTextFlags_CallbackResize,
		InputTextResizeCallback,
		value);
}

} // namespace

std::vector<std::pair<std::string, std::string>> ofApp::listEmbeddingFiles() const {
	std::vector<std::pair<std::string, std::string>> out;
	if (embedDir.empty()) return out;
	ofDirectory dir(embedDir);
	if (!dir.exists()) return out;
	dir.allowExt("pt");
	dir.allowExt("ckpt");
	dir.allowExt("safetensors");
	dir.allowExt("bin");
	dir.allowExt("gguf");
	dir.listDir();
	for (std::size_t i = 0; i < dir.size(); ++i) {
		const ofFile& f = dir.getFile(static_cast<int>(i));
		if (!f.isFile()) continue;
		out.emplace_back(f.getBaseName(), f.getAbsolutePath());
	}
	return out;
}

bool ofApp::loadImageIntoSlot(
	const std::string& dialogTitle,
	ofImage& targetImage,
	ofPixels& targetPixels,
	sd_image_t& targetSdImage,
	std::string& targetName) {
	ofFileDialogResult result = ofSystemLoadDialog(dialogTitle, false, "");
	if (!result.bSuccess) {
		return false;
	}
	if (!loadImageFile(result.getPath(), width, height, targetImage, targetPixels, targetSdImage)) {
		return false;
	}
	targetName = result.getName();
	return true;
}

bool ofApp::loadVideoControlFramesFromFolder(const std::string& folderPath) {
	ofDirectory dir(folderPath);
	if (!dir.exists()) {
		return false;
	}

	dir.allowExt("png");
	dir.allowExt("jpg");
	dir.allowExt("jpeg");
	dir.allowExt("bmp");
	dir.allowExt("webp");
	dir.listDir();
	dir.sort();

	std::vector<ofImage> loadedImages;
	std::vector<ofPixels> loadedPixels;
	std::vector<sd_image_t> loadedFrames;
	loadedImages.reserve(dir.size());
	loadedPixels.reserve(dir.size());
	loadedFrames.reserve(dir.size());

	for (std::size_t i = 0; i < dir.size(); ++i) {
		const ofFile& file = dir.getFile(static_cast<int>(i));
		if (!file.isFile()) {
			continue;
		}

		ofImage frameImage;
		ofPixels framePixels;
		sd_image_t frameView{0, 0, 0, nullptr};
		if (!loadImageFile(file.getAbsolutePath(), width, height, frameImage, framePixels, frameView)) {
			continue;
		}

		loadedImages.push_back(std::move(frameImage));
		loadedPixels.push_back(std::move(framePixels));
		loadedFrames.push_back(frameView);
	}

	if (loadedFrames.empty()) {
		return false;
	}

	videoControlFrameImages = std::move(loadedImages);
	videoControlFramePixels = std::move(loadedPixels);
	videoControlFrames = std::move(loadedFrames);
	videoControlFramesPath = folderPath;
	videoControlPreviewIndex = 0;
	return true;
}

void ofApp::clearVideoControlFrames() {
	videoControlFrameImages.clear();
	videoControlFramePixels.clear();
	videoControlFrames.clear();
	videoControlFramesPath = defaultVideoControlFramesPath;
	videoControlPreviewIndex = 0;
}

std::vector<std::pair<std::string, std::string>> ofApp::listLoraFiles() const {
	std::vector<std::pair<std::string, std::string>> out;
	if (loraModelDir.empty()) return out;
	ofDirectory dir(loraModelDir);
	if (!dir.exists()) return out;
	dir.allowExt("safetensors");
	dir.allowExt("ckpt");
	dir.allowExt("pt");
	dir.allowExt("bin");
	dir.allowExt("gguf");
	dir.listDir();
	for (std::size_t i = 0; i < dir.size(); ++i) {
		const ofFile& f = dir.getFile(static_cast<int>(i));
		if (!f.isFile()) continue;
		out.emplace_back(f.getBaseName(), f.getAbsolutePath());
	}
	return out;
}

void ofApp::loadAllLoras(float strength) {
	loras.clear();
	for (const auto& entry : listLoraFiles()) {
		ofxStableDiffusionLora l;
		l.path = entry.second;
		l.strength = strength;
		l.isHighNoise = false;
		loras.push_back(std::move(l));
	}
	stableDiffusion.setLoras(loras);
}

void ofApp::clearLoras() {
	loras.clear();
	stableDiffusion.setLoras(loras);
}

void ofApp::setupHoloscanBridge() {
	holoscanPrompt = prompt;
	holoscanNegativePrompt = negativePrompt;
	ofxStableDiffusionHoloscanSettings settings;
	settings.enabled = true;
	settings.useEventScheduler = true;
	settings.workerThreads = 2;
	holoscanBridgeEnabled = holoscanBridge.setup(&stableDiffusion, settings);
	if (holoscanBridgeEnabled) {
	#if defined(TARGET_LINUX)
		holoscanStatus = holoscanBridge.isHoloscanAvailable()
			? "Holoscan bridge ready. Native Linux runtime detected."
			: "Holoscan bridge ready in fallback mode. Install the Linux Holoscan SDK later to replace the addon fallback lane.";
	#else
		holoscanStatus = "Holoscan is Linux-only for now. The bridge stays on the addon fallback lane on this platform.";
	#endif
	} else {
		holoscanStatus = holoscanBridge.getLastError();
	}
}

void ofApp::drawHoloscanBridgeSection() {
	if (!ImGui::TreeNodeEx("Holoscan Bridge", ImGuiStyleVar_WindowPadding)) {
		return;
	}

	ImGui::Dummy(ImVec2(0, 10));
	ImGui::TextWrapped(
		"This MVP keeps a live frame -> conditioning -> diffusion -> preview lane behind an optional Holoscan-oriented bridge.");
	ImGui::Dummy(ImVec2(0, 10));
	ImGui::Text("Status: %s", holoscanStatus.empty() ? "Idle" : holoscanStatus.c_str());
	ImGui::Dummy(ImVec2(0, 10));
	ImGui::Checkbox("Use current prompt fields", &holoscanBridgeUseCurrentPrompts);
	if (holoscanBridgeUseCurrentPrompts) {
		holoscanPrompt = prompt;
		holoscanNegativePrompt = negativePrompt;
	}
	InputTextMultilineString(
		"Bridge Prompt",
		&holoscanPrompt,
		ImVec2(512, 70),
		ImGuiInputTextFlags_CallbackResize);
	ImGui::Dummy(ImVec2(0, 10));
	InputTextMultilineString(
		"Bridge Negative",
		&holoscanNegativePrompt,
		ImVec2(512, 50),
		ImGuiInputTextFlags_CallbackResize);
	ImGui::Dummy(ImVec2(0, 10));
	if (!holoscanBridgeRunning) {
		if (ImGui::Button("Start Bridge")) {
			holoscanBridge.submitPrompt(holoscanPrompt, holoscanNegativePrompt);
			holoscanBridgeRunning = holoscanBridge.startImagePipeline();
			holoscanStatus = holoscanBridgeRunning
				? "Bridge pipeline started."
				: holoscanBridge.getLastError();
		}
	} else {
		if (ImGui::Button("Stop Bridge")) {
			holoscanBridge.stop();
			holoscanBridgeRunning = false;
			holoscanStatus = "Bridge pipeline stopped.";
		}
	}
	ImGui::SameLine(0, 10);
	if (ImGui::Button("Send Loaded Image")) {
		holoscanBridge.submitPrompt(holoscanPrompt, holoscanNegativePrompt);
		if (pixels.isAllocated()) {
			holoscanBridge.submitFrame(pixels, ofGetElapsedTimef(), imageName.empty() ? "loaded-image" : imageName);
			holoscanStatus = "Queued loaded image for bridge processing.";
		} else {
			holoscanStatus = "Load an input image first.";
		}
	}
	ImGui::Dummy(ImVec2(0, 10));
	if (holoscanBridge.hasPreviewFrame()) {
		ImGui::Image(
			(ImTextureID)(uintptr_t)holoscanBridge.getPreviewTexture().getTextureData().textureID,
			ImVec2(192, 192));
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Text("Completed bridge frames: %d", holoscanCompletedFrames);
	}
	ImGui::TreePop();
}

//--------------------------------------------------------------
ofxStableDiffusionContextSettings ofApp::buildContextSettings() const {
	ofxStableDiffusionContextSettings settings;
	settings.modelPath = modelPath;
	settings.diffusionModelPath = diffusionModelPath;
	settings.clipLPath = clipLPath;
	settings.clipGPath = clipGPath;
	settings.t5xxlPath = t5xxlPath;
	settings.vaePath = vaePath;
	settings.taesdPath = taesdPath;
	settings.controlNetPath = controlNetPath;
	settings.loraModelDir = loraModelDir;
	settings.embedDir = embedDir;
	settings.stackedIdEmbedDir = stackedIdEmbedDir;
	settings.vaeDecodeOnly = vaeDecodeOnly;
	settings.vaeTiling = vaeTiling;
	settings.freeParamsImmediately = freeParamsImmediately;
	settings.nThreads = nThreads;
	settings.weightType = wType;
	settings.rngType = rngType;
	settings.schedule = schedule;
	settings.keepClipOnCpu = keepClipOnCpu;
	settings.keepControlNetCpu = keepControlNetCpu;
	settings.keepVaeOnCpu = keepVaeOnCpu;
	settings.offloadParamsToCpu = offloadParamsToCpu;
	settings.flashAttn = flashAttn;
	settings.diffusionFlashAttn = diffusionFlashAttn;
	return settings;
}

//--------------------------------------------------------------
ofxStableDiffusionVideoRequest ofApp::buildVideoRequest() const {
	ofxStableDiffusionVideoRequest request;
	const ofxStableDiffusionCapabilities capabilities = stableDiffusion.getCapabilities();
	request.initImage =
		(currentVideoModeUsesInputImage() && inputImage.data != nullptr) ?
			inputImage :
			sd_image_t{0, 0, 0, nullptr};
	request.endImage =
		(currentVideoModeUsesInputImage() && capabilities.videoEndFrame && useEndFrame && endInputImage.data != nullptr) ?
			endInputImage :
			sd_image_t{0, 0, 0, nullptr};
	if (useVideoControlFrames && !videoControlFrames.empty()) {
		request.controlFrames = videoControlFrames;
	}
	request.prompt = prompt;
	request.negativePrompt = negativePrompt;
	request.clipSkip = clipSkip;
	request.width = width;
	request.height = height;
	request.frameCount = videoFrames;
	request.fps = videoFps;
	request.cfgScale = cfgScale;
	request.guidance = guidance;
	request.sampleMethod = sampleMethodEnum;
	request.sampleSteps = sampleSteps;
	request.eta = eta;
	request.flowShift = flowShift;
	request.useHighNoiseOverrides = useHighNoiseOverrides;
	if (request.useHighNoiseOverrides) {
		request.highNoiseCfgScale = highNoiseCfgScale;
		request.highNoiseGuidance = highNoiseGuidance;
		request.highNoiseSampleMethod = highNoiseSampleMethodEnum;
		request.highNoiseSampleSteps = highNoiseSampleSteps;
		request.highNoiseEta = highNoiseEta;
		request.highNoiseFlowShift = highNoiseFlowShift;
	}
	request.strength = strength;
	request.seed = seed;
	request.moeBoundary = videoMoeBoundary;
	request.vaceStrength = vaceStrength;
	request.cache.mode = cacheModeFromLabel(videoCacheMode);
	if (cacheModeUsesThresholdWindow(request.cache.mode)) {
		request.cache.reuse_threshold = videoCacheThreshold;
		request.cache.start_percent = std::clamp(videoCacheStartPercent, 0.0f, 0.99f);
		request.cache.end_percent = std::clamp(
			std::max(videoCacheEndPercent, request.cache.start_percent + 0.01f),
			0.01f,
			1.0f);
	}
	request.mode = ofxStableDiffusionVideoMode::Standard;
	request.loras = loras;
	if (enablePromptInterpolation && !promptB.empty() && videoFrames > 1) {
		request.animationSettings.enablePromptInterpolation = true;
		request.animationSettings.promptInterpolationMode = interpolationModeEnum;
		request.animationSettings.promptKeyframes = {
			{0, prompt},
			{videoFrames - 1, promptB}
		};
	}
	if (useSeedSequence) {
		request.animationSettings.useSeedSequence = true;
		request.animationSettings.seedIncrement = seedIncrement;
	}
	return request;
}

//--------------------------------------------------------------
std::string ofApp::buildEquivalentSdCliCommand() const {
	std::ostringstream command;
	command << "sd-cli.exe";
	if (isImageToVideo) {
		const std::string resolvedVideoCliCommand =
			stableDiffusion.getLastResolvedVideoCliCommand();
		if (!resolvedVideoCliCommand.empty()) {
			return resolvedVideoCliCommand;
		}
		const ofxStableDiffusionVideoRequest request = buildVideoRequest();
		const bool usesModelDefaultSampleMethod = (request.sampleMethod == SAMPLE_METHOD_COUNT);
		command << " -M vid_gen";
		if (!modelPath.empty()) {
			command << " --model " << quoteCliArg(modelPath);
		}
		if (!diffusionModelPath.empty()) {
			command << " --diffusion-model " << quoteCliArg(diffusionModelPath);
		}
		if (!vaePath.empty()) {
			command << " --vae " << quoteCliArg(vaePath);
		}
		if (!t5xxlPath.empty()) {
			command << " --t5xxl " << quoteCliArg(t5xxlPath);
		}
		if (!clipLPath.empty()) {
			command << " --clip_l " << quoteCliArg(clipLPath);
		}
		if (!clipGPath.empty()) {
			command << " --clip_g " << quoteCliArg(clipGPath);
		}
		if (rngType != CUDA_RNG) {
			command << " --rng " << sd_rng_type_name(rngType);
		}
		command << " -p " << quoteCliArg(prompt);
		if (!negativePrompt.empty()) {
			command << " -n " << quoteCliArg(negativePrompt);
		}
		command << " --cfg-scale " << formatCliFloat(request.cfgScale);
		command << " --guidance " << formatCliFloat(request.guidance);
		if (schedule != SCHEDULER_COUNT) {
			command << " --scheduler " << schedulerToCliName(schedule);
		}
		if (!usesModelDefaultSampleMethod) {
			command << " --sampling-method " << sampleMethodToCliName(request.sampleMethod);
		}
		command << " --steps " << request.sampleSteps;
		if (request.useHighNoiseOverrides) {
			command << " --high-noise-steps " << request.highNoiseSampleSteps;
			command << " --high-noise-cfg-scale " << formatCliFloat(request.highNoiseCfgScale);
			command << " --high-noise-guidance " << formatCliFloat(request.highNoiseGuidance);
		}
		command << " -W " << width;
		command << " -H " << height;
		command << " --video-frames " << videoFrames;
		if (std::isfinite(request.eta)) {
			command << " --eta " << formatCliFloat(eta);
		}
		if (std::isfinite(request.flowShift)) {
			command << " --flow-shift " << formatCliFloat(flowShift);
		}
		if (seed >= 0) {
			command << " --seed " << seed;
		}
		command << " --moe-boundary " << formatCliFloat(request.moeBoundary);
		if (cacheModeFromLabel(videoCacheMode) != SD_CACHE_DISABLED) {
			command << " --cache-mode " << videoCacheMode;
			if (cacheModeUsesThresholdWindow(cacheModeFromLabel(videoCacheMode))) {
				command
					<< " --cache-option "
					<< quoteCliArg(
						"threshold=" + formatCliFloat(videoCacheThreshold) +
						",start=" + formatCliFloat(videoCacheStartPercent) +
						",end=" + formatCliFloat(videoCacheEndPercent));
			}
		}
		if (diffusionFlashAttn) {
			command << " --diffusion-fa";
		}
		if (flashAttn) {
			command << " --flash-attn";
		}
		if (offloadParamsToCpu) {
			command << " --offload-to-cpu";
		}
		if (keepVaeOnCpu) {
			command << " --vae-on-cpu";
		}
		if (keepClipOnCpu) {
			command << " --clip-on-cpu";
		}
		if (vaeTiling) {
			command << " --vae-tiling";
		}
		if (request.initImage.data != nullptr) {
			command << " --init-img <loaded input image path not tracked by example>";
		}
		if (request.endImage.data != nullptr) {
			command << " --end-img <loaded end frame path not tracked by example>";
		}
		if (!request.controlFrames.empty() && !videoControlFramesPath.empty()) {
			command << " --control-video " << quoteCliArg(videoControlFramesPath);
		}
		return command.str();
	}

	command << " [image-mode export not implemented yet]";
	return command.str();
}

//--------------------------------------------------------------
std::string ofApp::currentModelMenuLabel() const {
	if (!modelName.empty()) {
		return modelName;
	}
	const std::string activeModelPath = modelPath.empty() ? diffusionModelPath : modelPath;
	if (!activeModelPath.empty()) {
		return ofFilePath::getFileName(activeModelPath);
	}
	return "Backend";
}

//--------------------------------------------------------------
std::string ofApp::buildResolvedSampleMethodMenuLabel(sample_method_t requested) const {
	const std::string resolved = stableDiffusion.getResolvedSampleMethodName(requested);
	return resolved == "MODEL_DEFAULT" ? "Auto" : resolved;
}

//--------------------------------------------------------------
std::string ofApp::buildModelDefaultSampleMethodMenuLabel() const {
	return currentModelMenuLabel() + " default: " +
		buildResolvedSampleMethodMenuLabel(SAMPLE_METHOD_COUNT);
}

//--------------------------------------------------------------
std::string ofApp::buildResolvedSchedulerMenuLabel(
	sample_method_t requestedSampleMethod,
	scheduler_t requestedSchedule) const {
	const std::string resolved =
		stableDiffusion.getResolvedSchedulerName(requestedSampleMethod, requestedSchedule);
	return resolved == "MODEL_DEFAULT" ? "Auto" : resolved;
}

//--------------------------------------------------------------
std::string ofApp::buildModelDefaultSchedulerMenuLabel(sample_method_t requestedSampleMethod) const {
	return currentModelMenuLabel() + " default: " +
		buildResolvedSchedulerMenuLabel(requestedSampleMethod, SCHEDULER_COUNT);
}

//--------------------------------------------------------------
void ofApp::refreshModelContext() {
	modelName = ofFilePath::getFileName(modelPath.empty() ? diffusionModelPath : modelPath);
	stableDiffusion.newSdCtx(buildContextSettings());
	clampCurrentParametersToProfiles();
}

//--------------------------------------------------------------
bool ofApp::selectPath(
	const std::string& dialogTitle,
	std::string& targetPath,
	bool folderSelection) {
	ofFileDialogResult result = ofSystemLoadDialog(dialogTitle, folderSelection, "");
	if (!result.bSuccess) {
		return false;
	}

	targetPath = result.getPath();
	return true;
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxStableDiffusionExample");
	ofSetEscapeQuitsApp(false);
	ofSetWindowPosition((ofGetScreenWidth() - ofGetWindowWidth()) / 2, (ofGetScreenHeight() - ofGetWindowHeight()) / 2);
	ofDisableArbTex();
	modelPath = "";
	modelName = "";
	diffusionModelPath = "";
	clipLPath = "";
	clipGPath = "";
	t5xxlPath = "";
	controlNetPath = ""; // "data/models/controlnet/control_v11p_sd15_openpose_2.safetensors";
	embedDir = "";
	taesdPath = "";
	loraModelDir = ""; // "data/models/lora";
	vaePath = ""; // "data/models/vae/vae.safetensors";
	prompt = "animal with futuristic clothes"; // "man img, man with futuristic clothes";
	promptB = "animal with elegant retro-futuristic clothes";
	instruction = "edit the source image so the subject feels more futuristic while keeping the composition readable";
	rankingPrompt = "bright futuristic vivid cinematic";
	esrganPath = ""; // "data/models/esrgan/RealESRGAN_x4plus_anime_6B.pth"; // "data/models/esrgan/RealESRGAN_x4plus_anime_6B.pth";
	controlImagePath = ""; // "data/openpose.jpeg";
	stackedIdEmbedDir = ""; // "data/models/photomaker/photomaker-v1.safetensors";
	inputIdImagesPath = ""; // "data/photomaker_images/newton_man";
	keepClipOnCpu = false;
	keepControlNetCpu = false;
	keepVaeOnCpu = false;
	offloadParamsToCpu = false;
	flashAttn = false;
	diffusionFlashAttn = false;
	eta = 0.0f;
	flowShift = 5.0f;
	guidance = 3.5f;
	useHighNoiseOverrides = false;
	highNoiseSampleSteps = -1;
	highNoiseCfgScale = 7.0f;
	highNoiseGuidance = 3.5f;
	highNoiseEta = 0.0f;
	highNoiseFlowShift = 5.0f;
	styleStrength = 20;
	normalizeInput = true;
	width = 512;
	height = 512;
	cfgScale = 7.0f;
	sampleSteps = 20;
	clipSkip = -1;
	previewSize = batchCount = 4;
	selectedImage = 0;
	strength = 0.75f;
	controlStrength = 0.9;
	seed = -1;
	wType = SD_TYPE_F16;
	schedule = SCHEDULER_COUNT;
	rngType = CUDA_RNG;
	imageMode = "TextToImage";
	imageModeEnum = ofxStableDiffusionImageMode::TextToImage;
	selectionMode = "KeepOrder";
	selectionModeEnum = ofxStableDiffusionImageSelectionMode::KeepOrder;
	sampleMethod = "MODEL_DEFAULT";
	sampleMethodEnum = SAMPLE_METHOD_COUNT;
	highNoiseSampleMethod = sampleMethod;
	highNoiseSampleMethodEnum = sampleMethodEnum;
	videoCacheMode = "disabled";
	interpolationMode = "Smooth";
	promptIsEdited = true;
	negativePromptIsEdited = true;
	isTextToImage = true;
	isInstructImage = false;
	isImageToVideo = false;
	videoFrames = 6;
	videoFps = videoParameterProfile.defaultFps;
	videoMoeBoundary = 0.875f;
	videoCacheThreshold = 0.25f;
	videoCacheStartPercent = 0.2f;
	videoCacheEndPercent = 1.0f;
	vaceStrength = 1.0f;
	enablePromptInterpolation = false;
	useSeedSequence = false;
	useEndFrame = false;
	useVideoControlFrames = false;
	defaultVideoControlFramesPath = ofToDataPath("control_frames", true);
	ofDirectory defaultControlFrameDirectory(defaultVideoControlFramesPath);
	defaultControlFrameDirectory.create(true);
	const std::string defaultControlReadmePath =
		ofFilePath::join(defaultVideoControlFramesPath, "README.txt");
	if (!ofFile::doesFileExist(defaultControlReadmePath)) {
		ofBuffer readmeBuffer;
		readmeBuffer.set(defaultControlFramesReadmeText());
		ofBufferToFile(defaultControlReadmePath, readmeBuffer);
	}
	videoControlFramesPath = defaultVideoControlFramesPath;
	loadVideoControlFramesFromFolder(defaultVideoControlFramesPath);
	videoControlPreviewIndex = 0;
	seedIncrement = 1;
	isPlaying = false;
	currentFrame = 0;
	totalVideoFrames = 0;
	lastFrameTime = 0.0f;
	isFullScreen = false;
	isTAESD = false;
	isESRGAN = false;
	vaeDecodeOnly = false;
	vaeTiling = false;
	freeParamsImmediately = true;
	nThreads = 8;
	esrganMultiplier = 4;
	textureVector.resize(static_cast<std::size_t>(maxPreviewTextures));
	allocate();
	gui.setup(nullptr, true, ImGuiConfigFlags_None, true);
	applySelectedImageMode(imageModeEnum);
	stableDiffusion.setImageSelectionMode(selectionModeEnum);
	stableDiffusion.setNativeLoggingEnabled(true);
	stableDiffusion.setNativeLogLevel(SD_LOG_DEBUG);
	configureExampleRanker();
	setupHoloscanBridge();

	// Initial embedding enumeration (best-effort).
	auto embeds = stableDiffusion.listEmbeddings();
}

//--------------------------------------------------------------
void ofApp::update() {
	if (holoscanBridgeRunning) {
		holoscanBridge.update();
		auto completedFrames = holoscanBridge.consumeFinishedImages();
		if (!completedFrames.empty()) {
			holoscanCompletedFrames += static_cast<int>(completedFrames.size());
			holoscanStatus =
				"Completed " + ofToString(holoscanCompletedFrames) + " bridge frames.";
		}
		const std::string bridgeError = holoscanBridge.getLastError();
		if (!bridgeError.empty()) {
			holoscanStatus = bridgeError;
		}
	}

	if (stableDiffusion.isDiffused()) {
		if (isImageToVideo) {
			totalVideoFrames = stableDiffusion.getOutputCount();
			if (totalVideoFrames > static_cast<int>(textureVector.size())) {
				textureVector.resize(static_cast<std::size_t>(totalVideoFrames));
			}
			for (int i = 0; i < totalVideoFrames; i++) {
				ofPixels framePixels;
				if (!stableDiffusion.copyVideoFramePixels(i, framePixels) || !framePixels.isAllocated()) {
					continue;
				}
				const int frameWidth = framePixels.getWidth();
				const int frameHeight = framePixels.getHeight();
				const int frameChannels = framePixels.getNumChannels();
				if (frameWidth <= 0 || frameHeight <= 0 || frameChannels <= 0) {
					continue;
				}
				if (!textureVector[i].isAllocated() ||
					textureVector[i].getWidth() != frameWidth ||
					textureVector[i].getHeight() != frameHeight) {
					textureVector[i].allocate(frameWidth, frameHeight, frameChannels == 4 ? GL_RGBA : GL_RGB);
				}
				textureVector[i].loadData(framePixels);
			}
			previousSelectedImage = 0;
			currentFrame = 0;
			previewSize = totalVideoFrames;
			isPlaying = true;
			lastFrameTime = ofGetElapsedTimef();
		}
		else {
			const int outputCount = stableDiffusion.getOutputCount();
			if (outputCount > static_cast<int>(textureVector.size())) {
				textureVector.resize(static_cast<std::size_t>(outputCount));
			}
			for (int i = 0; i < outputCount; i++) {
				ofPixels imagePixels;
				if (!stableDiffusion.copyImagePixels(i, imagePixels) || !imagePixels.isAllocated()) {
					continue;
				}
				if (isESRGAN) {
					textureVector[i].loadData(
						imagePixels.getData(),
						width * esrganMultiplier,
						height * esrganMultiplier,
						GL_RGB);
				}
				else {
					textureVector[i].loadData(
						imagePixels.getData(),
						width,
						height,
						GL_RGB);
				}
			}
			previousSelectedImage = std::max(0, stableDiffusion.getSelectedImageIndex());
			previewSize = outputCount;
			totalVideoFrames = 0;
		}
		stableDiffusion.setDiffused(false);
	}
	if (isPlaying && totalVideoFrames > 0) {
		float now = ofGetElapsedTimef();
		const float fallbackFps = 10.0f;
		float frameInterval = (videoFps > 0) ? (1.0f / videoFps) : (1.0f / fallbackFps);
		if (now - lastFrameTime >= frameInterval) {
			currentFrame = (currentFrame + 1) % totalVideoFrames;
			previousSelectedImage = currentFrame;
			lastFrameTime = now;
		}
	}
	if (previewSize <= 0 || textureVector.empty()) {
		selectedImage = 0;
		previousSelectedImage = 0;
	} else {
		const int maxIndex = std::min(previewSize, static_cast<int>(textureVector.size())) - 1;
		selectedImage = std::clamp(selectedImage, 0, maxIndex);
		previousSelectedImage = std::clamp(previousSelectedImage, 0, maxIndex);
	}
	selectedImage = previousSelectedImage;
}

//--------------------------------------------------------------
void ofApp::draw() {
	struct Funcs {
		static int InputTextCallback(ImGuiInputTextCallbackData* data) {
			if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
				std::string* str = (std::string*)data->UserData;
				IM_ASSERT(data->Buf == str->c_str());
				str->resize(data->BufTextLen);
				data->Buf = (char*)str->c_str();
			}
			return 0;
		}
		static bool MyInputTextMultiline(const char* label, std::string* prompt, const ImVec2& size = ImVec2(0, 0), ImGuiInputTextFlags flags = 0) {
			IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
			return ImGui::InputTextMultiline(label, (char*)prompt->c_str(), prompt->capacity() + 1, size, flags | ImGuiInputTextFlags_CallbackResize, Funcs::InputTextCallback, (void*)prompt);
		}
	};
	ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse;
	static bool logOpenSettings{ true };
	const bool isBusy = stableDiffusion.isGenerating() || stableDiffusion.isModelLoading;
	const auto capabilities = stableDiffusion.getCapabilities();
	syncAutomaticMediaMode(capabilities);
	gui.begin();
	ImGui::StyleColorsDark();
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
	ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 10);
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
	ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(5, 0));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(0, 0));
	if (previewSize > 3) {
		previewWidth = width + 20;
	} else {
		previewWidth = width / 4 * previewSize + 20;
	}
	ImGui::SetNextWindowSizeConstraints(ImVec2(previewWidth, -1.f), ImVec2(previewWidth, -1.f));
	ImGui::SetNextWindowPos(ImVec2(20, 620), ImGuiCond_Once);
	ImGui::Begin("ofxStableDiffusion##foo0", NULL, flags);
	if (ImGui::TreeNodeEx("Image Preview", ImGuiStyleVar_WindowPadding | ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Dummy(ImVec2(0, 10));
		const int previewCount = std::min(previewSize, static_cast<int>(textureVector.size()));
		for (int i = 0; i < previewCount; i++) {
			if (i == previewSize - previewSize % 4 && i > 3) {
				ImGui::Indent(width / 8.f * (4 - previewSize % 4));
				ImGui::Image((ImTextureID)(uintptr_t)textureVector[i].getTextureData().textureID, ImVec2(width / 4, height / 4));
				ImGui::Indent(- width / 8.f * (4 - previewSize % 4));
			} else {
				ImGui::Image((ImTextureID)(uintptr_t)textureVector[i].getTextureData().textureID, ImVec2(width / 4, height / 4));
			}
			if (i % 4 != 3 && i != previewSize - 1) {
				ImGui::SameLine();
			}
			if (ImGui::IsItemClicked()) {
				previousSelectedImage = i;
			}
			if (ImGui::IsItemHovered()) {
				selectedImage = i;
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::TreePop();
	}
	ImGui::End();
	ImGui::SetNextWindowSizeConstraints(ImVec2(20 + width, -1.f), ImVec2(INFINITY, -1.f));
	ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_Once);
	ImGui::Begin("ofxStableDiffusion##foo1", NULL, flags);
	if (ImGui::TreeNodeEx("Image", ImGuiStyleVar_WindowPadding | ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Dummy(ImVec2(0, 10));
		if (selectedImage >= 0 && selectedImage < static_cast<int>(textureVector.size())) {
			ImGui::Image((ImTextureID)(uintptr_t)textureVector[selectedImage].getTextureData().textureID, ImVec2(width, height));
		}
		if (!isImageToVideo) {
			ofxStableDiffusionImageScore frameScore;
			bool frameSelected = false;
			if (stableDiffusion.getImageFrameMetadata(selectedImage, frameScore, frameSelected)) {
				if (frameScore.valid) {
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::Text("Score: %.3f", frameScore.score);
					if (!frameScore.scorer.empty()) {
						ImGui::SameLine(0, 10);
						ImGui::Text("Scorer: %s", frameScore.scorer.c_str());
					}
				}
				if (frameSelected) {
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::Text("Best-of-N Selection");
				}
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Indent(-10);
		if (ImGui::Button("Save")) {
			if (isImageToVideo) {
				ofPixels framePixels;
				if (stableDiffusion.copyVideoFramePixels(selectedImage, framePixels) && framePixels.isAllocated()) {
					ofSaveImage(
						framePixels,
						ofGetTimestampString("output/ofxStableDiffusion-%Y-%m-%d-%H-%M-%S.png"));
				}
			} else if (selectedImage >= 0 && selectedImage < static_cast<int>(textureVector.size())) {
				textureVector[selectedImage].readToPixels(pixels);
				ofSaveImage(pixels, ofGetTimestampString("output/ofxStableDiffusion-%Y-%m-%d-%H-%M-%S.png"));
			}
		}
		ImGui::TreePop();
	}
	ImGui::End();
	ImGui::SetNextWindowSizeConstraints(ImVec2(532.f, -1.f), ImVec2(532.f, -1.f));
	ImGui::SetNextWindowPos(ImVec2(602, 20), ImGuiCond_Once);
	ImGui::Begin("ofxStableDiffusion##foo2", &logOpenSettings, flags);
	if (!logOpenSettings) {
		ImGui::OpenPopup("Exit Program?");
	}
	if (ImGui::BeginPopupModal("Exit Program?", NULL, flags)) {
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Button("Yes", ImVec2(50, 17))) {
			ofExit();
		}
		ImGui::SameLine(0, 10);
		if (ImGui::Button("No", ImVec2(50, 17))) {
			ImGui::CloseCurrentPopup();
			logOpenSettings = true;
		}
		ImGui::EndPopup();
	}
	if (promptIsEdited) {
		addSoftReturnsToText(prompt, 500);
		promptIsEdited = false;
	}
	if (ImGui::TreeNodeEx("Prompt", ImGuiStyleVar_WindowPadding | ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Dummy(ImVec2(0, 10));
		Funcs::MyInputTextMultiline("##MyStr1", &prompt, ImVec2(512, 150), ImGuiInputTextFlags_CallbackResize);
		if (ImGui::IsItemDeactivatedAfterEdit()) {
			prompt.erase(std::remove(prompt.begin(), prompt.end(), '\r'), prompt.end());
			prompt.erase(std::remove(prompt.begin(), prompt.end(), '\n'), prompt.end());
			promptIsEdited = true;
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::TreePop();
	}
	if (isInstructImage) {
		if (ImGui::TreeNodeEx("Instruction", ImGuiStyleVar_WindowPadding | ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Dummy(ImVec2(0, 10));
			Funcs::MyInputTextMultiline("##MyStrInstruction", &instruction, ImVec2(512, 110), ImGuiInputTextFlags_CallbackResize);
			if (ImGui::IsItemDeactivatedAfterEdit()) {
				instruction.erase(std::remove(instruction.begin(), instruction.end(), '\r'), instruction.end());
				instruction.erase(std::remove(instruction.begin(), instruction.end(), '\n'), instruction.end());
			}
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::TreePop();
		}
	}
	if (negativePromptIsEdited) {
		addSoftReturnsToText(negativePrompt, 500);
		negativePromptIsEdited = false;
	}
	if (ImGui::TreeNodeEx("Negative Prompt", ImGuiStyleVar_WindowPadding)) {
		ImGui::Dummy(ImVec2(0, 10));
		Funcs::MyInputTextMultiline("##MyStr2", &negativePrompt, ImVec2(512, 150), ImGuiInputTextFlags_CallbackResize);
		if (ImGui::IsItemDeactivatedAfterEdit()) {
			negativePrompt.erase(std::remove(negativePrompt.begin(), negativePrompt.end(), '\r'), negativePrompt.end());
			negativePrompt.erase(std::remove(negativePrompt.begin(), negativePrompt.end(), '\n'), negativePrompt.end());
			negativePromptIsEdited = true;
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::TreePop();
	}
	if (ImGui::TreeNodeEx("Settings", ImGuiStyleVar_WindowPadding | ImGuiTreeNodeFlags_DefaultOpen)) {
		if (isBusy) {
			ImGui::BeginDisabled();
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Button("Full Screen")) {
			if (!isFullScreen) {
				isFullScreen = true;
				ofSetFullscreen(isFullScreen);
			}
			else {
				isFullScreen = false;
				ofSetFullscreen(isFullScreen);
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Button("Load Main Model")) {
			if (selectPath("Load Main Model", modelPath)) {
				modelName = ofFilePath::getFileName(modelPath);
				diffusionModelPath.clear();
			}
		}
		ImGui::SameLine(0, 5);
		if (ImGui::Button("Unload Main Model")) {
			modelPath.clear();
			modelName.clear();
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", modelPath.empty() ? "No main model selected" : modelPath.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load Diffusion Model")) {
			if (selectPath("Load Diffusion Model", diffusionModelPath)) {
				modelPath.clear();
				modelName.clear();
			}
		}
		ImGui::SameLine(0, 5);
		if (ImGui::Button("Unload Diffusion Model")) {
			diffusionModelPath.clear();
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", diffusionModelPath.empty() ? "No diffusion model selected" : diffusionModelPath.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load CLIP-L")) {
			selectPath("Load CLIP-L", clipLPath);
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", clipLPath.empty() ? "No CLIP-L selected" : clipLPath.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load CLIP-G")) {
			selectPath("Load CLIP-G", clipGPath);
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", clipGPath.empty() ? "No CLIP-G selected" : clipGPath.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load UMT5 / T5XXL")) {
			selectPath("Load UMT5 / T5XXL", t5xxlPath);
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", t5xxlPath.empty() ? "No UMT5 / T5XXL selected" : t5xxlPath.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load VAE")) {
			selectPath("Load VAE", vaePath);
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", vaePath.empty() ? "No VAE selected" : vaePath.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load TAESD")) {
			if (selectPath("Load TAESD", taesdPath)) {
				isTAESD = !taesdPath.empty();
			}
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", taesdPath.empty() ? "No TAESD selected" : taesdPath.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load ControlNet")) {
			selectPath("Load ControlNet", controlNetPath);
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", controlNetPath.empty() ? "No ControlNet selected" : controlNetPath.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load Embeddings Dir")) {
			selectPath("Load Embeddings Directory", embedDir, true);
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", embedDir.empty() ? "No embeddings directory selected" : embedDir.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load LoRA Dir")) {
			selectPath("Load LoRA Directory", loraModelDir, true);
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", loraModelDir.empty() ? "No LoRA directory selected" : loraModelDir.c_str());
		ImGui::Dummy(ImVec2(0, 8));
		if (ImGui::Button("Load PhotoMaker")) {
			selectPath("Load PhotoMaker Model", stackedIdEmbedDir);
		}
		ImGui::SameLine(0, 5);
		ImGui::Text("%s", stackedIdEmbedDir.empty() ? "No PhotoMaker model selected" : stackedIdEmbedDir.c_str());
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
		if (ImGui::Button("Load Context")) {
			refreshModelContext();
		}
		ImGui::PopItemFlag();
		ImGui::SameLine(0, 5);
		ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
		if (ImGui::Button("Unload Context")) {
			stableDiffusion.freeSdCtx();
		}
		ImGui::PopItemFlag();
		ImGui::Dummy(ImVec2(0, 10));
		if (stableDiffusion.getLastError().empty()) {
			ImGui::Text("Status: %s", isBusy ? "Running" : "Idle");
		} else {
			ImGui::TextWrapped("Status: %s", stableDiffusion.getLastError().c_str());
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::BeginCombo("Logging", nativeLogLevelLabels[nativeLogLevelIndex], ImGuiComboFlags_NoArrowButton)) {
			for (int n = 0; n < IM_ARRAYSIZE(nativeLogLevelLabels); ++n) {
				const bool isSelected = (nativeLogLevelIndex == n);
				if (ImGui::Selectable(nativeLogLevelLabels[n], isSelected)) {
					nativeLogLevelIndex = n;
					if (n == 4) {
						stableDiffusion.setNativeLoggingEnabled(false);
					} else {
						stableDiffusion.setNativeLoggingEnabled(true);
						stableDiffusion.setNativeLogLevel(static_cast<sd_log_level_t>(n));
					}
				}
				if (isSelected) {
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::BeginCombo("RNG", sd_rng_type_name(rngType), ImGuiComboFlags_NoArrowButton)) {
			for (rng_type_t option : rngTypeOptions) {
				const bool isSelected = (rngType == option);
				if (ImGui::Selectable(sd_rng_type_name(option), isSelected)) {
					rngType = option;
				}
				if (isSelected) {
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
		if (progressSteps > 0 && isBusy) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Text("Progress %d / %d (%.2fs)", progressStep, progressSteps, progressTime);
		}
		ImGui::Dummy(ImVec2(0, 10));
		drawHoloscanBridgeSection();
		ImGui::Dummy(ImVec2(0, 10));
		const bool supportsImageMode =
			capabilities.textToImage ||
			capabilities.imageToImage ||
			capabilities.instructImage ||
			capabilities.variation ||
			capabilities.restyle ||
			capabilities.inpainting;
		const bool supportsVideoMode = capabilities.imageToVideo;
		const bool showTopLevelModeSelector = supportsImageMode && supportsVideoMode;
		const char* currentTopLevelModeLabel =
			isImageToVideo ? "Video" : "Image";
		if (!showTopLevelModeSelector) {
			if (supportsVideoMode && !supportsImageMode) {
				currentTopLevelModeLabel = "Video";
			} else if (supportsImageMode && !supportsVideoMode) {
				currentTopLevelModeLabel = "Image";
			}
		}
		if (!showTopLevelModeSelector) {
			ImGui::BeginDisabled();
		}
		if (ImGui::BeginCombo("Mode", currentTopLevelModeLabel, ImGuiComboFlags_NoArrowButton)) {
			if (supportsImageMode) {
				const bool imageSelected = !isImageToVideo;
				if (ImGui::Selectable("Image", imageSelected)) {
					isImageToVideo = false;
					applySelectedImageMode(imageModeEnum);
				}
				if (imageSelected) {
					ImGui::SetItemDefaultFocus();
				}
			}
			if (supportsVideoMode) {
				const bool videoSelected = isImageToVideo;
				if (ImGui::Selectable("Video", videoSelected)) {
					isImageToVideo = true;
					videoUseInputImage = videoModeSupportsTextAndImage() ? false : capabilities.videoRequiresInputImage;
					applyRecommendedVideoParameters();
					clampCurrentParametersToProfiles();
				}
				if (videoSelected) {
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
		if (!showTopLevelModeSelector) {
			ImGui::EndDisabled();
		}
		if (!isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			if (ImGui::BeginCombo("Image Mode", imageMode, ImGuiComboFlags_NoArrowButton)) {
				for (int n = 0; n < IM_ARRAYSIZE(imageModeArray); n++) {
					const auto mode = static_cast<ofxStableDiffusionImageMode>(n);
					const bool supported =
						(mode == ofxStableDiffusionImageMode::TextToImage && capabilities.textToImage) ||
						(mode == ofxStableDiffusionImageMode::ImageToImage && capabilities.imageToImage) ||
						(mode == ofxStableDiffusionImageMode::InstructImage && capabilities.instructImage) ||
						(mode == ofxStableDiffusionImageMode::Variation && capabilities.variation) ||
						(mode == ofxStableDiffusionImageMode::Restyle && capabilities.restyle) ||
						(mode == ofxStableDiffusionImageMode::Inpainting && capabilities.inpainting);
					if (!supported) {
						continue;
					}
					const bool is_selected = (imageModeEnum == mode);
					if (ImGui::Selectable(imageModeArray[n], is_selected)) {
						applySelectedImageMode(mode);
					}
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
		} else if (supportsVideoMode) {
			const bool allowVideoModeChoice = videoModeSupportsTextAndImage();
			const bool fixedImageToVideo = capabilities.videoRequiresInputImage && !allowVideoModeChoice;
			const char* currentVideoModeLabel =
				allowVideoModeChoice ?
					(videoUseInputImage ? "ImageToVideo" : "TextToVideo") :
					(fixedImageToVideo ? "ImageToVideo" : "TextToVideo");
			ImGui::Dummy(ImVec2(0, 10));
			if (!allowVideoModeChoice) {
				ImGui::BeginDisabled();
			}
			if (ImGui::BeginCombo("Video Mode", currentVideoModeLabel, ImGuiComboFlags_NoArrowButton)) {
				if (!fixedImageToVideo) {
					const bool t2vSelected = !videoUseInputImage;
					if (ImGui::Selectable("TextToVideo", t2vSelected)) {
						videoUseInputImage = false;
						applyRecommendedVideoParameters();
						clampCurrentParametersToProfiles();
					}
					if (t2vSelected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				if (allowVideoModeChoice || fixedImageToVideo) {
					const bool i2vSelected = videoUseInputImage;
					if (ImGui::Selectable("ImageToVideo", i2vSelected)) {
						videoUseInputImage = true;
						applyRecommendedVideoParameters();
						clampCurrentParametersToProfiles();
					}
					if (i2vSelected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
			if (!allowVideoModeChoice) {
				ImGui::EndDisabled();
			}
		}
		if (!isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			if (ImGui::BeginCombo("Selection", selectionMode, ImGuiComboFlags_NoArrowButton)) {
				for (int n = 0; n < IM_ARRAYSIZE(selectionModeArray); n++) {
					const bool is_selected = (selectionMode == selectionModeArray[n]);
					if (ImGui::Selectable(selectionModeArray[n], is_selected)) {
						selectionMode = selectionModeArray[n];
						selectionModeEnum = static_cast<ofxStableDiffusionImageSelectionMode>(n);
						stableDiffusion.setImageSelectionMode(selectionModeEnum);
						configureExampleRanker();
					}
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
			if (selectionModeEnum != ofxStableDiffusionImageSelectionMode::KeepOrder) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::Checkbox("Use Demo Ranking Callback", &useDemoRanking);
				if (ImGui::IsItemEdited()) {
					configureExampleRanker();
				}
				ImGui::Dummy(ImVec2(0, 10));
				Funcs::MyInputTextMultiline(
					"Ranking Prompt",
					&rankingPrompt,
					ImVec2(512, 80),
					ImGuiInputTextFlags_CallbackResize);
				ImGui::Dummy(ImVec2(0, 10));
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Text(
			"Model Family: %s",
			ofxStableDiffusionModelFamilyLabel(capabilities.modelFamily));
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::TextWrapped(
			"%s",
			isImageToVideo ? videoParameterProfile.summary : imageParameterProfile.summary);
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Button("Default")) {
			if (isImageToVideo) {
				applyRecommendedVideoParameters();
			} else {
				applyRecommendedImageParameters();
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Checkbox("TAESD", &isTAESD)) {
			if (!isBusy && isTAESD) {
				taesdPath = "data/models/taesd/taesd.safetensors";

			}
			else if (!isBusy && !isTAESD) {
				taesdPath = "";

			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Checkbox("Offload Params To CPU", &offloadParamsToCpu);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Checkbox("CLIP On CPU", &keepClipOnCpu);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Checkbox("VAE On CPU", &keepVaeOnCpu);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Checkbox("Flash Attention", &flashAttn);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Checkbox("Diffusion Flash Attention", &diffusionFlashAttn);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Checkbox("VAE Tiling", &vaeTiling);
		if (usesInputImageMode() || currentVideoModeUsesInputImage()) {
			ImGui::Dummy(ImVec2(0, 10));
			if (ImGui::Button("Load Image")) {
				if (loadImageIntoSlot("Load Image", image, pixels, inputImage, imageName)) {
					allocate();
				}
			}
			ImGui::SameLine(0, 5);
			ImGui::Text("%s", imageName.empty() ? "No image loaded" : imageName.c_str());
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Image((ImTextureID)(uintptr_t)fbo.getTexture().getTextureData().textureID, ImVec2(128, 128));
		}
		if (imageModeEnum == ofxStableDiffusionImageMode::Inpainting) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Checkbox("Use Mask", &useMaskGuide);
			if (useMaskGuide) {
				ImGui::Dummy(ImVec2(0, 10));
				if (ImGui::Button("Load Mask")) {
					if (loadImageIntoSlot("Load Mask Image", maskGuideImage, maskGuidePixels, maskGuideInput, maskImageName)) {
						useMaskGuide = true;
					}
				}
				ImGui::SameLine(0, 5);
				ImGui::Text("%s", maskImageName.empty() ? "No mask loaded" : maskImageName.c_str());
				if (maskGuideImage.isAllocated()) {
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::Image(
						(ImTextureID)(uintptr_t)maskGuideImage.getTexture().getTextureData().textureID,
						ImVec2(128, 128));
				}
			}
		}
		if (!controlNetPath.empty() && !isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Checkbox("Use Control Image", &useControlGuide);
			if (useControlGuide) {
				ImGui::Dummy(ImVec2(0, 10));
				if (ImGui::Button("Load Control Image")) {
					loadImageIntoSlot(
						"Load Control Image",
						controlGuideImage,
						controlGuidePixels,
						controlGuideInput,
						controlGuideName);
				}
				ImGui::SameLine(0, 5);
				ImGui::Text("%s", controlGuideName.empty() ? "No control image loaded" : controlGuideName.c_str());
				if (controlGuideImage.isAllocated()) {
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::Image(
						(ImTextureID)(uintptr_t)controlGuideImage.getTexture().getTextureData().textureID,
						ImVec2(128, 128));
				}
			}
		}
		if (isImageToVideo) {
			const auto& videoCapabilities = capabilities;
			const bool highNoiseOverridesAvailable =
				videoCapabilities.modelFamily == ofxStableDiffusionModelFamily::WAN ||
				videoCapabilities.modelFamily == ofxStableDiffusionModelFamily::WANI2V ||
				videoCapabilities.modelFamily == ofxStableDiffusionModelFamily::WANTI2V ||
				videoCapabilities.modelFamily == ofxStableDiffusionModelFamily::WANFLF2V ||
				videoCapabilities.modelFamily == ofxStableDiffusionModelFamily::WANVACE;
			if (!highNoiseOverridesAvailable) {
				useHighNoiseOverrides = false;
			}
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::InputInt("Video Frames", &videoFrames, 1, 10);
			if (videoParameterProfile.supportsVaceStrength) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::SliderFloat(
					"VACE Strength",
					&vaceStrength,
					videoParameterProfile.minVaceStrength,
					videoParameterProfile.maxVaceStrength);
			}
			if (!videoCapabilities.videoAnimation) {
				enablePromptInterpolation = false;
				useSeedSequence = false;
			} else {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::Checkbox("Image-Sequence Prompt Morph", &enablePromptInterpolation);
				if (enablePromptInterpolation) {
					ImGui::Dummy(ImVec2(0, 10));
					Funcs::MyInputTextMultiline(
						"Sequence Prompt B",
						&promptB,
						ImVec2(512, 80),
						ImGuiInputTextFlags_CallbackResize);
					ImGui::Dummy(ImVec2(0, 10));
					if (ImGui::BeginCombo("Interpolation", interpolationMode, ImGuiComboFlags_NoArrowButton)) {
						for (int n = 0; n < IM_ARRAYSIZE(interpolationModeArray); n++) {
							const bool is_selected = (interpolationMode == interpolationModeArray[n]);
							if (ImGui::Selectable(interpolationModeArray[n], is_selected)) {
								interpolationMode = interpolationModeArray[n];
								interpolationModeEnum = static_cast<ofxStableDiffusionInterpolationMode>(n);
							}
							if (is_selected) {
								ImGui::SetItemDefaultFocus();
							}
						}
						ImGui::EndCombo();
					}
				}
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::Checkbox("Image-Sequence Seed Sweep", &useSeedSequence);
				if (useSeedSequence) {
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::InputInt("Seed Increment", &seedIncrement, 1, 10);
				}
			}
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat("MoE Boundary", &videoMoeBoundary, 0.0f, 1.0f);
			ImGui::Dummy(ImVec2(0, 10));
			if (ImGui::BeginCombo("Video Cache", videoCacheMode, ImGuiComboFlags_NoArrowButton)) {
				for (int n = 0; n < IM_ARRAYSIZE(videoCacheModeArray); n++) {
					const bool is_selected = (videoCacheMode == videoCacheModeArray[n]);
					if (ImGui::Selectable(videoCacheModeArray[n], is_selected)) {
						videoCacheMode = videoCacheModeArray[n];
					}
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
			const sd_cache_mode_t selectedCacheMode = cacheModeFromLabel(videoCacheMode);
			if (selectedCacheMode != SD_CACHE_DISABLED) {
				if (cacheModeUsesThresholdWindow(selectedCacheMode)) {
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::SliderFloat("Cache Threshold", &videoCacheThreshold, 0.0f, 3.0f);
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::SliderFloat("Cache Start", &videoCacheStartPercent, 0.0f, 0.95f);
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::SliderFloat("Cache End", &videoCacheEndPercent, 0.05f, 1.0f);
				}
			}
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat("Flow Shift", &flowShift, 0.0f, 10.0f);
			if (highNoiseOverridesAvailable) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::Checkbox("Use High-Noise Overrides", &useHighNoiseOverrides);
				if (useHighNoiseOverrides) {
					ImGui::Dummy(ImVec2(0, 10));
					const std::string resolvedHighNoiseSampleLabel =
						buildResolvedSampleMethodMenuLabel(highNoiseSampleMethodEnum);
					const std::string highNoiseDefaultOptionLabel =
						buildModelDefaultSampleMethodMenuLabel();
					if (ImGui::BeginCombo(
							"High-Noise Sample",
							resolvedHighNoiseSampleLabel.c_str(),
							ImGuiComboFlags_NoArrowButton)) {
						const bool is_default_selected = (highNoiseSampleMethodEnum == SAMPLE_METHOD_COUNT);
						if (ImGui::Selectable(highNoiseDefaultOptionLabel.c_str(), is_default_selected)) {
							highNoiseSampleMethod = "MODEL_DEFAULT";
							highNoiseSampleMethodEnum = SAMPLE_METHOD_COUNT;
						}
						if (is_default_selected) {
							ImGui::SetItemDefaultFocus();
						}
						for (int n = 0; n < IM_ARRAYSIZE(sampleMethodArray); n++) {
							const bool is_selected = (highNoiseSampleMethod == sampleMethodArray[n]);
							if (ImGui::Selectable(sampleMethodArray[n], is_selected)) {
								highNoiseSampleMethod = sampleMethodArray[n];
								highNoiseSampleMethodEnum = static_cast<sample_method_t>(n);
							}
							if (is_selected) {
								ImGui::SetItemDefaultFocus();
							}
						}
						ImGui::EndCombo();
					}
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::SliderInt(
						"High-Noise Steps",
						&highNoiseSampleSteps,
						videoParameterProfile.minSampleSteps,
						videoParameterProfile.maxSampleSteps);
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::SliderFloat(
						"High-Noise CFG",
						&highNoiseCfgScale,
						videoParameterProfile.minCfgScale,
						videoParameterProfile.maxCfgScale);
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::SliderFloat("High-Noise Guidance", &highNoiseGuidance, 0.0f, 10.0f);
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::SliderFloat("High-Noise Eta", &highNoiseEta, 0.0f, 1.0f);
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::SliderFloat("High-Noise Flow Shift", &highNoiseFlowShift, 0.0f, 12.0f);
				}
			}
			if (currentVideoModeUsesInputImage() && videoCapabilities.videoEndFrame) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::Checkbox("Use End Frame", &useEndFrame);
				if (useEndFrame) {
					ImGui::Dummy(ImVec2(0, 10));
					if (ImGui::Button("Load End Frame")) {
						ofFileDialogResult result = ofSystemLoadDialog("Load End Frame", false, "");
						if (result.bSuccess) {
							endImageName = result.getName();
							if (endFrameImage.load(result.getPath())) {
								endFrameImage.resize(width, height);
								endFramePixels = endFrameImage.getPixels();
								endInputImage = {
									static_cast<uint32_t>(endFramePixels.getWidth()),
									static_cast<uint32_t>(endFramePixels.getHeight()),
									static_cast<uint32_t>(endFramePixels.getNumChannels()),
									endFramePixels.getData()
								};
							}
						}
					}
					ImGui::SameLine(0, 5);
					ImGui::Text("%s", endImageName.empty() ? "No end frame loaded" : endImageName.c_str());
					if (endFrameImage.isAllocated()) {
						ImGui::Dummy(ImVec2(0, 10));
						ImGui::Image(
							(ImTextureID)(uintptr_t)endFrameImage.getTexture().getTextureData().textureID,
							ImVec2(128, 128));
					}
				}
			}
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Checkbox("Use Video Control Frames", &useVideoControlFrames);
			if (useVideoControlFrames) {
				ImGui::Dummy(ImVec2(0, 10));
				if (ImGui::Button("Use Default Folder")) {
					if (!loadVideoControlFramesFromFolder(defaultVideoControlFramesPath)) {
						clearVideoControlFrames();
					}
				}
				ImGui::SameLine(0, 10);
				if (ImGui::Button("Fill Default Folder From Current Video")) {
					const auto clip = stableDiffusion.getVideoClip();
					if (!clip.empty()) {
						stableDiffusion.saveVideoFramesWithMetadata(
							defaultVideoControlFramesPath,
							"control",
							"metadata.json");
						loadVideoControlFramesFromFolder(defaultVideoControlFramesPath);
						useVideoControlFrames = true;
					}
				}
				ImGui::SameLine(0, 10);
				if (ImGui::Button("Load Control Frame Folder")) {
					std::string selectedFolder;
					if (selectPath("Load Control Frame Folder", selectedFolder, true)) {
						if (!loadVideoControlFramesFromFolder(selectedFolder)) {
							videoControlFramesPath = selectedFolder;
						}
					}
				}
				ImGui::SameLine(0, 10);
				if (ImGui::Button("Clear Control Frames")) {
					clearVideoControlFrames();
				}
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::TextWrapped(
					"%s",
					videoControlFramesPath.empty()
						? "No control-frame folder loaded"
						: videoControlFramesPath.c_str());
				if (!defaultVideoControlFramesPath.empty()) {
					ImGui::Dummy(ImVec2(0, 6));
					ImGui::TextWrapped("Default control-frame folder: %s", defaultVideoControlFramesPath.c_str());
				}
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::Text("Loaded Control Frames: %d", static_cast<int>(videoControlFrames.size()));
				if (!videoControlFrames.empty() && static_cast<int>(videoControlFrames.size()) != videoFrames) {
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::TextWrapped(
						"The native backend will use up to the smaller of frame count and loaded control frames. Current request: %d frames, %d controls.",
						videoFrames,
						static_cast<int>(videoControlFrames.size()));
				}
				if (!videoControlFrameImages.empty()) {
					videoControlPreviewIndex = std::clamp(
						videoControlPreviewIndex,
						0,
						static_cast<int>(videoControlFrameImages.size()) - 1);
					ImGui::Dummy(ImVec2(0, 10));
					if (videoControlFrameImages.size() > 1) {
						ImGui::SliderInt(
							"Control Preview",
							&videoControlPreviewIndex,
							0,
							static_cast<int>(videoControlFrameImages.size()) - 1);
					}
					ImGui::Dummy(ImVec2(0, 10));
					ImGui::Image(
						(ImTextureID)(uintptr_t)videoControlFrameImages[static_cast<std::size_t>(videoControlPreviewIndex)].getTexture().getTextureData().textureID,
						ImVec2(128, 128));
				}
			}
		}
		if (isImageToVideo ? videoParameterProfile.supportsClipSkip : imageParameterProfile.supportsClipSkip) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderInt(
				"Clip Skip Layers",
				&clipSkip,
				isImageToVideo ? videoParameterProfile.minClipSkip : imageParameterProfile.minClipSkip,
					isImageToVideo ? videoParameterProfile.maxClipSkip : imageParameterProfile.maxClipSkip);
		}
		if (!isImageToVideo && useControlGuide && !controlNetPath.empty()) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat("Control Strength", &controlStrength, 0, 2);
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderFloat(
			"CFG Scale",
			&cfgScale,
			isImageToVideo ? videoParameterProfile.minCfgScale : imageParameterProfile.minCfgScale,
			isImageToVideo ? videoParameterProfile.maxCfgScale : imageParameterProfile.maxCfgScale);
		if (isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat("Guidance", &guidance, 0.0f, 10.0f);
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderInt(
			"Sample Steps",
			&sampleSteps,
			isImageToVideo ? videoParameterProfile.minSampleSteps : imageParameterProfile.minSampleSteps,
			isImageToVideo ? videoParameterProfile.maxSampleSteps : imageParameterProfile.maxSampleSteps);
		ImGui::Dummy(ImVec2(0, 10));
		int nextWidth = width;
		if (ImGui::InputInt("Width", &nextWidth, 64, 128)) {
			if (nextWidth > 0 && nextWidth != width) {
				width = nextWidth;
				allocate();
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		int nextHeight = height;
		if (ImGui::InputInt("Height", &nextHeight, 64, 128)) {
			if (nextHeight > 0 && nextHeight != height) {
				height = nextHeight;
				allocate();
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		const std::string resolvedSampleMethodLabel =
			buildResolvedSampleMethodMenuLabel(sampleMethodEnum);
		const std::string sampleMethodDefaultOptionLabel =
			buildModelDefaultSampleMethodMenuLabel();
		if (ImGui::BeginCombo(
				"Sample Method",
				resolvedSampleMethodLabel.c_str(),
				ImGuiComboFlags_NoArrowButton)) {
			const bool is_default_selected = (sampleMethodEnum == SAMPLE_METHOD_COUNT);
			if (ImGui::Selectable(sampleMethodDefaultOptionLabel.c_str(), is_default_selected)) {
				sampleMethod = "MODEL_DEFAULT";
				sampleMethodEnum = SAMPLE_METHOD_COUNT;
				if (!useHighNoiseOverrides) {
					highNoiseSampleMethod = sampleMethod;
					highNoiseSampleMethodEnum = sampleMethodEnum;
				}
			}
			if (is_default_selected)
				ImGui::SetItemDefaultFocus();
			for (int n = 0; n < IM_ARRAYSIZE(sampleMethodArray); n++) {
				bool is_selected = (sampleMethod == sampleMethodArray[n]);
				if (ImGui::Selectable(sampleMethodArray[n], is_selected)) {
					sampleMethod = sampleMethodArray[n];
					sampleMethodEnum = (sample_method_t)n;
					if (!useHighNoiseOverrides) {
						highNoiseSampleMethod = sampleMethod;
						highNoiseSampleMethodEnum = sampleMethodEnum;
					}
				}
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		if (isImageToVideo) {
			static const scheduler_t schedulerOptions[] = {
				DISCRETE_SCHEDULER,
				KARRAS_SCHEDULER,
				EXPONENTIAL_SCHEDULER,
				AYS_SCHEDULER,
				GITS_SCHEDULER,
				SGM_UNIFORM_SCHEDULER,
				SIMPLE_SCHEDULER,
				SMOOTHSTEP_SCHEDULER,
				KL_OPTIMAL_SCHEDULER,
				LCM_SCHEDULER,
				BONG_TANGENT_SCHEDULER
			};
			ImGui::Dummy(ImVec2(0, 10));
			const std::string resolvedSchedulerLabel =
				buildResolvedSchedulerMenuLabel(sampleMethodEnum, schedule);
			const std::string schedulerDefaultOptionLabel =
				buildModelDefaultSchedulerMenuLabel(sampleMethodEnum);
			if (ImGui::BeginCombo(
					"Scheduler",
					resolvedSchedulerLabel.c_str(),
					ImGuiComboFlags_NoArrowButton)) {
				const bool is_default_scheduler = (schedule == SCHEDULER_COUNT);
				if (ImGui::Selectable(schedulerDefaultOptionLabel.c_str(), is_default_scheduler)) {
					schedule = SCHEDULER_COUNT;
				}
				if (is_default_scheduler) {
					ImGui::SetItemDefaultFocus();
				}
				for (scheduler_t option : schedulerOptions) {
					const bool is_selected = (schedule == option);
					if (ImGui::Selectable(schedulerToCliName(option), is_selected)) {
						schedule = option;
					}
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
		}
		if (isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat("Eta", &eta, 0.0f, 1.0f);
		}
		if ((isImageToVideo && videoParameterProfile.maxStrength > videoParameterProfile.minStrength) ||
			(!isImageToVideo && imageParameterProfile.supportsStrength)) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat(
				"Strength",
				&strength,
				isImageToVideo ? videoParameterProfile.minStrength : imageParameterProfile.minStrength,
				isImageToVideo ? videoParameterProfile.maxStrength : imageParameterProfile.maxStrength);
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::InputInt("Seed", &seed, 1, 100);
		if (!isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderInt("Batch Size", &batchCount, 1, 16);
			ImGui::Dummy(ImVec2(0, 10));
			if (ImGui::Checkbox("4x Upscale", &isESRGAN)) {
				allocate();
			}
		}
		if (isBusy) {
			ImGui::EndDisabled();
		}
		ImGui::TreePop();
	}
	if (isBusy) {
		ImGui::BeginDisabled();
	}
	ImGui::Dummy(ImVec2(0, 10));
	const bool modelReady = capabilities.contextConfigured;
	const bool needsInputImage =
		usesInputImageMode() ||
		currentVideoModeRequiresInputImage();
	const bool hasInputImage = inputImage.data != nullptr;
	const bool needsMaskImage = imageModeEnum == ofxStableDiffusionImageMode::Inpainting;
	const bool maskReady = needsMaskImage ? (useMaskGuide && maskGuideInput.data != nullptr) : true;
	if (!modelReady) {
		ImGui::TextWrapped("Load a model and click Load Context before generating.");
	}
	if (needsInputImage && !hasInputImage) {
		ImGui::TextWrapped("Load an input image for the selected mode.");
	}
	if (needsMaskImage && !maskReady) {
		ImGui::TextWrapped("Inpainting needs a mask image. Enable Use Mask and load one before generating.");
	}
	ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Button("Generate")) {
			progressStep = 0;
			progressSteps = 0;
			progressTime = 0.0f;
			isPlaying = false;
			currentFrame = 0;
			totalVideoFrames = 0;
			selectedImage = 0;
			previousSelectedImage = 0;
			configureExampleRanker();
			clampCurrentParametersToProfiles();
			if (isImageToVideo) {
				ofxStableDiffusionVideoRequest request = buildVideoRequest();
				stableDiffusion.generateVideo(request);
			}
		else {
			ofxStableDiffusionImageRequest request;
			request.mode = imageModeEnum;
			request.selectionMode = selectionModeEnum;
			request.initImage = inputImage;
			request.prompt = prompt;
			request.instruction = isInstructImage ? instruction : "";
			request.negativePrompt = negativePrompt;
			request.clipSkip = clipSkip;
			request.cfgScale = cfgScale;
			request.width = width;
			request.height = height;
			request.sampleMethod = sampleMethodEnum;
			request.sampleSteps = sampleSteps;
			request.strength =
				imageParameterProfile.supportsStrength ?
					strength :
					std::numeric_limits<float>::infinity();
			request.seed = seed;
			request.batchCount = batchCount;
			request.maskImage =
				(imageModeEnum == ofxStableDiffusionImageMode::Inpainting && useMaskGuide && maskGuideInput.data != nullptr) ?
					maskGuideInput :
					sd_image_t{0, 0, 0, nullptr};
			request.controlCond =
				(useControlGuide && controlGuideInput.data != nullptr) ?
					&controlGuideInput :
					nullptr;
			request.controlStrength = controlStrength;
			request.styleStrength = styleStrength;
			request.normalizeInput = normalizeInput;
			request.inputIdImagesPath = inputIdImagesPath;
			stableDiffusion.generate(request);
		}
	}
	if (isImageToVideo) {
		ImGui::SameLine(0, 10);
		if (ImGui::Button("Print Equivalent sd-cli")) {
			equivalentSdCliCommand = buildEquivalentSdCliCommand();
			ofLogNotice("ofApp") << "Equivalent sd-cli: " << equivalentSdCliCommand;
		}
		if (!equivalentSdCliCommand.empty()) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::TextWrapped("%s", equivalentSdCliCommand.c_str());
		}
	}
	if (totalVideoFrames > 0) {
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Button(isPlaying ? "Stop" : "Play")) {
			isPlaying = !isPlaying;
			if (isPlaying) {
				lastFrameTime = ofGetElapsedTimef();
			}
		}
		ImGui::SameLine(0, 10);
		ImGui::Text("Frame %d / %d", currentFrame + 1, totalVideoFrames);
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::SliderInt("Frame", &currentFrame, 0, totalVideoFrames - 1)) {
			previousSelectedImage = currentFrame;
			isPlaying = false;
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Button("Save Frames")) {
			stableDiffusion.saveVideoFramesWithMetadata(
				ofGetTimestampString("output/ofxStableDiffusion-video-%Y-%m-%d-%H-%M-%S"),
				"frame",
				"metadata.json");
		}
		ImGui::SameLine(0, 10);
		if (ImGui::Button("Save Video")) {
			const std::string path = ofToDataPath(
				ofGetTimestampString("output/ofxStableDiffusion-video-%Y-%m-%d-%H-%M-%S.avi"),
				true);
			if (stableDiffusion.saveVideoWebm(path)) {
				ofLogNotice("ofApp") << "Saved result video to '" << path << "'";
			} else {
				ofLogError("ofApp") << "Failed to save result video to '" << path << "'";
			}
		}
	}
	if (isBusy) {
		ImGui::EndDisabled();
	}
	ImGui::End();
	ImGui::PopStyleVar(5);
	gui.end();
}

//--------------------------------------------------------------
void ofApp::addSoftReturnsToText(std::string& str, float multilineWidth) {
	float textSize = 0;
	std::string tmpStr = "";
	std::string finalStr = "";
	int curChr = 0;
	while (curChr < str.size()) {
		if (str[curChr] == '\n') {
			finalStr += tmpStr + "\n";
			tmpStr = "";
		}
		tmpStr += str[curChr];
		textSize = ImGui::CalcTextSize(tmpStr.c_str()).x;
		if (textSize > multilineWidth) {
			int lastSpace = static_cast<int>(tmpStr.size()) - 1;
			while (tmpStr[lastSpace] != ' ' && lastSpace > 0) {
				lastSpace--;
			}
			if (lastSpace == 0) {
				lastSpace = static_cast<int>(tmpStr.size()) - 2;
			}
			finalStr += tmpStr.substr(0, lastSpace + 1) + "\r\n";
			if (lastSpace + 1 > tmpStr.size()) {
				tmpStr = "";
			} else {
				tmpStr = tmpStr.substr(lastSpace + 1);
			}
		}
		curChr++;
	}
	if (tmpStr.size() > 0) {
		finalStr += tmpStr;
	}
	str = finalStr;
}

//--------------------------------------------------------------
void ofApp::applySelectedImageMode(ofxStableDiffusionImageMode mode) {
	imageModeEnum = mode;
	imageMode = imageModeArray[static_cast<int>(mode)];
	isTextToImage = (mode == ofxStableDiffusionImageMode::TextToImage);
	isInstructImage = (mode == ofxStableDiffusionImageMode::InstructImage);
	stableDiffusion.setImageGenerationMode(mode);
	applyRecommendedImageParameters();
	clampCurrentParametersToProfiles();
}

//--------------------------------------------------------------
void ofApp::applyRecommendedImageParameters() {
	imageParameterProfile = ofxStableDiffusionParameterTuningHelpers::resolveImageProfile(
		stableDiffusion.getContextSettings(),
		imageModeEnum);
	freeParamsImmediately = true;
	width = 512;
	height = 512;
	cfgScale = std::clamp(
		imageParameterProfile.defaultCfgScale,
		imageParameterProfile.minCfgScale,
		imageParameterProfile.maxCfgScale);
	sampleSteps = std::clamp(
		imageParameterProfile.defaultSampleSteps,
		imageParameterProfile.minSampleSteps,
		imageParameterProfile.maxSampleSteps);
	clipSkip = imageParameterProfile.supportsClipSkip ? -1 : imageParameterProfile.defaultClipSkip;
	sampleMethod = "MODEL_DEFAULT";
	sampleMethodEnum = SAMPLE_METHOD_COUNT;
	if (imageParameterProfile.supportsStrength) {
		strength = std::clamp(
			imageParameterProfile.defaultStrength,
			imageParameterProfile.minStrength,
			imageParameterProfile.maxStrength);
	}
	allocate();
}

//--------------------------------------------------------------
void ofApp::applyRecommendedVideoParameters() {
	videoParameterProfile = ofxStableDiffusionParameterTuningHelpers::resolveVideoProfile(
		stableDiffusion.getContextSettings());
	freeParamsImmediately = true;
	width = 512;
	height = 512;
	cfgScale = std::clamp(7.0f, videoParameterProfile.minCfgScale, videoParameterProfile.maxCfgScale);
	guidance = 3.5f;
	sampleSteps = std::clamp(20, videoParameterProfile.minSampleSteps, videoParameterProfile.maxSampleSteps);
	strength = std::clamp(0.75f, videoParameterProfile.minStrength, videoParameterProfile.maxStrength);
	clipSkip = videoParameterProfile.supportsClipSkip ? -1 : videoParameterProfile.defaultClipSkip;
	videoFrames = 6;
	videoFps = videoParameterProfile.defaultFps;
	vaceStrength = videoParameterProfile.supportsVaceStrength ?
		std::clamp(1.0f, videoParameterProfile.minVaceStrength, videoParameterProfile.maxVaceStrength) :
		1.0f;
	videoMoeBoundary = 0.875f;
	schedule = SCHEDULER_COUNT;
	sampleMethod = "MODEL_DEFAULT";
	sampleMethodEnum = SAMPLE_METHOD_COUNT;
	eta = 0.0f;
	switch (videoParameterProfile.modelFamily) {
	case ofxStableDiffusionModelFamily::WAN:
	case ofxStableDiffusionModelFamily::WANI2V:
	case ofxStableDiffusionModelFamily::WANTI2V:
	case ofxStableDiffusionModelFamily::WANFLF2V:
	case ofxStableDiffusionModelFamily::WANVACE:
		flowShift = 5.0f;
		break;
	default:
		flowShift = 3.0f;
		break;
	}
	highNoiseSampleMethod = sampleMethod;
	highNoiseSampleMethodEnum = sampleMethodEnum;
	useHighNoiseOverrides = false;
	highNoiseSampleSteps = -1;
	highNoiseCfgScale = cfgScale;
	highNoiseGuidance = guidance;
	highNoiseEta = eta;
	highNoiseFlowShift = flowShift;
	allocate();
}

//--------------------------------------------------------------
void ofApp::clampCurrentParametersToProfiles() {
	imageParameterProfile = ofxStableDiffusionParameterTuningHelpers::resolveImageProfile(
		stableDiffusion.getContextSettings(),
		imageModeEnum);
	videoParameterProfile = ofxStableDiffusionParameterTuningHelpers::resolveVideoProfile(
		stableDiffusion.getContextSettings());
}

//--------------------------------------------------------------
void ofApp::configureExampleRanker() {
	if (!useDemoRanking ||
		selectionModeEnum == ofxStableDiffusionImageSelectionMode::KeepOrder) {
		stableDiffusion.setImageRankCallback({});
		return;
	}

	stableDiffusion.setImageRankCallback(
		[this](
			const ofxStableDiffusionImageRequest& request,
			const std::vector<ofxStableDiffusionImageFrame>& images) {
			std::vector<ofxStableDiffusionImageScore> scores;
			scores.reserve(images.size());
			const std::string effectiveRankingPrompt =
				rankingPrompt.empty()
					? (request.instruction.empty() ? request.prompt : request.instruction)
					: rankingPrompt;
			for (const auto& frame : images) {
				ofxStableDiffusionImageScore score;
				if (frame.isAllocated()) {
					score.valid = true;
					score.score = computeHeuristicPromptScore(
						effectiveRankingPrompt,
						frame.pixels);
					score.scorer = "ExampleHeuristic";
					score.summary =
						"Standalone demo scorer using prompt keywords plus brightness/colorfulness heuristics";
				} else {
					score.scorer = "ExampleHeuristic";
					score.summary = "Frame is not allocated";
				}
				scores.push_back(std::move(score));
			}
			return scores;
		});
}

//--------------------------------------------------------------
bool ofApp::usesInputImageMode() const {
	return ofxStableDiffusionImageModeUsesInputImage(imageModeEnum);
}

//--------------------------------------------------------------
void ofApp::syncAutomaticMediaMode(const ofxStableDiffusionCapabilities& capabilities) {
	if (!capabilities.contextConfigured) {
		return;
	}

	const bool supportsImageMode =
		capabilities.textToImage ||
		capabilities.imageToImage ||
		capabilities.instructImage ||
		capabilities.variation ||
		capabilities.restyle ||
		capabilities.inpainting;
	const bool supportsVideoMode = capabilities.imageToVideo;

	if (supportsVideoMode && !supportsImageMode) {
		if (!isImageToVideo) {
			isImageToVideo = true;
			videoUseInputImage =
				(capabilities.modelFamily == ofxStableDiffusionModelFamily::WANTI2V) ?
					false :
					capabilities.videoRequiresInputImage;
			applyRecommendedVideoParameters();
			clampCurrentParametersToProfiles();
		}
		return;
	}

	if (supportsImageMode && !supportsVideoMode && isImageToVideo) {
		isImageToVideo = false;
		applySelectedImageMode(imageModeEnum);
	}
}

//--------------------------------------------------------------
bool ofApp::videoModeSupportsTextAndImage() const {
	const ofxStableDiffusionCapabilities capabilities = stableDiffusion.getCapabilities();
	return capabilities.modelFamily == ofxStableDiffusionModelFamily::WANTI2V;
}

//--------------------------------------------------------------
bool ofApp::currentVideoModeUsesInputImage() const {
	if (!isImageToVideo) {
		return false;
	}
	const ofxStableDiffusionCapabilities capabilities = stableDiffusion.getCapabilities();
	if (videoModeSupportsTextAndImage()) {
		return videoUseInputImage;
	}
	return capabilities.videoRequiresInputImage;
}

//--------------------------------------------------------------
bool ofApp::currentVideoModeRequiresInputImage() const {
	if (!isImageToVideo) {
		return false;
	}
	const ofxStableDiffusionCapabilities capabilities = stableDiffusion.getCapabilities();
	if (videoModeSupportsTextAndImage()) {
		return videoUseInputImage;
	}
	return capabilities.videoRequiresInputImage;
}

//--------------------------------------------------------------
void ofApp::allocate() {
	const std::size_t previewTextureCount =
		std::min<std::size_t>(textureVector.size(), static_cast<std::size_t>(maxPreviewTextures));
	for (std::size_t i = 0; i < previewTextureCount; ++i) {
		auto& texture = textureVector[i];
		if (texture.isAllocated()) {
			texture.clear();
		}
		if (isESRGAN) {
			texture.allocate(width * esrganMultiplier, height * esrganMultiplier, GL_RGB);
		} else {
			texture.allocate(width, height, GL_RGB);
		}
	}
	fbo.allocate(width, height, GL_RGB);
	if (image.isAllocated()) {
		fbo.begin();
		image.draw(0, 0, width, height);
		fbo.end();
		fbo.getTexture().readToPixels(pixels);
		inputImage = { (uint32_t)width, (uint32_t)height, 3, pixels.getData() };
	}
	if (endFrameImage.isAllocated()) {
		endFrameImage.resize(width, height);
		endFramePixels = endFrameImage.getPixels();
		endInputImage = {
			static_cast<uint32_t>(endFramePixels.getWidth()),
			static_cast<uint32_t>(endFramePixels.getHeight()),
			static_cast<uint32_t>(endFramePixels.getNumChannels()),
			endFramePixels.getData()
		};
	}
	if (maskGuideImage.isAllocated()) {
		maskGuideImage.resize(width, height);
		maskGuidePixels = maskGuideImage.getPixels();
		maskGuideInput = {
			static_cast<uint32_t>(maskGuidePixels.getWidth()),
			static_cast<uint32_t>(maskGuidePixels.getHeight()),
			static_cast<uint32_t>(maskGuidePixels.getNumChannels()),
			maskGuidePixels.getData()
		};
	}
	if (controlGuideImage.isAllocated()) {
		controlGuideImage.resize(width, height);
		controlGuidePixels = controlGuideImage.getPixels();
		controlGuideInput = {
			static_cast<uint32_t>(controlGuidePixels.getWidth()),
			static_cast<uint32_t>(controlGuidePixels.getHeight()),
			static_cast<uint32_t>(controlGuidePixels.getNumChannels()),
			controlGuidePixels.getData()
		};
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch (key) {
	case 'e': { // reload embeddings
		stableDiffusion.reloadEmbeddings(embedDir);
		ofLogNotice("ofApp") << "Reloaded embeddings from " << embedDir;
		break;
	}
	case 'E': { // list embeddings
		auto embeds = stableDiffusion.listEmbeddings();
		ofLogNotice("ofApp") << "Embeddings (" << embeds.size() << "):";
		for (const auto& e : embeds) {
			ofLogNotice("ofApp") << "  " << e.first << " -> " << e.second;
		}
		break;
	}
	case 'l': { // list loras
		auto entries = listLoraFiles();
		ofLogNotice("ofApp") << "LoRAs (" << entries.size() << ") in " << loraModelDir;
		for (const auto& e : entries) {
			ofLogNotice("ofApp") << "  " << e.first << " -> " << e.second;
		}
		break;
	}
	case 'a': { // apply all loras with default strength
		loadAllLoras(1.0f);
		ofLogNotice("ofApp") << "Applied " << loras.size() << " LoRAs";
		break;
	}
	case 'u': { // unload loras
		clearLoras();
		ofLogNotice("ofApp") << "Cleared active LoRAs";
		break;
	}
	default:
		break;
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//------------- -------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}
