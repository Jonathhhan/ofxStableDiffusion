#include "ofApp.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <sstream>

namespace {

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

bool loadImageFile(
	const std::string& path,
	int width,
	int height,
	ofImage& targetImage,
	ofPixels& targetPixels,
	sd_image_t& targetSdImage) {
	ofFile file(path);
	const std::string extension = ofToUpper(file.getExtension());
	if (extension != "JPG" && extension != "JPEG" && extension != "PNG") {
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
	settings.backend = backendPreference;
	settings.rngType = rngType;
	settings.schedule = schedule;
	settings.keepClipOnCpu = keepClipOnCpu;
	settings.keepControlNetCpu = keepControlNetCpu;
	settings.keepVaeOnCpu = keepVaeOnCpu;
	settings.offloadParamsToCpu = offloadParamsToCpu;
	return settings;
}

//--------------------------------------------------------------
void ofApp::refreshModelContext() {
	modelName = ofFilePath::getFileName(modelPath.empty() ? diffusionModelPath : modelPath);
	stableDiffusion.newSdCtx(buildContextSettings());
	applyRecommendedImageParameters();
	applyRecommendedVideoParameters();
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
	printf("%s", stableDiffusion.getSystemInfo());
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
	styleStrength = 20;
	normalizeInput = true;
	width = 512;
	height = 512;
	cfgScale = 1.0;
	sampleSteps = 5;
	clipSkip = -1;
	previewSize = batchCount = 4;
	selectedImage = 0;
	strength = 0.5;
	controlStrength = 0.9;
	seed = -1;
	wType = SD_TYPE_F16;
	schedule = SCHEDULER_COUNT;
	rngType = STD_DEFAULT_RNG;
	imageWidth = "512";
	imageHeight = "512";
	imageMode = "TextToImage";
	imageModeEnum = ofxStableDiffusionImageMode::TextToImage;
	selectionMode = "KeepOrder";
	selectionModeEnum = ofxStableDiffusionImageSelectionMode::KeepOrder;
	sampleMethod = "DPMPP2Mv2_SAMPLE_METHOD";
	sampleMethodEnum = DPMPP2Mv2_SAMPLE_METHOD;
	backendMode = "CUDA";
	backendPreference = SD_BACKEND_CUDA;
	videoMode = "Standard";
	interpolationMode = "Smooth";
	promptIsEdited = true;
	negativePromptIsEdited = true;
	isTextToImage = true;
	isInstructImage = false;
	isImageToVideo = false;
	videoFrames = 6;
	videoFps = 6;
	vaceStrength = 1.0f;
	enablePromptInterpolation = false;
	useSeedSequence = false;
	useEndFrame = false;
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
	freeParamsImmediately = false;
	nThreads = 8;
	esrganMultiplier = 4;
	for (int i = 0; i < maxPreviewTextures; i++) {
		ofTexture texture;
		textureVector.push_back(texture);
	}
	allocate();
	gui.setup(nullptr, true, ImGuiConfigFlags_None, true);
	applySelectedImageMode(imageModeEnum);
	stableDiffusion.setImageSelectionMode(selectionModeEnum);
	stableDiffusion.setProgressCallback([this](int step, int steps, float time) {
		progressStep = step;
		progressSteps = steps;
		progressTime = time;
	});
	configureExampleRanker();
	setupHoloscanBridge();

	// Initial embedding enumeration (best-effort).
	auto embeds = stableDiffusion.listEmbeddings();
	ofLogNotice("ofApp") << "Found " << embeds.size() << " embeddings in " << embedDir;
	ofLogNotice("ofApp") << "Keys: 'e' reload embeddings, 'E' list embeddings, 'l' list LoRAs, 'a' apply all LoRAs (strength 1.0), 'u' unload LoRAs";
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
		outputImages = stableDiffusion.returnImages();
		if (isImageToVideo) {
			totalVideoFrames = stableDiffusion.getOutputCount();
			for (int i = 0; i < totalVideoFrames; i++) {
				textureVector[i].loadData(outputImages[i].data, width, height, GL_RGB);
			}
			previousSelectedImage = 0;
			currentFrame = 0;
			previewSize = totalVideoFrames;
			isPlaying = true;
			lastFrameTime = ofGetElapsedTimef();
		}
		else {
			const int outputCount = stableDiffusion.getOutputCount();
			for (int i = 0; i < outputCount; i++) {
				if (isESRGAN) {
					textureVector[i].loadData(outputImages[i].data, width * esrganMultiplier, height * esrganMultiplier, GL_RGB);
				}
				else {
					textureVector[i].loadData(outputImages[i].data, width, height, GL_RGB);
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
		for (int i = 0; i < previewSize; i++) {
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
		ImGui::Image((ImTextureID)(uintptr_t)textureVector[selectedImage].getTextureData().textureID, ImVec2(width, height));
		const auto& result = stableDiffusion.getLastResult();
		if (selectedImage >= 0 && selectedImage < static_cast<int>(result.images.size())) {
			const auto& frame = result.images[static_cast<std::size_t>(selectedImage)];
			if (frame.score.valid) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::Text("Score: %.3f", frame.score.score);
				if (!frame.score.scorer.empty()) {
					ImGui::SameLine(0, 10);
					ImGui::Text("Scorer: %s", frame.score.scorer.c_str());
				}
			}
			if (frame.isSelected) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::Text("Best-of-N Selection");
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Indent(-10);
		if (ImGui::Button("Save")) {
			textureVector[selectedImage].readToPixels(pixels);
			ofSaveImage(pixels, ofGetTimestampString("output/ofxStableDiffusion-%Y-%m-%d-%H-%M-%S.png"));
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
		const auto capabilities = stableDiffusion.getCapabilities();
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
		if (ImGui::BeginCombo("Backend", backendMode, ImGuiComboFlags_NoArrowButton)) {
			for (int n = 0; n < IM_ARRAYSIZE(backendArray); n++) {
				const bool is_selected = (backendMode == backendArray[n]);
				if (ImGui::Selectable(backendArray[n], is_selected)) {
					backendMode = backendArray[n];
					backendPreference =
						n == 0 ? SD_BACKEND_CUDA :
						n == 1 ? SD_BACKEND_VULKAN :
						SD_BACKEND_CPU;
				}
				if (is_selected) {
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::TextWrapped("CUDA prefers CUDA first and falls back to Vulkan, then CPU if needed.");
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Button("Load Context")) {
			refreshModelContext();
		}
		ImGui::SameLine(0, 5);
		if (ImGui::Button("Unload Context")) {
			stableDiffusion.freeSdCtx();
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (stableDiffusion.getLastError().empty()) {
			ImGui::Text("Status: %s", isBusy ? "Running" : "Idle");
		} else {
			ImGui::TextWrapped("Status: %s", stableDiffusion.getLastError().c_str());
		}
		if (progressSteps > 0 && isBusy) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Text("Progress %d / %d (%.2fs)", progressStep, progressSteps, progressTime);
		}
		ImGui::Dummy(ImVec2(0, 10));
		drawHoloscanBridgeSection();
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::BeginCombo("Image Mode", imageMode, ImGuiComboFlags_NoArrowButton)) {
			for (int n = 0; n < IM_ARRAYSIZE(imageModeArray); n++) {
				const bool is_selected = (imageMode == imageModeArray[n]);
				if (ImGui::Selectable(imageModeArray[n], is_selected)) {
					applySelectedImageMode(static_cast<ofxStableDiffusionImageMode>(n));
				}
				if (is_selected) {
					ImGui::SetItemDefaultFocus();
				}
			}
			ImGui::EndCombo();
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
			ImGui::Dummy(ImVec2(0, 10));
			if (selectionModeEnum != ofxStableDiffusionImageSelectionMode::KeepOrder) {
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
				if (useDemoRanking) {
					ImGui::TextWrapped(
						"This example scorer is local and heuristic. It demonstrates the Best-of-N API, but real semantic reranking belongs on the ofxGgml CLIP side.");
				} else {
					ImGui::TextWrapped(
						"Attach a CLIP scorer from ofxGgml for semantic Best-of-N reranking, or enable the demo callback here to preview the ranking workflow.");
				}
			} else {
				ImGui::TextWrapped("KeepOrder leaves outputs untouched. Switch to Rerank or BestOnly to exercise the ranking API.");
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Checkbox("Generate Video", &isImageToVideo);
		if (ImGui::IsItemEdited()) {
			if (isImageToVideo) {
				applyRecommendedVideoParameters();
			} else {
				applyRecommendedImageParameters();
			}
			clampCurrentParametersToProfiles();
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
		if (ImGui::Button(isImageToVideo ? "Apply Recommended Video Tuning" : "Apply Recommended Image Tuning")) {
			if (isImageToVideo) {
				applyRecommendedVideoParameters();
			} else {
				applyRecommendedImageParameters();
			}
		}
		if (isInstructImage) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::TextWrapped("InstructImage uses the prompt as overall target context and the instruction field as the concrete edit request.");
		} else if (imageModeEnum == ofxStableDiffusionImageMode::Variation) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::TextWrapped("Variation keeps the source composition closer and uses a lighter denoise strength.");
		} else if (imageModeEnum == ofxStableDiffusionImageMode::Restyle) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::TextWrapped("Restyle pushes the source image further toward the prompt aesthetic.");
		} else if (imageModeEnum == ofxStableDiffusionImageMode::Inpainting) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::TextWrapped("Inpainting uses the source image plus a mask image. White areas are repainted and black areas are preserved.");
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
		if (ImGui::Checkbox("VAE Tiling", &vaeTiling)) {
				if (!isBusy) {

				}
		}
		if (usesInputImageMode() || isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			if (ImGui::Button("Load Image")) {
				if (loadImageIntoSlot("Load Image", image, pixels, inputImage, imageName)) {
					allocate();
				}
			}
			ImGui::SameLine(0, 5);
			ImGui::Text(&imageName[0]);
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
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderInt(
				"Video Frames",
				&videoFrames,
				videoParameterProfile.minFrameCount,
				videoParameterProfile.maxFrameCount);
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderInt(
				"FPS",
				&videoFps,
				videoParameterProfile.minFps,
				videoParameterProfile.maxFps);
			if (videoParameterProfile.supportsVaceStrength) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::SliderFloat(
					"VACE Strength",
					&vaceStrength,
					videoParameterProfile.minVaceStrength,
					videoParameterProfile.maxVaceStrength);
			}
			ImGui::Dummy(ImVec2(0, 10));
			if (ImGui::BeginCombo("Video Mode", videoMode, ImGuiComboFlags_NoArrowButton)) {
				for (int n = 0; n < IM_ARRAYSIZE(videoModeArray); n++) {
					const bool is_selected = (videoMode == videoModeArray[n]);
					if (ImGui::Selectable(videoModeArray[n], is_selected)) {
						videoMode = videoModeArray[n];
						stableDiffusion.setVideoGenerationMode(static_cast<ofxStableDiffusionVideoMode>(n));
					}
					if (is_selected) {
						ImGui::SetItemDefaultFocus();
					}
				}
				ImGui::EndCombo();
			}
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Checkbox("Prompt Morph", &enablePromptInterpolation);
			if (enablePromptInterpolation) {
				ImGui::Dummy(ImVec2(0, 10));
				Funcs::MyInputTextMultiline(
					"Prompt B",
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
			ImGui::Checkbox("Seed Sequence", &useSeedSequence);
			if (useSeedSequence) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::InputInt("Seed Increment", &seedIncrement, 1, 10);
			}
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
		if (isImageToVideo ? videoParameterProfile.supportsClipSkip : imageParameterProfile.supportsClipSkip) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderInt(
				"Clip Skip Layers",
				&clipSkip,
				isImageToVideo ? videoParameterProfile.minClipSkip : imageParameterProfile.minClipSkip,
				isImageToVideo ? videoParameterProfile.maxClipSkip : imageParameterProfile.maxClipSkip);
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderFloat("Control Strength", &controlStrength, 0, 2);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderFloat(
			"CFG Scale",
			&cfgScale,
			isImageToVideo ? videoParameterProfile.minCfgScale : imageParameterProfile.minCfgScale,
			isImageToVideo ? videoParameterProfile.maxCfgScale : imageParameterProfile.maxCfgScale);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderInt(
			"Sample Steps",
			&sampleSteps,
			isImageToVideo ? videoParameterProfile.minSampleSteps : imageParameterProfile.minSampleSteps,
			isImageToVideo ? videoParameterProfile.maxSampleSteps : imageParameterProfile.maxSampleSteps);
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::BeginCombo("Width", imageWidth, ImGuiComboFlags_NoArrowButton)) {
			for (int n = 0; n < IM_ARRAYSIZE(imageSizeArray); n++) {
				bool is_selected = (imageWidth == imageSizeArray[n]);
				if (ImGui::Selectable(imageSizeArray[n], is_selected))
					imageWidth = imageSizeArray[n];
				if (is_selected)
					ImGui::SetItemDefaultFocus();
				if (width != atoi(imageWidth)) {
					width = atoi(imageWidth);
					allocate();
				}
			}
			ImGui::EndCombo();
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::BeginCombo("Height", imageHeight, ImGuiComboFlags_NoArrowButton)) {
			for (int n = 0; n < IM_ARRAYSIZE(imageSizeArray); n++) {
				bool is_selected = (imageHeight == imageSizeArray[n]);
				if (ImGui::Selectable(imageSizeArray[n], is_selected))
					imageHeight = imageSizeArray[n];
				if (is_selected)
					ImGui::SetItemDefaultFocus();
				if (height != atoi(imageHeight)) {
					height = atoi(imageHeight);
					allocate();
				}
			}
			ImGui::EndCombo();
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::BeginCombo("Sample Method", sampleMethod, ImGuiComboFlags_NoArrowButton)) {
			for (int n = 0; n < IM_ARRAYSIZE(sampleMethodArray); n++) {
				bool is_selected = (sampleMethod == sampleMethodArray[n]);
				if (ImGui::Selectable(sampleMethodArray[n], is_selected)) {
					sampleMethod = sampleMethodArray[n];
					sampleMethodEnum = (sample_method_t)n;
				}
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
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
	const auto capabilitiesForGenerate = stableDiffusion.getCapabilities();
	const bool modelReady = capabilitiesForGenerate.contextConfigured;
	const bool needsInputImage = usesInputImageMode() || isImageToVideo;
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
	const bool disableGenerate = isBusy || !modelReady || (needsInputImage && !hasInputImage) || !maskReady;
	ImGui::Dummy(ImVec2(0, 10));
	if (disableGenerate) {
		ImGui::BeginDisabled();
	}
	if (ImGui::Button("Generate")) {
		progressStep = 0;
		progressSteps = 0;
		progressTime = 0.0f;
		configureExampleRanker();
		clampCurrentParametersToProfiles();
		if (isImageToVideo) {
			isPlaying = false;
			totalVideoFrames = 0;
			ofxStableDiffusionVideoRequest request;
			request.initImage = inputImage;
			request.endImage = (useEndFrame && endInputImage.data != nullptr) ? endInputImage : sd_image_t{0, 0, 0, nullptr};
			request.prompt = prompt;
			request.negativePrompt = negativePrompt;
			request.clipSkip = clipSkip;
			request.width = width;
			request.height = height;
			request.frameCount = videoFrames;
			request.fps = videoFps;
			request.cfgScale = cfgScale;
			request.sampleMethod = sampleMethodEnum;
			request.sampleSteps = sampleSteps;
			request.strength = strength;
			request.seed = seed;
			request.vaceStrength = vaceStrength;
			request.mode = static_cast<ofxStableDiffusionVideoMode>(
				std::distance(std::begin(videoModeArray),
					std::find(std::begin(videoModeArray), std::end(videoModeArray), videoMode)));
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
			request.strength = strength;
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
	if (disableGenerate) {
		ImGui::EndDisabled();
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
		const auto clip = stableDiffusion.getVideoClip();
		if (currentFrame >= 0 && currentFrame < static_cast<int>(clip.frames.size())) {
			const auto& frame = clip.frames[static_cast<std::size_t>(currentFrame)];
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Text("Seed: %lld", static_cast<long long>(frame.seed));
			if (!frame.generation.prompt.empty()) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::TextWrapped("Frame Prompt: %s", frame.generation.prompt.c_str());
			}
			if (!frame.generation.negativePrompt.empty()) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::TextWrapped("Frame Negative Prompt: %s", frame.generation.negativePrompt.c_str());
			}
			if (frame.generation.cfgScale >= 0.0f || frame.generation.strength >= 0.0f) {
				ImGui::Dummy(ImVec2(0, 10));
				ImGui::Text(
					"CFG %.2f  Strength %.2f",
					frame.generation.cfgScale,
					frame.generation.strength);
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
	cfgScale = imageParameterProfile.defaultCfgScale;
	sampleSteps = imageParameterProfile.defaultSampleSteps;
	clipSkip = imageParameterProfile.defaultClipSkip;
	if (imageParameterProfile.supportsStrength) {
		strength = imageParameterProfile.defaultStrength;
	}
}

//--------------------------------------------------------------
void ofApp::applyRecommendedVideoParameters() {
	videoParameterProfile = ofxStableDiffusionParameterTuningHelpers::resolveVideoProfile(
		stableDiffusion.getContextSettings());
	cfgScale = videoParameterProfile.defaultCfgScale;
	sampleSteps = videoParameterProfile.defaultSampleSteps;
	strength = videoParameterProfile.defaultStrength;
	clipSkip = videoParameterProfile.defaultClipSkip;
	videoFrames = videoParameterProfile.defaultFrameCount;
	videoFps = videoParameterProfile.defaultFps;
	vaceStrength = videoParameterProfile.defaultVaceStrength;
}

//--------------------------------------------------------------
void ofApp::clampCurrentParametersToProfiles() {
	imageParameterProfile = ofxStableDiffusionParameterTuningHelpers::resolveImageProfile(
		stableDiffusion.getContextSettings(),
		imageModeEnum);
	videoParameterProfile = ofxStableDiffusionParameterTuningHelpers::resolveVideoProfile(
		stableDiffusion.getContextSettings());

	if (isImageToVideo) {
		ofxStableDiffusionParameterTuningHelpers::clampVideoParametersToProfile(
			videoParameterProfile,
			cfgScale,
			sampleSteps,
			strength,
			clipSkip,
			vaceStrength,
			videoFrames,
			videoFps);
	} else {
		ofxStableDiffusionParameterTuningHelpers::clampImageParametersToProfile(
			imageParameterProfile,
			cfgScale,
			sampleSteps,
			strength,
			clipSkip);
	}
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
void ofApp::allocate() {
	for (int i = 0; i < maxPreviewTextures; i++) {
		if (textureVector[i].isAllocated()) {
			textureVector[i].clear();
		}
		if (isESRGAN) {
			textureVector[i].allocate(width * esrganMultiplier, height * esrganMultiplier, GL_RGB);
		} else {
			textureVector[i].allocate(width, height, GL_RGB);
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
