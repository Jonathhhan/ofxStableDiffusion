#include "ofApp.h"

#include <algorithm>
#include <cctype>
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

float computeBrightnessScore(const ofPixels& pixels) {
	if (!pixels.isAllocated() || pixels.getNumChannels() < 3) {
		return 0.0f;
	}
	const std::size_t pixelCount =
		static_cast<std::size_t>(pixels.getWidth()) *
		static_cast<std::size_t>(pixels.getHeight());
	if (pixelCount == 0) {
		return 0.0f;
	}
	const unsigned char* data = pixels.getData();
	const std::size_t channels =
		static_cast<std::size_t>(pixels.getNumChannels());
	double brightnessSum = 0.0;
	double colorfulnessSum = 0.0;
	for (std::size_t i = 0; i < pixelCount; ++i) {
		const std::size_t offset = i * static_cast<std::size_t>(channels);
		const float r = static_cast<float>(data[offset + 0]) / 255.0f;
		const float g = static_cast<float>(data[offset + 1]) / 255.0f;
		const float b = static_cast<float>(data[offset + 2]) / 255.0f;
		const float maxChannel = std::max(r, std::max(g, b));
		const float minChannel = std::min(r, std::min(g, b));
		brightnessSum += (r + g + b) / 3.0f;
		colorfulnessSum += (maxChannel - minChannel);
	}
	const float meanBrightness =
		static_cast<float>(brightnessSum / static_cast<double>(pixelCount));
	const float meanColorfulness =
		static_cast<float>(colorfulnessSum / static_cast<double>(pixelCount));
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

} // namespace

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxStableDiffusionExample");
	ofSetEscapeQuitsApp(false);
	ofSetWindowPosition((ofGetScreenWidth() - ofGetWindowWidth()) / 2, (ofGetScreenHeight() - ofGetWindowHeight()) / 2);
	ofDisableArbTex();
	printf("%s", stableDiffusion.getSystemInfo());
	modelPath = "data/models/sd_turbo.safetensors";
	modelName = "sd_turbo.safetensors";
	controlNetPath = ""; // "data/models/controlnet/control_v11p_sd15_openpose_2.safetensors";
	embedDir = "";
	taesdPath = "";
	loraModelDir = ""; // "data/models/lora";
	vaePath = ""; // "data/models/vae/vae.safetensors";
	prompt = "animal with futuristic clothes"; // "man img, man with futuristic clothes";
	instruction = "edit the source image so the subject feels more futuristic while keeping the composition readable";
	rankingPrompt = "bright futuristic vivid cinematic";
	esrganPath = ""; // "data/models/esrgan/RealESRGAN_x4plus_anime_6B.pth"; // "data/models/esrgan/RealESRGAN_x4plus_anime_6B.pth";
	controlImagePath = ""; // "data/openpose.jpeg";
	stackedIdEmbedDir = ""; // "data/models/photomaker/photomaker-v1.safetensors";
	inputIdImagesPath = ""; // "data/photomaker_images/newton_man";
	keepClipOnCpu = false;
	keepControlNetCpu = false;
	keepVaeOnCpu = false;
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
	videoMode = "Standard";
	promptIsEdited = true;
	negativePromptIsEdited = true;
	isTextToImage = true;
	isInstructImage = false;
	isImageToVideo = false;
	videoFrames = 6;
	motionBucketId = 127;
	videoFps = 6;
	augmentationLevel = 0.0f;
	minCfg = 1.0f;
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
	stableDiffusion.newSdCtx(modelPath,
		vaePath,
		taesdPath,
		controlNetPath,
		loraModelDir,
		embedDir,
		stackedIdEmbedDir,
		vaeDecodeOnly,
		vaeTiling,
		freeParamsImmediately,
		nThreads,
		wType,
		rngType,
		schedule,
		keepClipOnCpu,
		keepControlNetCpu,
		keepVaeOnCpu);
}

//--------------------------------------------------------------
void ofApp::update() {
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
		if (ImGui::Button("Load Model")) {
			ofFileDialogResult result = ofSystemLoadDialog("Load Model", false, "");
			if (result.bSuccess) {
				modelPath = result.getPath();
				modelName = result.getName();
				stableDiffusion.newSdCtx(modelPath,
					vaePath,
					taesdPath,
					controlNetPath,
					loraModelDir,
					embedDir,
					stackedIdEmbedDir,
					vaeDecodeOnly,
					vaeTiling,
					freeParamsImmediately,
					nThreads,
					wType,
					rngType,
					schedule,
					keepClipOnCpu,
					keepControlNetCpu,
					keepVaeOnCpu);
			}
		}
		ImGui::SameLine(0, 5);
		ImGui::Text(&modelName[0]);
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
		if (isInstructImage) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::TextWrapped("InstructImage uses the prompt as overall target context and the instruction field as the concrete edit request.");
		} else if (imageModeEnum == ofxStableDiffusionImageMode::Variation) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::TextWrapped("Variation keeps the source composition closer and uses a lighter denoise strength.");
		} else if (imageModeEnum == ofxStableDiffusionImageMode::Restyle) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::TextWrapped("Restyle pushes the source image further toward the prompt aesthetic.");
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
		if (ImGui::Checkbox("VAE Tiling", &vaeTiling)) {
				if (!isBusy) {

				}
		}
		if (usesInputImageMode() || isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			if (ImGui::Button("Load Image")) {
				ofFileDialogResult result = ofSystemLoadDialog("Load Image", false, "");
				if (result.bSuccess) {
					imageName = result.getName();
					ofFile file(result.getPath());
					string fileExtension = ofToUpper(file.getExtension());
					if (fileExtension == "JPG" || fileExtension == "JPEG" || fileExtension == "PNG") {
						image.load(result.getPath());


						fbo.begin();
						image.draw(0, 0, width, height);
						fbo.end();
						fbo.getTexture().readToPixels(pixels);
						inputImage = { (uint32_t)width,
						  (uint32_t)height,
						  3,
						  pixels.getData() };
					}
				}
			}
			ImGui::SameLine(0, 5);
			ImGui::Text(&imageName[0]);
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Image((ImTextureID)(uintptr_t)fbo.getTexture().getTextureData().textureID, ImVec2(128, 128));
		}
		if (isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderInt("Video Frames", &videoFrames, 1, 16);
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderInt("Motion Bucket ID", &motionBucketId, 1, 255);
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderInt("FPS", &videoFps, 1, 30);
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat("Augmentation Level", &augmentationLevel, 0.0f, 1.0f);
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat("Min CFG", &minCfg, 0.0f, 20.0f);
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
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderInt("Clip Skip Layers", &clipSkip, -1, 33);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderFloat("Control Strength", &controlStrength, 0, 2);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderFloat("CFG Scale", &cfgScale, 0, 20);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderInt("Sample Steps", &sampleSteps, 1, 50);
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
		if (usesInputImageMode() || isImageToVideo) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat("Strength", &strength, 0, 1);
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
	if (ImGui::Button("Generate")) {
		progressStep = 0;
		progressSteps = 0;
		progressTime = 0.0f;
		configureExampleRanker();
		if (isImageToVideo) {
			isPlaying = false;
			totalVideoFrames = 0;
			stableDiffusion.img2vid(inputImage,
				width,
				height,
				videoFrames,
				motionBucketId,
				videoFps,
				augmentationLevel,
				minCfg,
				cfgScale,
				sampleMethodEnum,
				sampleSteps,
				strength,
				seed);
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
			request.controlCond = controlImage;
			request.controlStrength = controlStrength;
			request.styleStrength = styleStrength;
			request.normalizeInput = normalizeInput;
			request.inputIdImagesPath = inputIdImagesPath;
			stableDiffusion.generate(request);
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
			stableDiffusion.saveVideoFrames(
				ofGetTimestampString("output/ofxStableDiffusion-video-%Y-%m-%d-%H-%M-%S"),
				"frame");
		}
	}
	if (isBusy) {
		ImGui::EndDisabled();
	}
	ImGui::End();
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
	cfgScale = ofxStableDiffusionDefaultCfgScaleForImageMode(mode);
	if (usesInputImageMode()) {
		strength = ofxStableDiffusionDefaultStrengthForImageMode(mode);
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
		delete controlImage;
		controlImage = NULL;
		controlImage = new sd_image_t{ (uint32_t)width,
		  (uint32_t)height,
		  3,
		  pixels.getData() };
	}
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

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
