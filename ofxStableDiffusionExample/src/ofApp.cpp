#include "ofApp.h"

//--------------------------------------------------------------
void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
	if (level <= SD_LOG_INFO) {
		fputs(log, stdout);
		fflush(stdout);
	}
	else {
		fputs(log, stderr);
		fflush(stderr);
	}
}

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxStableDiffusionExample");
	ofSetEscapeQuitsApp(false);
	ofSetWindowPosition((ofGetScreenWidth() - ofGetWindowWidth()) / 2, (ofGetScreenHeight() - ofGetWindowHeight()) / 2);
	ofDisableArbTex();
	printf("%s", sd_get_system_info());
	sd_set_log_callback(sd_log_cb, NULL);
	modelPath = "data/models/sd_xl_base_1.0.safetensors";
	modelName = "sd_xl_base_1.0.safetensors";
	controlNetPath = "";
	embedDir = "data/models/embeddings";
	taesdPath = "";
	loraModelDir = "data/models/lora";
	vaePath = "data/models/vae/vae.safetensors";
	prompt = "man img, man with futuristic clothes";
	esrganPath = "data/models/esrgan/RealESRGAN_x4plus_anime_6B.pth";
	controlImagePath = "data/control_2.png";
	stackedIdEmbedDir = ""; //"data/models/photomaker/photomaker-v1.safetensors";
	inputIdImagesPath = "data/photomaker_images/newton_man";
	keepClipOnCpu = false;
	keepControlNetCpu = false;
	keepVaeOnCpu = false;
	styleStrength = 20;
	normalizeInput = true;
	width = 512;
	height = 512;
	cfgScale = 7.0;
	sampleSteps = 15;
	clipSkipLayers = -1;
	previewSize = batchSize = 4;
	selectedImage = 0;
	strength = 0.5;
	controlStrength = 0.9;
	seed = -1;
	sdType = SD_TYPE_COUNT;
	schedule = DEFAULT;
	rngType = CUDA_RNG;
	imageWidth = "512";
	imageHeight = "512";
	sampleMethod = "DPMPP2S_Mv2";
	promptIsEdited = true;
	negativePromptIsEdited = true;
	isTextToImage = true;
	isFullScreen = false;
	isTAESD = false;
	isESRGAN = false;
	isVaeDecodeOnly = false;
	isVaeTiling = false;
	isFreeParamsImmediatly = false;
	numThreads = 8;
	esrganMultiplier = 4;
	for (int i = 0; i < 16; i++) {
		ofTexture texture;
		textureVector.push_back(texture);
	}
	allocate();
	if (!thread.isThreadRunning()) {
		isModelLoading = true;
		thread.userData = this;
		thread.startThread();
	}
	gui.setup(nullptr, true, ImGuiConfigFlags_None, true);
}

//--------------------------------------------------------------
void ofApp::update() {
	if (diffused) {
		for (int i = 0; i < batchSize; i++) {
			if (isESRGAN) {
				textureVector[i].loadData(outputImages[i].data, width * esrganMultiplier, height * esrganMultiplier, GL_RGB);
			}
			else {
				textureVector[i].loadData(outputImages[i].data, width, height, GL_RGB);
			}
		}
		previousSelectedImage = 0;
		previewSize = batchSize;
		diffused = false;
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
	ImGui::SetNextWindowPos(ImVec2(828, 20), ImGuiCond_Once);
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
		if (thread.isThreadRunning()) {
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
				if (!thread.isThreadRunning()) {
					isModelLoading = true;
					thread.userData = this;
					thread.startThread();
				}
			}
		}
		ImGui::SameLine(0, 5);
		ImGui::Text(&modelName[0]);
		ImGui::Dummy(ImVec2(0, 10));
		static int e = 0;
		if (ImGui::RadioButton("Text to Image", &e, 0)) {
			isTextToImage = true;
			batchSize = 4;
		}
		ImGui::SameLine(0, 10);
		if (ImGui::RadioButton("Image to Image", &e, 1)) {
			isTextToImage = false;
			batchSize = 4;
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Checkbox("TAESD", &isTAESD)) {
			if (!thread.isThreadRunning() && isTAESD) {
				taesdPath = "data/models/taesd/taesd.safetensors";
				isModelLoading = true;
				thread.userData = this;
				thread.startThread();
			}
			else if (!thread.isThreadRunning() && !isTAESD) {
				taesdPath = "";
				isModelLoading = true;
				thread.userData = this;
				thread.startThread();
			}
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Checkbox("VAE Tiling", &isVaeTiling)) {
				if (!thread.isThreadRunning()) {
					isModelLoading = true;
					thread.userData = this;
					thread.startThread();
				}
		}
		if (!isTextToImage) {
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
						//delete control_image;
						//control_image = NULL;
						//control_image = new sd_image_t{ (uint32_t)width,
						//  (uint32_t)height,
						//  3,
						//  pixels.getData() };
					}
				}
			}
			ImGui::SameLine(0, 5);
			ImGui::Text(&imageName[0]);
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::Image((ImTextureID)(uintptr_t)fbo.getTexture().getTextureData().textureID, ImVec2(128, 128));
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderInt("Clip Skip Layers", &clipSkipLayers, -1, 33);
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
		if (!isTextToImage) {
			ImGui::Dummy(ImVec2(0, 10));
			ImGui::SliderFloat("Strength", &strength, 0, 1);
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::InputInt("Seed", &seed, 1, 100);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderInt("Batch Size", &batchSize, 1, 16);
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Checkbox("4x Upscale", &isESRGAN)) {
			allocate();
		}
		if (thread.isThreadRunning()) {
			ImGui::EndDisabled();
		}
		ImGui::TreePop();
	}
	if (thread.isThreadRunning()) {
		ImGui::BeginDisabled();
	}
	ImGui::Dummy(ImVec2(0, 10));
	if (ImGui::Button("Generate")) {
		if (!thread.isThreadRunning()) {
			thread.userData = this;
			thread.startThread();
		}
	}
	if (thread.isThreadRunning()) {
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
			int lastSpace = tmpStr.size() - 1;
			while (tmpStr[lastSpace] != ' ' && lastSpace > 0) {
				lastSpace--;
			}
			if (lastSpace == 0) {
				lastSpace = tmpStr.size() - 2;
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
void ofApp::allocate() {
	for (int i = 0; i < 16; i++) {
		if (textureVector[i].isAllocated()) {
			textureVector[i].getTextureData().textureID = NULL;
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