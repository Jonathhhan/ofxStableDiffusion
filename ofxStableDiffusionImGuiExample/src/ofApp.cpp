#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxStableDiffusionExample");
	ofSetEscapeQuitsApp(false);
	ofSetWindowPosition((ofGetScreenWidth() - ofGetWindowWidth()) / 2, (ofGetScreenHeight() - ofGetWindowHeight()) / 2);
	printf("%s", sd_get_system_info().c_str());
	// set_sd_log_level(INFO);
	ofSetWindowShape(ofGetScreenWidth(), ofGetScreenHeight());
	thread.stableDiffusion.setup(8, true, "data/models/taesd/diffusion_pytorch_model.SAFETENSORS", false, "data/models/lora/", STD_DEFAULT_RNG);
	thread.stableDiffusion.load_from_file("data/models/v1-5-pruned-emaonly.SAFETENSORS", "", GGML_TYPE_COUNT, DEFAULT);
	gui.setup(nullptr, true, ImGuiConfigFlags_None, true);
	prompt = "<lora:ohara_koson:1>mushroom, ohara koson, traditional media, botanic painting";
	modelName = "v1-5-pruned-emaonly.safetensors";
	width = 512;
	height = 512;
	cfgScale = 7.0;
	sampleSteps = 10;
	previewSize = batchSize = 4;
	selectedImage = 0;
	strength = 0.5;
	imageWidth = "512";
	imageHeight = "512";
	sampleMethod = "DPMPP2S_Mv2";
	promptIsEdited = true;
	negativePromptIsEdited = true;
	isTextToImage = true;
	isFullScreen = false;
	ofFbo::Settings fboSettings;
	fboSettings.width = width;
	fboSettings.height = height;
	fboSettings.internalformat = GL_RGB;
	fboSettings.textureTarget = GL_TEXTURE_2D;
	for (int i = 0; i < 16; i++) {
		ofFbo fbo;
		fboVector.push_back(fbo);
		fboVector[i].allocate(fboSettings);
	}
	fbo.allocate(fboSettings);
	fbo.begin();
	image.draw(0, 0, width, height);
	fbo.end();
}

//--------------------------------------------------------------
void ofApp::update() {
	if (thread.diffused) {
		for (int i = 0; i < batchSize; i++) {
			fboVector[i].getTexture().loadData(&thread.stableDiffusionPixelVector[i][0], width, height, GL_RGB);
		}
		previousSelectedImage = 0;
		previewSize = batchSize;
		thread.diffused = false;
	}
	selectedImage = previousSelectedImage;
}

//--------------------------------------------------------------
void ofApp::draw() {
	ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse;
	struct Funcs
	{
		static int InputTextCallback(ImGuiInputTextCallbackData* data)
		{
			if (data->EventFlag == ImGuiInputTextFlags_CallbackResize)
			{
				std::string* str = (std::string*)data->UserData;
				IM_ASSERT(data->Buf == str->c_str());
				str->resize(data->BufTextLen);
				data->Buf = (char*)str->c_str();
			}
			return 0;
		}

		static bool MyInputTextMultiline(const char* label, std::string* prompt, const ImVec2& size = ImVec2(0, 0), ImGuiInputTextFlags flags = 0)
		{
			IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
			return ImGui::InputTextMultiline(label, (char*)prompt->c_str(), prompt->capacity() + 1, size, flags | ImGuiInputTextFlags_CallbackResize, Funcs::InputTextCallback, (void*)prompt);
		}
	};

	gui.begin();
	static bool logOpenSettings{ true };
	ImVec2 center = ImVec2(ofGetScreenWidth() / 2, ofGetScreenHeight() / 2);
	ImGui::StyleColorsDark();
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
	ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 0);
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
	ImGui::PushStyleVar(ImGuiStyleVar_ItemInnerSpacing, ImVec2(5, 0));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowMinSize, ImVec2(0, 0));

	ImGui::SetNextWindowSizeConstraints(ImVec2(20 + width, -1.f), ImVec2(INFINITY, -1.f));
	ImGui::SetNextWindowPos(ImVec2(center.x / 1.75, center.y), ImGuiCond_Once, ImVec2(0.5f, 0.5f));
	ImGui::Begin("ofxStableDiffusion##foo1", NULL, flags | ImGuiWindowFlags_NoBringToFrontOnFocus);
	if (ImGui::TreeNodeEx("Image Preview", ImGuiStyleVar_WindowPadding | ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Indent((ImGui::GetWindowSize().x - width) / 2);
		for (int i = 0; i < previewSize; i++) {
			if (i == previewSize - previewSize % 4) {
				ImGui::Indent((ImGui::GetWindowSize().x - width * (previewSize % 4) / 4) / 2 - (ImGui::GetWindowSize().x - width) / 2);
			}
			ImGui::Image((ImTextureID)(uintptr_t)fboVector[i].getTexture().getTextureData().textureID, ImVec2(width / 4, height / 4));
			if (i == previewSize - previewSize % 4) {
				ImGui::Indent(-(ImGui::GetWindowSize().x - width * (previewSize % 4) / 4) / 2 + (ImGui::GetWindowSize().x - width) / 2);
			}
			if (i != 3 && i != 7 && i != 11 && i != 15 && i != previewSize - 1) {
				ImGui::SameLine();
			}
			if (ImGui::IsItemClicked()) {
				selectedImage = i;
				previousSelectedImage = selectedImage;
			}
			if (ImGui::IsItemHovered()) {
				selectedImage = i;
			}
		}
		ImGui::Indent(-(ImGui::GetWindowSize().x - width) / 2);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::TreePop();
	}
	if (ImGui::TreeNodeEx("Image", ImGuiStyleVar_WindowPadding | ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Indent((ImGui::GetWindowSize().x - width) / 2);
		ImGui::Image((ImTextureID)(uintptr_t)fboVector[selectedImage].getTexture().getTextureData().textureID, ImVec2(width, height));
		ImGui::Indent(-(ImGui::GetWindowSize().x - width) / 2);
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::Button("Save")) {
			fboVector[selectedImage].readToPixels(pixels);
			ofSaveImage(pixels, ofGetTimestampString("output/ofxStableDiffusion-%Y-%m-%d-%H-%M-%S.png"));
		}
		ImGui::TreePop();
	}
	ImGui::End();

	ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 10);
	ImGui::SetNextWindowSizeConstraints(ImVec2(532.f, -1.f), ImVec2(532.f, -1.f));
	ImGui::SetNextWindowPos(ImVec2(center.x * 1.4, center.y), ImGuiCond_Once, ImVec2(0.5f, 0.5f));
	ImGui::Begin("ofxStableDiffusion##foo2", &logOpenSettings, flags);
	if (!logOpenSettings) {
		ImGui::OpenPopup("Exit Program?");
	}
	ImGui::SetNextWindowPos(ofGetWindowSize() / 2, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
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
				modelName = result.getName();
				thread.stableDiffusion.load_from_file(result.getName(),"", GGML_TYPE_COUNT, DEFAULT);
			}
		}
		ImGui::SameLine(0, 5);
		ImGui::Text(&modelName[0]);
		if (!isTextToImage) {
			ImGui::BeginDisabled();
		}
		ImGui::Dummy(ImVec2(0, 10));
		static bool check = true;
		if (ImGui::Checkbox("TAESD", &check)) {
			if (check) {
				thread.stableDiffusion.setup(8, true, "data/models/taesd/diffusion_pytorch_model.SAFETENSORS", false, "data/models/lora/", STD_DEFAULT_RNG);
				thread.stableDiffusion.load_from_file("data/models/v1-5-pruned-emaonly.safetensors", "", GGML_TYPE_COUNT, DEFAULT);
			}
			else {
				thread.stableDiffusion.setup(8, false, "", false, "data/models/lora/", STD_DEFAULT_RNG);
				thread.stableDiffusion.load_from_file("data/models/v1-5-pruned-emaonly.safetensors", "", GGML_TYPE_COUNT, DEFAULT);
			}
		}
		if (!isTextToImage) {
			ImGui::EndDisabled();
		}
		ImGui::Dummy(ImVec2(0, 10));
		static int e = 0;
		if (ImGui::RadioButton("Text to Image", &e, 0)) {
			isTextToImage = true;
			batchSize = 4;
			sampleMethod = "DPMPP2S_Mv2";
		}
		ImGui::SameLine(0, 10);
		if (ImGui::RadioButton("Image to Image", &e, 1)) {
			isTextToImage = false;
			batchSize = 1;
			sampleMethod = "DPMPP2S_A";
			if (check) {
				thread.stableDiffusion.setup(8, false, "", false, "data/models/lora/", STD_DEFAULT_RNG);
				// thread.stableDiffusion.load_from_file("data/models/v1-5-pruned-emaonly.safetensors", "", GGML_TYPE_COUNT, DEFAULT);
				check = false;
			}
		}
		if (isTextToImage) {
			ImGui::BeginDisabled();
		}
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
				}
			}
		}
		ImGui::SameLine(0, 5);
		ImGui::Text(&imageName[0]);
		if (isTextToImage) {
			ImGui::EndDisabled();
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::Image((ImTextureID)(uintptr_t)fbo.getTexture().getTextureData().textureID, ImVec2(128, 128));
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderFloat("CFG Scale", &cfgScale, 0, 20);
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderInt("Sample Steps", &sampleSteps, 1, 50);
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::BeginCombo("Width", imageWidth)) {
			for (int n = 0; n < IM_ARRAYSIZE(imageSizeArray); n++) {
				bool is_selected = (imageWidth == imageSizeArray[n]);
				if (ImGui::Selectable(imageSizeArray[n], is_selected))
					imageWidth = imageSizeArray[n];
				if (is_selected)
					ImGui::SetItemDefaultFocus();
				if (width != atoi(imageWidth)) {
					width = atoi(imageWidth);
					ofFbo::Settings fboSettings;
					fboSettings.width = width;
					fboSettings.height = height;
					fboSettings.internalformat = GL_RGB;
					fboSettings.textureTarget = GL_TEXTURE_2D;
					for (int i = 0; i < 16; i++) {
						ofFbo fbo;
						fboVector.push_back(fbo);
						fboVector[i].allocate(fboSettings);
					}
					fbo.allocate(fboSettings);
					fbo.begin();
					image.draw(0, 0, width, height);
					fbo.end();
				}
			}
			ImGui::EndCombo();
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::BeginCombo("Height", imageHeight)) {
			for (int n = 0; n < IM_ARRAYSIZE(imageSizeArray); n++) {
				bool is_selected = (imageHeight == imageSizeArray[n]);
				if (ImGui::Selectable(imageSizeArray[n], is_selected))
					imageHeight = imageSizeArray[n];
				if (is_selected)
					ImGui::SetItemDefaultFocus();
				if (height != atoi(imageHeight)) {
					height = atoi(imageHeight);
					ofFbo::Settings fboSettings;
					fboSettings.width = width;
					fboSettings.height = height;
					fboSettings.internalformat = GL_RGB;
					fboSettings.textureTarget = GL_TEXTURE_2D;
					for (int i = 0; i < 16; i++) {
						ofFbo fbo;
						fboVector.push_back(fbo);
						fboVector[i].allocate(fboSettings);
					}
					fbo.allocate(fboSettings);
					fbo.begin();
					image.draw(0, 0, width, height);
					fbo.end();
				}
			}
			ImGui::EndCombo();
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (ImGui::BeginCombo("Sample Method", sampleMethod)) {
			for (int n = 0; n < IM_ARRAYSIZE(sampleMethodArray); n++)
			{
				bool is_selected = (sampleMethod == sampleMethodArray[n]);
				if (ImGui::Selectable(sampleMethodArray[n], is_selected)) {
					sampleMethod = sampleMethodArray[n];
					sampleMethodEnum = n;
				}
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
		if (isTextToImage) {
			ImGui::BeginDisabled();
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderFloat("Strength", &strength, 0, 1);
		if (isTextToImage) {
			ImGui::EndDisabled();
		}
		if (!isTextToImage) {
			ImGui::BeginDisabled();
		}
		ImGui::Dummy(ImVec2(0, 10));
		ImGui::SliderInt("Batch Size", &batchSize, 1, 16);
		if (!isTextToImage) {
			ImGui::EndDisabled();
		}
		ImGui::Dummy(ImVec2(0, 10));
		if (thread.isThreadRunning()) {
			ImGui::EndDisabled();
		}
		ImGui::TreePop();
	}
	if (thread.isThreadRunning()) {
		ImGui::BeginDisabled();
	}
	if (ImGui::Button("Generate")) {
		fbo.getTexture().readToPixels(pixels);
		thread.pixels = pixels.getData();
		thread.prompt = prompt;
		thread.negativePrompt = negativePrompt;
		thread.cfgScale = cfgScale;
		thread.width = width;
		thread.height = height;
		thread.sampleMethod = (SampleMethod)sampleMethodEnum;
		thread.sampleSteps = sampleSteps;
		thread.strength = strength;
		thread.seed = -1;
		thread.batch_count = batchSize;
		thread.isTextToImage = isTextToImage;
		thread.startThread();
	}
	if (thread.isThreadRunning()) {
		ImGui::EndDisabled();
	}
	ImGui::End();
	gui.end();
}

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
			while (tmpStr[lastSpace] != ' ' && lastSpace > 0)
				lastSpace--;
			if (lastSpace == 0)
				lastSpace = tmpStr.size() - 2;
			finalStr += tmpStr.substr(0, lastSpace + 1) + "\r\n";
			if (lastSpace + 1 > tmpStr.size())
				tmpStr = "";
			else
				tmpStr = tmpStr.substr(lastSpace + 1);
		}
		curChr++;
	}
	if (tmpStr.size() > 0)
		finalStr += tmpStr;
	str = finalStr;
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
