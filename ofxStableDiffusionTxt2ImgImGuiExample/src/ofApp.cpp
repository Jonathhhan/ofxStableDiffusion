#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxStableDiffusionTxt2ImgExample");
	printf("%s", sd_get_system_info().c_str());
	set_sd_log_level(DEBUG);
	thread.stableDiffusion.setup(8, true, "data/models/taesd/taesd-model.gguf", false, "data/models/lora/", STD_DEFAULT_RNG);
	thread.stableDiffusion.load_from_file("data/models/v1-5-pruned-emaonly-f16.gguf");
	gui.setup(nullptr, true, ImGuiConfigFlags_None, true);
	prompt = "<lora:ohara_koson:1>mushroom, ohara koson, traditional media, botanic painting";
	width = 512;
	height = 512;
	cfgScale = 7.0;
	sampleSteps = 10;
	currentImageWidth = "512";
	currentImageHeight = "512";
	currentSampleMethod = "DPMPP2Mv2";
	promptIsEdited = true;
	negativePromptIsEdited = true;
	ofFbo::Settings fboSettings;
	fboSettings.width = width;
	fboSettings.height = height;
	fboSettings.internalformat = GL_RGB;
	fboSettings.textureTarget = GL_TEXTURE_2D;
	fbo.allocate(fboSettings);
}

//--------------------------------------------------------------
void ofApp::update() {
	if (thread.diffused) {
		fbo.getTexture().loadData(&thread.stableDiffusionPixelVector[0][0], width, height, GL_RGB);
		fbo.readToPixels(pixels);
		ofSaveImage(pixels, ofGetTimestampString("output/ofxStableDiffusion-%Y-%m-%d-%H-%M-%S.png"));
		thread.diffused = false;
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	ImGuiWindowFlags flags = ImGuiWindowFlags_AlwaysAutoResize;
	static bool log_open{ true };
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
	ImGui::StyleColorsDark();
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
	ImVec2 center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(ImVec2(center.x / 1.5, center.y), ImGuiCond_Once, ImVec2(0.5f, 0.5f));
	ImGui::Begin("Stable Diffusion", &log_open, flags | ImGuiWindowFlags_NoCollapse);
	ImGui::Image((ImTextureID)(uintptr_t)fbo.getTexture().getTextureData().textureID, ImVec2(width, height));
	ImGui::End();
	ImGui::SetNextWindowSizeConstraints(ImVec2(532.f, -1.f), ImVec2(INFINITY, -1.f));
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
	ImGui::PushStyleVar(ImGuiStyleVar_IndentSpacing, 10);
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 10));
	ImGui::SetNextWindowPos(ImVec2(center.x*1.25, center.y), ImGuiCond_Once, ImVec2(0.5f, 0.5f));
	ImGui::Begin("Control", &log_open, flags);
	if (promptIsEdited) {
			addSoftReturnsToText(prompt, 500);
			promptIsEdited = false;
		}
	if (ImGui::TreeNodeEx("Prompt", ImGuiStyleVar_WindowPadding | ImGuiTreeNodeFlags_DefaultOpen)) {
		Funcs::MyInputTextMultiline("##MyStr1", &prompt, ImVec2(512, 150), ImGuiInputTextFlags_CallbackResize);
		ImGui::TreePop();
	}	
	if (ImGui::IsItemDeactivatedAfterEdit()) {
		prompt.erase(std::remove(prompt.begin(), prompt.end(), '\r'), prompt.end());
		prompt.erase(std::remove(prompt.begin(), prompt.end(), '\n'), prompt.end());
		promptIsEdited = true;
	}
	if (negativePromptIsEdited) {
		addSoftReturnsToText(negativePrompt, 500);
		negativePromptIsEdited = false;
	}
	if (ImGui::TreeNodeEx("Negative Prompt", ImGuiStyleVar_WindowPadding)) {
		Funcs::MyInputTextMultiline("##MyStr2", &negativePrompt, ImVec2(512, 150), ImGuiInputTextFlags_CallbackResize);
		ImGui::TreePop();
	}
	if (ImGui::IsItemDeactivatedAfterEdit()) {
		negativePrompt.erase(std::remove(negativePrompt.begin(), negativePrompt.end(), '\r'), negativePrompt.end());
		negativePrompt.erase(std::remove(negativePrompt.begin(), negativePrompt.end(), '\n'), negativePrompt.end());
		negativePromptIsEdited = true;
	}
	if (ImGui::TreeNodeEx("Parameters", ImGuiStyleVar_WindowPadding | ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat("CFG Scale", &cfgScale, 0, 20);
		ImGui::SliderInt("Sample Steps", &sampleSteps, 1, 50);
		if (ImGui::BeginCombo("Width", currentImageWidth)) {
			for (int n = 0; n < IM_ARRAYSIZE(imageSizeArray); n++)
			{
				bool is_selected = (currentImageWidth == imageSizeArray[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(imageSizeArray[n], is_selected))
					currentImageWidth = imageSizeArray[n];
					if (is_selected)
						ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
					if (width != atoi(currentImageWidth)) {
						width = atoi(currentImageWidth);
						ofFbo::Settings fboSettings;
						fboSettings.width = width;
						fboSettings.height = height;
						fboSettings.internalformat = GL_RGB;
						fboSettings.textureTarget = GL_TEXTURE_2D;
						fbo.allocate(fboSettings);
					}
			}
			ImGui::EndCombo();
		}
		if (ImGui::BeginCombo("Height", currentImageHeight)) {
			for (int n = 0; n < IM_ARRAYSIZE(imageSizeArray); n++)
			{
				bool is_selected = (currentImageHeight == imageSizeArray[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(imageSizeArray[n], is_selected))
					currentImageHeight = imageSizeArray[n];
				if (is_selected)
					ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
				if (height != atoi(currentImageHeight)) {
					height = atoi(currentImageHeight);
					ofFbo::Settings fboSettings;
					fboSettings.width = width;
					fboSettings.height = height;
					fboSettings.internalformat = GL_RGB;
					fboSettings.textureTarget = GL_TEXTURE_2D;
					fbo.allocate(fboSettings);
				}
			}
			ImGui::EndCombo();
		}
		if (ImGui::BeginCombo("Sample Method", currentSampleMethod)) {
			for (int n = 0; n < IM_ARRAYSIZE(sampleMethodArray); n++)
			{
				bool is_selected = (currentSampleMethod == sampleMethodArray[n]); // You can store your selection however you want, outside or inside your objects
				if (ImGui::Selectable(sampleMethodArray[n], is_selected)) {
					currentSampleMethod = sampleMethodArray[n];
					currentSampleMethodEnum = n;
				}
				if (is_selected)
					ImGui::SetItemDefaultFocus();   // You may set the initial focus when opening the combo (scrolling + for keyboard navigation support)
			}
			ImGui::EndCombo();
		}
		ImGui::TreePop();
	}
	if (ImGui::Button("Generate")) {
		if (!thread.isThreadRunning()) {
			thread.prompt = prompt;
			thread.negativePrompt = negativePrompt;
			thread.cfgScale = cfgScale;
			std::cout << currentImageWidth << std::endl;
			thread.width = atoi(currentImageWidth);
			thread.height = atoi(currentImageHeight);
			thread.sampleMethod = (SampleMethod)currentSampleMethodEnum;
			thread.sampleSteps = sampleSteps;
			thread.seed = -1;
			thread.batch_count = 1;
			thread.startThread();
		}
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
