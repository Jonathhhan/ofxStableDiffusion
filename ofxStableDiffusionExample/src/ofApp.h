#pragma once

#include "ofMain.h"
#include "stableDiffusionThread.h"
#include "stable-diffusion.h"
#include "ofxImGui.h"

class ofApp : public ofBaseApp {
	public:
		void setup();
		void update();
		void draw();

		void keyPressed  (int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		void addSoftReturnsToText(std::string& str, float multilineWidth);
		void allocate();

		stableDiffusionThread thread;
		std::vector<ofTexture> textureVector;
		StableDiffusion stableDiffusion;
		std::vector<uint8_t*> stableDiffusionPixelVector;
		ofFbo fbo;
		ofImage image;
		ofPixels pixels;
		std::string prompt;
		std::string negativePrompt;
		int width;
		int height;
		float cfgScale;
		int batchSize;
		float strength;
		int seed;
		int clipSkipLayers;
		int previewSize;
		int selectedImage;
		int previousSelectedImage;
		const char* imageSizeArray[8] = {"128", "256", "384", "512", "640", "768", "896", "1024"};
		const char* imageWidth;
		const char* imageHeight;
		const char* sampleMethodArray[8] = {"EULER_A", "EULER", "HEUN", "DPM2", "DPMPP2S_A", "DPMPP2M", "DPMPP2Mv2", "LCM"};
		const char* sampleMethod;
		std::string modelPath;
		std::string modelName;
		std::string taesdPath;
		std::string loraModelDir;
		std::string vaePath;
		std::string esrganPath;
		SampleMethod sampleMethodEnum;
		int sampleSteps;
		bool promptIsEdited;
		bool isVaeDecodeOnly;
		bool isVaeTiling;
		bool isFreeParamsImmediatly;
		bool negativePromptIsEdited;
		bool isTextToImage;
		bool isFullScreen;
		bool isTAESD;
		bool isESRGAN;
		int numThreads;
		int esrganMultiplier;
		std::string imageName;
		ofxImGui::Gui gui;
		ggml_type ggmlType;
		Schedule schedule;
		RNGType rngType;
		bool diffused;
		bool isModelLoading;
};