#pragma once

#include "ofMain.h"
#include "ofxImGui.h"
#include "ofxStableDiffusion.h"

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
		void applySelectedImageMode(ofxStableDiffusionImageMode mode);
		bool usesInputImageMode() const;

		std::vector<ofTexture> textureVector;
		ofFbo fbo;
		ofImage image;
		ofPixels pixels;
		std::string prompt;
		std::string negativePrompt;
		int width;
		int height;
		float cfgScale;
		int batchCount;
		float strength;
		int seed;
		int clipSkip;
		int previewSize;
		int selectedImage;
		int previousSelectedImage;
		const char* imageSizeArray[8] = {"128", "256", "384", "512", "640", "768", "896", "1024"};
		const char* imageWidth;
		const char* imageHeight;
		const char* imageModeArray[5] = {"TextToImage", "ImageToImage", "InstructImage", "Variation", "Restyle"};
		const char* imageMode;
		const char* selectionModeArray[3] = {"KeepOrder", "Rerank", "BestOnly"};
		const char* selectionMode;
		const char* sampleMethodArray[8] = {"EULER_A", "EULER", "HEUN", "DPM2", "DPMPP2S_A", "DPMPP2M", "DPMPP2Mv2", "LCM"};
		const char* sampleMethod;
		const char* videoModeArray[4] = {"Standard", "Loop", "PingPong", "Boomerang"};
		const char* videoMode;
		std::string modelPath;
		std::string modelName;
		std::string taesdPath;
		std::string controlNetPath;
		std::string embedDir;
		std::string loraModelDir;
		std::string vaePath;
		std::string esrganPath;
		std::string stackedIdEmbedDir;
		std::string inputIdImagesPath;
		sample_method_t sampleMethodEnum;
		int sampleSteps;
		bool promptIsEdited;
		bool vaeDecodeOnly;
		bool vaeTiling;
		bool freeParamsImmediately;
		bool negativePromptIsEdited;
		bool isFullScreen;
		bool isTAESD;
		bool isESRGAN;
		bool keepClipOnCpu;
		bool keepControlNetCpu;
		bool keepVaeOnCpu;
		float styleStrength;
		bool normalizeInput;
		int nThreads;
		int esrganMultiplier;
		sd_type_t wType;
		schedule_t schedule;
		rng_type_t rngType;
		std::string controlImagePath;
		float controlStrength;

		bool diffused;
		bool isModelLoading;
		sd_image_t inputImage = {0, 0, 0, nullptr};
		sd_image_t* outputImages = nullptr;
		sd_image_t* controlImage = nullptr;
		ofxStableDiffusionImageMode imageModeEnum = ofxStableDiffusionImageMode::TextToImage;
		ofxStableDiffusionImageSelectionMode selectionModeEnum =
			ofxStableDiffusionImageSelectionMode::KeepOrder;
		bool isTextToImage;
		bool isInstructImage;
		bool isImageToVideo;
		int videoFrames;
		int motionBucketId;
		int videoFps;
		float augmentationLevel;
		float minCfg;
		bool isPlaying;
		int currentFrame;
		int totalVideoFrames;
		float lastFrameTime;
		int previewWidth;
		int maxPreviewTextures = 64;
		int progressStep = 0;
		int progressSteps = 0;
		float progressTime = 0.0f;
		std::string imageName;
		ofxImGui::Gui gui;
		ofxStableDiffusion stableDiffusion;
};
