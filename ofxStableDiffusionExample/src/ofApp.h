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
		void applyRecommendedImageParameters();
		void applyRecommendedVideoParameters();
		void clampCurrentParametersToProfiles();
		void configureExampleRanker();
		bool usesInputImageMode() const;
		bool loadImageIntoSlot(
			const std::string& dialogTitle,
			ofImage& targetImage,
			ofPixels& targetPixels,
			sd_image_t& targetSdImage,
			std::string& targetName);
		void setupHoloscanBridge();
		void drawHoloscanBridgeSection();
		ofxStableDiffusionContextSettings buildContextSettings() const;
		void refreshModelContext();
		bool selectPath(
			const std::string& dialogTitle,
			std::string& targetPath,
			bool folderSelection = false);
		std::vector<std::pair<std::string, std::string>> listEmbeddingFiles() const;
		std::vector<std::pair<std::string, std::string>> listLoraFiles() const;
		void loadAllLoras(float strength = 1.0f);
		void clearLoras();

		std::vector<ofTexture> textureVector;
		ofFbo fbo;
		ofImage image;
		ofImage endFrameImage;
		ofImage maskGuideImage;
		ofImage controlGuideImage;
		ofPixels pixels;
		ofPixels endFramePixels;
		ofPixels maskGuidePixels;
		ofPixels controlGuidePixels;
		std::string prompt;
		std::string promptB;
		std::string instruction;
		std::string negativePrompt;
		std::string rankingPrompt;
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
		const char* imageModeArray[6] = {"TextToImage", "ImageToImage", "InstructImage", "Variation", "Restyle", "Inpainting"};
		const char* imageMode;
		const char* selectionModeArray[3] = {"KeepOrder", "Rerank", "BestOnly"};
		const char* selectionMode;
		const char* sampleMethodArray[8] = {"EULER_A_SAMPLE_METHOD", "EULER_SAMPLE_METHOD", "HEUN_SAMPLE_METHOD", "DPM2_SAMPLE_METHOD", "DPMPP2S_A_SAMPLE_METHOD", "DPMPP2M_SAMPLE_METHOD", "DPMPP2Mv2_SAMPLE_METHOD", "LCM_SAMPLE_METHOD"};
		const char* sampleMethod;
		const char* videoModeArray[4] = {"Standard", "Loop", "PingPong", "Boomerang"};
		const char* videoMode;
		const char* interpolationModeArray[5] = {"Linear", "Smooth", "EaseIn", "EaseOut", "EaseInOut"};
		const char* interpolationMode;
		std::string modelPath;
		std::string modelName;
		std::string diffusionModelPath;
		std::string clipLPath;
		std::string clipGPath;
		std::string t5xxlPath;
		std::string taesdPath;
		std::string controlNetPath;
		std::string embedDir;
		std::string loraModelDir;
		std::string vaePath;
		std::string esrganPath;
		std::string stackedIdEmbedDir;
		std::string inputIdImagesPath;
		std::vector<ofxStableDiffusionLora> loras;
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
		bool offloadParamsToCpu;
		bool keepClipOnCpu;
		bool keepControlNetCpu;
		bool keepVaeOnCpu;
		float styleStrength;
		bool normalizeInput;
		int nThreads;
		int esrganMultiplier;
		sd_type_t wType;
		scheduler_t schedule;
		rng_type_t rngType;
		std::string controlImagePath;
		float controlStrength;

		bool diffused;
		bool isModelLoading;
		sd_image_t inputImage = {0, 0, 0, nullptr};
		sd_image_t* outputImages = nullptr;
		sd_image_t controlGuideInput = {0, 0, 0, nullptr};
		ofxStableDiffusionImageMode imageModeEnum = ofxStableDiffusionImageMode::TextToImage;
		ofxStableDiffusionImageSelectionMode selectionModeEnum =
			ofxStableDiffusionImageSelectionMode::KeepOrder;
		bool isTextToImage;
		bool isInstructImage;
		bool isImageToVideo;
		int videoFrames;
		int videoFps;
		float vaceStrength;
		bool isPlaying;
		int currentFrame;
		int totalVideoFrames;
		float lastFrameTime;
		int previewWidth;
		int maxPreviewTextures = 64;
		int progressStep = 0;
		int progressSteps = 0;
		float progressTime = 0.0f;
		bool useDemoRanking = false;
		std::string imageName;
		std::string endImageName;
		std::string maskImageName;
		std::string controlGuideName;
		ofxImGui::Gui gui;
		ofxStableDiffusion stableDiffusion;
		bool enablePromptInterpolation = false;
		bool useSeedSequence = false;
		bool useEndFrame = false;
		bool useMaskGuide = false;
		bool useControlGuide = false;
		int seedIncrement = 1;
		ofxStableDiffusionInterpolationMode interpolationModeEnum =
			ofxStableDiffusionInterpolationMode::Smooth;
		ofxStableDiffusionImageParameterProfile imageParameterProfile;
		ofxStableDiffusionVideoParameterProfile videoParameterProfile;
		sd_image_t endInputImage = {0, 0, 0, nullptr};
		sd_image_t maskGuideInput = {0, 0, 0, nullptr};
		ofxStableDiffusionHoloscanBridge holoscanBridge;
		bool holoscanBridgeEnabled = false;
		bool holoscanBridgeRunning = false;
		bool holoscanBridgeUseCurrentPrompts = true;
		std::string holoscanPrompt;
		std::string holoscanNegativePrompt;
		std::string holoscanStatus;
		int holoscanCompletedFrames = 0;
};
