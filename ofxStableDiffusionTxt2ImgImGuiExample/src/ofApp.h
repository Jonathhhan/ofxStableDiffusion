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
		stableDiffusionThread thread;
		std::vector<ofFbo> fboVector;
		ofPixels pixels;
		std::string prompt;
		std::string negativePrompt;
		int width;
		int height;
		float cfgScale;
		int batchSize;
		int previewSize;
		int selectedImage;
		int previousSelectedImage;
		const char* imageSizeArray[6] = {"128", "256", "384", "512", "768", "1024"};
		const char* currentImageWidth;
		const char* currentImageHeight;
		const char* sampleMethodArray[8] = {"EULER_A", "EULER", "HEUN", "DPM2", "DPMPP2S_A", "DPMPP2M", "DPMPP2Mv2", "LCM"};
		const char* currentSampleMethod;
		int currentSampleMethodEnum;
		int sampleSteps;
		bool promptIsEdited;
		bool negativePromptIsEdited;
		ofxImGui::Gui gui;
};