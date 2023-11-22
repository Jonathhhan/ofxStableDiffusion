#pragma once

#include "ofMain.h"
#include "stableDiffusionThread.h"
#include "stable-diffusion.h"

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

		stableDiffusionThread thread;
		ofImage image;
		ofTexture texture;
		ofPixels pixels;
		ofFbo fbo;
		std::string prompt;
		int width;
		int height;
};