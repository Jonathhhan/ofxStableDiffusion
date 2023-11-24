#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetWindowTitle("ofxStableDiffusionTxt2ImgExample");
	printf("%s", sd_get_system_info().c_str());
	set_sd_log_level(INFO);
	thread.stableDiffusion.load_from_file("data/models/v1-5-pruned-emaonly-f16.gguf");
	width = 512;
	height = 512;
	texture.allocate(width, height, GL_RGB);
	texture.setTextureMinMagFilter(GL_NEAREST, GL_NEAREST);
	prompt = "a lovely cat, van gogh";
}

//--------------------------------------------------------------
void ofApp::update() {
	if (thread.diffused) {
		texture.loadData(&thread.stableDiffusionPixelVector[0][0], width, height, GL_RGB);
		texture.readToPixels(pixels);
		ofSaveImage(pixels, "output/image_of_" + thread.prompt + "_" + ofGetTimestampString("%Y-%m-%d-%H-%M-%S") + ".png");
		thread.diffused = false;	
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	ofSetColor(255);
	texture.draw(20, 20, 512, 512);
	ofDrawBitmapString("Type something and press enter to generate an image.", 40, 550);
	ofDrawBitmapString("Prompt: " + prompt, 40, 570);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	if (key == OF_KEY_RETURN) {
		if (!thread.isThreadRunning()) {
			thread.prompt = prompt;
			thread.negativePrompt = "";
			thread.cfgScale = 7.0;
			thread.width = width;
			thread.height = height;
			thread.sampleMethod = DPMPP2Mv2;
			thread.sampleSteps = 10;
			thread.seed = -1;
			thread.batch_count = 1;
			thread.startThread();
		}
	} else if (key == 8 && prompt.size() > 0) {
		prompt = prompt.substr(0, prompt.size() - 1);
	} else if (key == 127) {
		prompt  = "";
	} else if (prompt.size() < 50) {
		prompt.append(1, (char)key);
	}
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {
	
}

//------------- -------------------------------------------------
void ofApp::mouseMoved(int x, int y ) {

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
