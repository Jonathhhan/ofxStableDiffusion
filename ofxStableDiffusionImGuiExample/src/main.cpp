#include "ofMain.h"
#include "ofApp.h"

//========================================================================
int main( ){

	ofGLFWWindowSettings settings;
	settings.setSize(1280, 720);
	// settings.setPosition(glm::vec2(0,0));
	// settings.maximized = true;
	settings.visible = true;
	settings.monitor = 0;
	settings.multiMonitorFullScreen = false;
	settings.windowMode = OF_WINDOW;

	auto window = ofCreateWindow(settings);

	ofRunApp(window, make_shared<ofApp>());
	ofRunMainLoop();
	
}
