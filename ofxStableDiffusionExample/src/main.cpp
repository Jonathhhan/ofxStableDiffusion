#include "ofMain.h"
#include "ofApp.h"

//========================================================================
int main() {

	ofGLFWWindowSettings settings;
	settings.setSize(1200, 850);
	settings.visible = true;
	settings.windowMode = OF_WINDOW;

	auto window = ofCreateWindow(settings);

	ofRunApp(window, make_shared<ofApp>());
	ofRunMainLoop();
	
}
