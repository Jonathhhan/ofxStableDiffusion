#pragma once

#include "ofMain.h"
#include "ofxStableDiffusion.h"

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);

private:
    ofxStableDiffusion sd;
    ofImage resultImage;
    bool generating;
    float progress;
};
