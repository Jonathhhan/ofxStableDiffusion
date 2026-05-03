#pragma once

#include "ofMain.h"
#include "ofxStableDiffusion.h"
#include <atomic>

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
    std::atomic<float> progress{0.0f};
};
