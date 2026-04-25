#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    generating = false;
    progress = 0.0f;

    // Configure the SD context with a model
    ofxStableDiffusionContextSettings settings;
    settings.modelPath = ofToDataPath("models/sd_v1.5.safetensors");
    settings.wType = SD_TYPE_F16;  // Half precision for faster performance
    settings.nThreads = -1;        // Auto-detect optimal thread count

    sd.configureContext(settings);

    // Set up progress callback
    sd.setProgressCallback([this](int step, int steps, float time) {
        progress = (float)step / (float)steps;
        ofLogNotice() << "Progress: " << (progress * 100.0f) << "%";
    });

    ofLogNotice() << "Ready! Press SPACE to generate an image.";
}

//--------------------------------------------------------------
void ofApp::update() {
    generating = sd.isGenerating();

    // Check if generation just completed
    if (!generating && sd.hasImageResult()) {
        // Get the first generated image
        auto images = sd.getImages();
        if (!images.empty()) {
            resultImage.setFromPixels(images[0].pixels);
            ofLogNotice() << "Generation complete! Seed: " << sd.getLastUsedSeed();
        }

        // Check for errors
        auto error = sd.getLastErrorInfo();
        if (error.code != ofxStableDiffusionErrorCode::None) {
            ofLogError() << "Error: " << error.message;
            ofLogNotice() << "Suggestion: " << error.suggestion;
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw() {
    ofBackground(30);

    // Draw the result image if available
    if (resultImage.isAllocated()) {
        float scale = std::min(
            ofGetWidth() / resultImage.getWidth(),
            ofGetHeight() / resultImage.getHeight()
        ) * 0.9f;

        float w = resultImage.getWidth() * scale;
        float h = resultImage.getHeight() * scale;
        float x = (ofGetWidth() - w) * 0.5f;
        float y = (ofGetHeight() - h) * 0.5f;

        resultImage.draw(x, y, w, h);
    }

    // Draw status
    ofSetColor(255);
    if (generating) {
        ofDrawBitmapString("Generating... " + ofToString(progress * 100.0f, 1) + "%", 20, 20);
    } else {
        ofDrawBitmapString("Press SPACE to generate\nPress 'S' to save", 20, 20);
    }
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
    if (key == ' ' && !generating) {
        // Create a generation request
        ofxStableDiffusionImageRequest request;
        request.prompt = "A serene mountain landscape at sunset, photorealistic";
        request.negativePrompt = "blurry, low quality, distorted";
        request.width = 512;
        request.height = 512;
        request.sampleSteps = 20;
        request.cfgScale = 7.0f;
        request.seed = -1;  // Random seed
        request.batchCount = 1;

        // Start generation
        sd.generate(request);
        ofLogNotice() << "Starting generation...";
    }

    if (key == 's' && resultImage.isAllocated()) {
        std::string filename = "output_" + ofGetTimestampString() + ".png";
        resultImage.save(filename);
        ofLogNotice() << "Saved: " << filename;
    }
}
