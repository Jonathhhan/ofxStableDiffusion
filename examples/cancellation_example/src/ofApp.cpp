#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
    generating = false;
    wasCancelled = false;
    progress = 0.0f;
    statusMessage = "Ready";

    // Configure SD context
    ofxStableDiffusionContextSettings settings;
    settings.modelPath = ofToDataPath("models/sd_v1.5.safetensors");
    settings.wType = SD_TYPE_F16;
    settings.nThreads = -1;

    sd.configureContext(settings);

    // Progress callback
    sd.setProgressCallback([this](int step, int steps, float time) {
        progress = (float)step / (float)steps;
    });

    ofLogNotice() << "Press SPACE to generate (long operation)";
    ofLogNotice() << "Press 'C' to cancel during generation";
}

//--------------------------------------------------------------
void ofApp::update() {
    bool wasGenerating = generating;
    generating = sd.isGenerating();

    // Check if generation just completed
    if (wasGenerating && !generating) {
        if (sd.wasCancelled()) {
            wasCancelled = true;
            statusMessage = "Generation cancelled by user";
            ofLogNotice() << statusMessage;
        } else if (sd.hasImageResult()) {
            auto images = sd.getImages();
            if (!images.empty()) {
                resultImage.setFromPixels(images[0].pixels);
                wasCancelled = false;
                statusMessage = "Generation complete!";
                ofLogNotice() << "Completed. Seed: " << sd.getLastUsedSeed();
            }
        } else {
            // Check for errors
            auto error = sd.getLastErrorInfo();
            if (error.code == ofxStableDiffusionErrorCode::Cancelled) {
                wasCancelled = true;
                statusMessage = "Cancelled: " + error.message;
            } else if (error.code != ofxStableDiffusionErrorCode::None) {
                statusMessage = "Error: " + error.message;
                ofLogError() << statusMessage;
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::draw() {
    ofBackground(30);

    // Draw result if available
    if (resultImage.isAllocated() && !wasCancelled) {
        float scale = std::min(
            ofGetWidth() / resultImage.getWidth(),
            ofGetHeight() / resultImage.getHeight()
        ) * 0.8f;

        float w = resultImage.getWidth() * scale;
        float h = resultImage.getHeight() * scale;
        float x = (ofGetWidth() - w) * 0.5f;
        float y = (ofGetHeight() - h) * 0.5f;

        resultImage.draw(x, y, w, h);
    }

    // Status UI
    ofSetColor(255);
    std::stringstream ss;
    ss << "Status: " << statusMessage << "\n\n";

    if (generating) {
        ss << "Progress: " << ofToString(progress * 100.0f, 1) << "%\n";
        ss << "Press 'C' to CANCEL\n\n";

        // Show cancellation status
        if (sd.isCancellationRequested()) {
            ofSetColor(255, 200, 0);
            ss << "CANCELLATION REQUESTED\n";
            ss << "Waiting for current step to complete...\n";
        }
    } else {
        ss << "Press SPACE to start generation\n";
        ss << "(This will take 50 steps - long operation)\n\n";

        if (wasCancelled) {
            ofSetColor(255, 150, 0);
            ss << "Last operation was CANCELLED\n";
        }
    }

    ofDrawBitmapString(ss.str(), 20, 20);
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
    if (key == ' ' && !generating) {
        startGeneration();
    }

    if (key == 'c' || key == 'C') {
        cancelGeneration();
    }
}

//--------------------------------------------------------------
void ofApp::startGeneration() {
    // Create a long-running request (50 steps)
    ofxStableDiffusionImageRequest request;
    request.prompt = "A detailed fantasy castle on a mountain, highly detailed, epic";
    request.negativePrompt = "blurry, low quality";
    request.width = 768;   // Larger size
    request.height = 768;  // = longer generation
    request.sampleSteps = 50;  // More steps = longer time
    request.cfgScale = 8.0f;
    request.seed = -1;
    request.batchCount = 1;

    wasCancelled = false;
    statusMessage = "Generating...";
    sd.generate(request);

    ofLogNotice() << "Started long generation (50 steps at 768x768)";
    ofLogNotice() << "This will take a while - press 'C' to cancel";
}

//--------------------------------------------------------------
void ofApp::cancelGeneration() {
    if (sd.isGenerating()) {
        bool requested = sd.requestCancellation();
        if (requested) {
            statusMessage = "Cancelling... please wait";
            ofLogNotice() << "Cancellation requested";
            ofLogNotice() << "Generation will stop after current step";
        }
    } else {
        ofLogNotice() << "Nothing to cancel - not generating";
    }
}
