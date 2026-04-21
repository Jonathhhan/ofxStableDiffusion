#include "ofxStableDiffusionRealtimeSession.h"
#include <numeric>

ofxStableDiffusionRealtimeSession::ofxStableDiffusionRealtimeSession() {
}

ofxStableDiffusionRealtimeSession::~ofxStableDiffusionRealtimeSession() {
	stop();
}

bool ofxStableDiffusionRealtimeSession::start(const ofxStableDiffusionRealtimeSettings& settings_) {
	if (active) {
		ofLogWarning("ofxStableDiffusionRealtimeSession") << "Session already active";
		return false;
	}

	settings = settings_;
	active = true;
	stats = ofxStableDiffusionRealtimeStats();
	stats.sessionStartTime = ofGetElapsedTimeMillis();
	stats.isActive = true;

	// Initialize pending request with defaults
	pendingRequest.cfgScale = settings.cfgScale;
	pendingRequest.sampleSteps = settings.minSampleSteps;

	// Perform warmup if enabled
	if (settings.enableWarmup) {
		ofLogNotice("ofxStableDiffusionRealtimeSession") << "Starting warmup...";
		warmup();
	}

	ofLogNotice("ofxStableDiffusionRealtimeSession")
		<< "Real-time session started with target latency: " << settings.targetLatencyMs << "ms";

	return true;
}

void ofxStableDiffusionRealtimeSession::stop() {
	if (!active) {
		return;
	}

	active = false;
	stats.isActive = false;

	ofLogNotice("ofxStableDiffusionRealtimeSession")
		<< "Session stopped. Total generations: " << stats.totalGenerations
		<< ", Avg latency: " << stats.averageLatencyMs << "ms";
}

bool ofxStableDiffusionRealtimeSession::isActive() const {
	return active;
}

bool ofxStableDiffusionRealtimeSession::submit(const ofxStableDiffusionRealtimeRequest& request) {
	if (!active) {
		ofLogWarning("ofxStableDiffusionRealtimeSession") << "Session not active";
		return false;
	}

	std::lock_guard<std::mutex> lock(requestMutex);

	// Check queue depth and potentially drop frame
	// In a real implementation, this would check against the actual queue
	// For now, we just accept the request

	pendingRequest = request;

	// Placeholder: in real implementation, this would submit to generation queue
	// and the result would come back asynchronously

	return true;
}

void ofxStableDiffusionRealtimeSession::updatePrompt(const std::string& prompt) {
	std::lock_guard<std::mutex> lock(requestMutex);
	pendingRequest.prompt = prompt;
}

void ofxStableDiffusionRealtimeSession::updateCfgScale(float cfgScale) {
	std::lock_guard<std::mutex> lock(requestMutex);
	pendingRequest.cfgScale = cfgScale;
}

void ofxStableDiffusionRealtimeSession::updateStrength(float strength) {
	std::lock_guard<std::mutex> lock(requestMutex);
	pendingRequest.strength = strength;
}

void ofxStableDiffusionRealtimeSession::setResultCallback(ofxSdRealtimeResultCallback callback) {
	resultCallback = callback;
}

void ofxStableDiffusionRealtimeSession::setLatencyCallback(ofxSdRealtimeLatencyCallback callback) {
	latencyCallback = callback;
}

ofxStableDiffusionRealtimeStats ofxStableDiffusionRealtimeSession::getStats() const {
	return stats;
}

ofxStableDiffusionRealtimeSettings ofxStableDiffusionRealtimeSession::getSettings() const {
	return settings;
}

void ofxStableDiffusionRealtimeSession::resetStats() {
	stats = ofxStableDiffusionRealtimeStats();
	stats.sessionStartTime = ofGetElapsedTimeMillis();
	stats.isActive = active;
	latencyHistory.clear();
}

ofxStableDiffusionResult ofxStableDiffusionRealtimeSession::getLastResult() const {
	return lastResult;
}

bool ofxStableDiffusionRealtimeSession::warmup() {
	if (!active) {
		return false;
	}

	ofLogNotice("ofxStableDiffusionRealtimeSession") << "Performing warmup generation...";

	// Create a simple warmup request
	ofxStableDiffusionRealtimeRequest warmupRequest;
	warmupRequest.prompt = "warmup";
	warmupRequest.width = 512;
	warmupRequest.height = 512;
	warmupRequest.sampleSteps = settings.minSampleSteps;

	// In real implementation, this would actually generate an image to warm up the model
	// For now, just simulate the warmup

	ofLogNotice("ofxStableDiffusionRealtimeSession") << "Warmup complete";

	return true;
}

void ofxStableDiffusionRealtimeSession::processQueue() {
	// Placeholder for queue processing
	// In real implementation, this would be called from a thread to process pending requests
}

void ofxStableDiffusionRealtimeSession::updateStats(float latencyMs) {
	stats.totalGenerations++;
	latencyHistory.push_back(latencyMs);

	// Keep latency history reasonable size
	if (latencyHistory.size() > 100) {
		latencyHistory.erase(latencyHistory.begin());
	}

	// Update min/max
	if (stats.minLatencyMs == 0.0f || latencyMs < stats.minLatencyMs) {
		stats.minLatencyMs = latencyMs;
	}
	if (latencyMs > stats.maxLatencyMs) {
		stats.maxLatencyMs = latencyMs;
	}

	// Calculate average
	float sum = std::accumulate(latencyHistory.begin(), latencyHistory.end(), 0.0f);
	stats.averageLatencyMs = sum / latencyHistory.size();

	// Call latency callback if set
	if (latencyCallback) {
		latencyCallback(latencyMs);
	}

	// Check if we're dropping frames (latency exceeds target)
	if (latencyMs > settings.targetLatencyMs * 1.5f) {
		stats.droppedFrames++;
	}
}
