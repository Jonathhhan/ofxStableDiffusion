#include "ofxStableDiffusionRealtimeSession.h"
#include "../ofxStableDiffusion.h"
#include <algorithm>
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
	// Ensure maxSampleSteps is never below minSampleSteps
	settings.maxSampleSteps = std::max(settings.maxSampleSteps, settings.minSampleSteps);
	active = true;
	generationInFlight = false;
	hasPendingRequest = false;
	warmingUp = false;
	currentSampleSteps = settings.minSampleSteps;
	stats = ofxStableDiffusionRealtimeStats();
	stats.sessionStartTime = ofGetElapsedTimeMillis();
	stats.isActive = true;
	latencyHistory.clear();

	// Initialize pending request defaults
	pendingRequest.cfgScale = settings.cfgScale;
	pendingRequest.sampleSteps = settings.minSampleSteps;

	// Warmup if a generator is attached and warmup is requested
	if (settings.enableWarmup && generator) {
		warmup();
	}

	ofLogNotice("ofxStableDiffusionRealtimeSession")
		<< "Real-time session started with target latency: " << settings.targetLatencyMs << "ms";

	return true;
}

bool ofxStableDiffusionRealtimeSession::start(const ofxStableDiffusionRealtimeSettings& settings_,
	ofxStableDiffusion& sd) {
	setGenerator(&sd);
	return start(settings_);
}

void ofxStableDiffusionRealtimeSession::setGenerator(ofxStableDiffusion* sd) {
	generator = sd;
}

void ofxStableDiffusionRealtimeSession::stop() {
	if (!active) {
		return;
	}

	active = false;
	stats.isActive = false;
	generationInFlight = false;
	hasPendingRequest = false;
	warmingUp = false;
	currentSampleSteps = settings.minSampleSteps;

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

	if (!generator) {
		ofLogWarning("ofxStableDiffusionRealtimeSession")
			<< "submit: no generator attached; call setGenerator() or start(settings, sd) first";
		return false;
	}

	// Drop if busy and no buffering is allowed.
	// LowLatency mode always drops rather than queuing, regardless of maxQueueDepth.
	const bool mustDropIfBusy =
		(settings.maxQueueDepth == 0 ||
		settings.mode == ofxStableDiffusionRealtimeMode::LowLatency);
	if (generationInFlight && mustDropIfBusy) {
		stats.droppedFrames++;
		return false;
	}

	{
		std::lock_guard<std::mutex> lock(requestMutex);
		pendingRequest = request;
		hasPendingRequest = true;
	}

	// Fire immediately if the generator is idle
	if (!generationInFlight) {
		processQueue();
	}

	return true;
}

void ofxStableDiffusionRealtimeSession::update() {
	if (!active || !generator) {
		return;
	}

	if (!generationInFlight) {
		// Nothing running — fire a pending request if one is waiting
		processQueue();
		return;
	}

	// A generation is in flight; check whether it has finished
	if (generator->isGenerating()) {
		return;
	}

	// Generation just completed
	generationInFlight = false;
	const float latencyMs =
		static_cast<float>(ofGetElapsedTimeMicros() - generationStartMicros) / 1000.0f;

	if (warmingUp) {
		warmingUp = false;
		ofLogNotice("ofxStableDiffusionRealtimeSession")
			<< "Warmup complete (latency: " << latencyMs << "ms)";
	} else {
		const ofxStableDiffusionResult result = generator->getLastResult();
		{
			std::lock_guard<std::mutex> lock(requestMutex);
			lastResult = result;
		}
		updateStats(latencyMs);
		if (resultCallback && result.success) {
			try {
				resultCallback(result);
			} catch (const std::exception& e) {
				ofLogWarning("ofxStableDiffusionRealtimeSession") << "Result callback threw: " << e.what();
			} catch (...) {
				ofLogWarning("ofxStableDiffusionRealtimeSession") << "Result callback threw an unknown exception";
			}
		}
	}

	// Immediately fire any queued request so latency stays minimal
	processQueue();
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

void ofxStableDiffusionRealtimeSession::updateNegativePrompt(const std::string& negativePrompt) {
	std::lock_guard<std::mutex> lock(requestMutex);
	pendingRequest.negativePrompt = negativePrompt;
}

void ofxStableDiffusionRealtimeSession::updateSeed(int seed) {
	std::lock_guard<std::mutex> lock(requestMutex);
	pendingRequest.seed = seed;
}

void ofxStableDiffusionRealtimeSession::updateSampleSteps(int steps) {
	std::lock_guard<std::mutex> lock(requestMutex);
	pendingRequest.sampleSteps = steps;
	// Also reset the adaptive step counter so the explicit value takes effect immediately
	currentSampleSteps = clampToStepRange(steps);
}

void ofxStableDiffusionRealtimeSession::updateDimensions(int width, int height) {
	std::lock_guard<std::mutex> lock(requestMutex);
	pendingRequest.width = width;
	pendingRequest.height = height;
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
	std::lock_guard<std::mutex> lock(requestMutex);
	return lastResult;
}

bool ofxStableDiffusionRealtimeSession::isGenerating() const {
	return generationInFlight;
}

int ofxStableDiffusionRealtimeSession::getCurrentSampleSteps() const {
	return currentSampleSteps;
}

bool ofxStableDiffusionRealtimeSession::warmup() {
	if (!active) {
		ofLogWarning("ofxStableDiffusionRealtimeSession") << "warmup: session not active";
		return false;
	}

	if (!generator) {
		ofLogWarning("ofxStableDiffusionRealtimeSession")
			<< "warmup: no generator attached; call setGenerator() or start(settings, sd) first";
		return false;
	}

	if (generationInFlight || generator->isGenerating()) {
		ofLogWarning("ofxStableDiffusionRealtimeSession")
			<< "warmup: cannot warm up while generation is in progress";
		return false;
	}

	ofxStableDiffusionImageRequest warmupReq;
	warmupReq.prompt = "warmup";
	warmupReq.width = 512;
	warmupReq.height = 512;
	warmupReq.sampleSteps = settings.minSampleSteps;
	warmupReq.cfgScale = settings.cfgScale;
	warmupReq.seed = 42;
	warmupReq.batchCount = 1;
	warmupReq.mode = ofxStableDiffusionImageMode::TextToImage;

	generator->generate(warmupReq);
	generationInFlight = true;
	warmingUp = true;
	generationStartMicros = ofGetElapsedTimeMicros();

	ofLogNotice("ofxStableDiffusionRealtimeSession") << "Warming up...";
	return true;
}

int ofxStableDiffusionRealtimeSession::clampToStepRange(int steps) const {
	return std::max(settings.minSampleSteps, std::min(steps, settings.maxSampleSteps));
}

void ofxStableDiffusionRealtimeSession::processQueue() {
	if (!generator || generator->isGenerating() || generationInFlight) {
		return;
	}

	ofxStableDiffusionRealtimeRequest req;
	{
		std::lock_guard<std::mutex> lock(requestMutex);
		if (!hasPendingRequest) {
			return;
		}
		req = pendingRequest;
		hasPendingRequest = false;
	}

	ofxStableDiffusionImageRequest imageReq;
	imageReq.prompt = req.prompt;
	imageReq.negativePrompt = req.negativePrompt;
	imageReq.cfgScale = req.cfgScale;
	imageReq.strength = req.strength;
	imageReq.seed = static_cast<int64_t>(req.seed);
	imageReq.width = req.width;
	imageReq.height = req.height;
	imageReq.sampleMethod = req.sampleMethod;
	// Use the session-managed adaptive step count when progressive refinement is on;
	// otherwise honour the value the caller placed on the request.
	imageReq.sampleSteps = settings.enableProgressiveRefinement ? currentSampleSteps : req.sampleSteps;
	imageReq.batchCount = 1;
	imageReq.mode = ofxStableDiffusionImageMode::TextToImage;

	generator->generate(imageReq);
	generationInFlight = true;
	generationStartMicros = ofGetElapsedTimeMicros();
}

void ofxStableDiffusionRealtimeSession::updateStats(float latencyMs) {
	stats.totalGenerations++;
	latencyHistory.push_back(latencyMs);

	if (latencyHistory.size() > 100) {
		latencyHistory.pop_front();
	}

	// Update min/max
	if (stats.minLatencyMs == 0.0f || latencyMs < stats.minLatencyMs) {
		stats.minLatencyMs = latencyMs;
	}
	if (latencyMs > stats.maxLatencyMs) {
		stats.maxLatencyMs = latencyMs;
	}

	// Calculate rolling average
	const float sum = std::accumulate(latencyHistory.begin(), latencyHistory.end(), 0.0f);
	stats.averageLatencyMs = sum / static_cast<float>(latencyHistory.size());

	if (latencyCallback) {
		latencyCallback(latencyMs);
	}

	// Track frames that significantly exceeded the latency target (distinct from dropped frames)
	const float target = static_cast<float>(settings.targetLatencyMs);
	if (latencyMs > target * 1.5f) {
		stats.slowFrames++;
	}

	// Adaptive step count: ramp quality up when fast, down when slow
	if (settings.enableProgressiveRefinement) {
		if (latencyMs < target * 0.8f) {
			// Plenty of headroom — raise quality
			currentSampleSteps = clampToStepRange(currentSampleSteps + 1);
		} else if (latencyMs > target) {
			// Over budget — reduce quality
			currentSampleSteps = clampToStepRange(currentSampleSteps - 1);
		}
	}
}
