#include "ofxStableDiffusionRealtimeSession.h"
#include "../ofxStableDiffusion.h"
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
	generationInFlight = false;
	hasPendingRequest = false;
	warmingUp = false;
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

	// If no queuing is allowed and a generation is already running, drop this request
	if (generationInFlight && settings.maxQueueDepth == 0) {
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
	imageReq.sampleSteps = req.sampleSteps;
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

	// Increment dropped-frames counter if latency significantly exceeds target
	if (latencyMs > settings.targetLatencyMs * 1.5f) {
		stats.droppedFrames++;
	}
}
