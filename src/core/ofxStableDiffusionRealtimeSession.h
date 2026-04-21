#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionTypes.h"
#include <functional>

/// Real-time generation mode
enum class ofxStableDiffusionRealtimeMode {
	Streaming,      // Streaming mode with progressive refinement
	LowLatency,     // Optimized for minimal latency
	Interactive     // Interactive mode with live parameter updates
};

/// Real-time session settings
struct ofxStableDiffusionRealtimeSettings {
	ofxStableDiffusionRealtimeMode mode = ofxStableDiffusionRealtimeMode::Streaming;
	int targetLatencyMs = 500;           // Target generation time in milliseconds
	bool enableWarmup = true;            // Warmup model on session start
	int maxQueueDepth = 2;               // Maximum pending requests
	bool enableProgressiveRefinement = true;  // Enable progressive quality improvement
	int minSampleSteps = 4;              // Minimum steps for real-time (LCM/Turbo optimized)
	bool enableModelCaching = true;      // Cache model in memory during session
	float cfgScale = 1.5f;               // Default CFG scale for real-time
};

/// Real-time session statistics
struct ofxStableDiffusionRealtimeStats {
	uint64_t sessionStartTime = 0;
	int totalGenerations = 0;
	float averageLatencyMs = 0.0f;
	float minLatencyMs = 0.0f;
	float maxLatencyMs = 0.0f;
	int droppedFrames = 0;
	bool isActive = false;
};

/// Real-time generation request (lightweight)
struct ofxStableDiffusionRealtimeRequest {
	std::string prompt;
	std::string negativePrompt;
	float cfgScale = 1.5f;
	float strength = 0.7f;
	int seed = -1;
	int width = 512;
	int height = 512;
	int sampleSteps = 4;  // LCM/Turbo optimized default
};

/// Callback types for real-time generation
using ofxSdRealtimeResultCallback = std::function<void(const ofxStableDiffusionResult&)>;
using ofxSdRealtimeLatencyCallback = std::function<void(float latencyMs)>;

/// Real-time generation session manager
class ofxStableDiffusionRealtimeSession {
public:
	ofxStableDiffusionRealtimeSession();
	~ofxStableDiffusionRealtimeSession();

	/// Start real-time generation session
	/// @param settings Session configuration
	/// @return True if session started successfully
	bool start(const ofxStableDiffusionRealtimeSettings& settings);

	/// Stop real-time generation session
	void stop();

	/// Check if session is active
	/// @return True if session is running
	bool isActive() const;

	/// Submit a real-time generation request
	/// @param request Lightweight request for real-time generation
	/// @return True if request was accepted (not dropped)
	bool submit(const ofxStableDiffusionRealtimeRequest& request);

	/// Update prompt on the fly (for next generation)
	/// @param prompt New prompt text
	void updatePrompt(const std::string& prompt);

	/// Update CFG scale on the fly (for next generation)
	/// @param cfgScale New CFG scale value
	void updateCfgScale(float cfgScale);

	/// Update strength on the fly (for next generation)
	/// @param strength New strength value
	void updateStrength(float strength);

	/// Set result callback (called when generation completes)
	/// @param callback Result callback function
	void setResultCallback(ofxSdRealtimeResultCallback callback);

	/// Set latency callback (called with generation latency)
	/// @param callback Latency callback function
	void setLatencyCallback(ofxSdRealtimeLatencyCallback callback);

	/// Get current session statistics
	/// @return Session statistics
	ofxStableDiffusionRealtimeStats getStats() const;

	/// Get current settings
	/// @return Session settings
	ofxStableDiffusionRealtimeSettings getSettings() const;

	/// Reset statistics
	void resetStats();

	/// Get last generated result
	/// @return Last result (may be empty if none generated yet)
	ofxStableDiffusionResult getLastResult() const;

	/// Warmup model (pre-generate to eliminate first-run latency)
	/// @return True if warmup successful
	bool warmup();

private:
	void processQueue();
	void updateStats(float latencyMs);

	ofxStableDiffusionRealtimeSettings settings;
	ofxStableDiffusionRealtimeStats stats;
	ofxSdRealtimeResultCallback resultCallback;
	ofxSdRealtimeLatencyCallback latencyCallback;

	ofxStableDiffusionRealtimeRequest pendingRequest;
	ofxStableDiffusionResult lastResult;

	bool active = false;
	std::mutex requestMutex;
	std::vector<float> latencyHistory;
};
