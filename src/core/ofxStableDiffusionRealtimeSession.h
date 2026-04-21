#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionTypes.h"
#include <deque>
#include <functional>
#include <mutex>

class ofxStableDiffusion;

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
	int maxQueueDepth = 2;               // Maximum pending requests (0 = always drop when busy)
	/// Enable adaptive step count: steps increase toward maxSampleSteps when latency is
	/// comfortably below target, and decrease back to minSampleSteps when over target.
	bool enableProgressiveRefinement = true;
	int minSampleSteps = 4;              // Minimum steps for real-time (LCM/Turbo optimized)
	/// Upper bound for adaptive refinement.  Set higher than minSampleSteps to allow
	/// the session to increase quality when generation is fast enough.
	int maxSampleSteps = 4;
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
	/// Requests dropped because the generator was busy and could not be queued
	/// (maxQueueDepth == 0, or LowLatency mode, or the session was inactive).
	int droppedFrames = 0;
	/// Completed generations whose measured latency exceeded targetLatencyMs * 1.5.
	int slowFrames = 0;
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
	int sampleSteps = 4;                                 // LCM/Turbo optimized default
	sample_method_t sampleMethod = EULER_A_SAMPLE_METHOD;
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

	/// Start real-time generation session and attach a generator in one step.
	/// @param settings Session configuration
	/// @param sd Generator to use for all real-time generations
	/// @return True if session started successfully
	bool start(const ofxStableDiffusionRealtimeSettings& settings, ofxStableDiffusion& sd);

	/// Stop real-time generation session
	void stop();

	/// Check if session is active
	/// @return True if session is running
	bool isActive() const;

	/// Attach or detach the ofxStableDiffusion instance used for generation.
	/// Must be called before submit() or warmup() will do real work.
	/// Pass nullptr to detach.
	/// @param sd Pointer to the generator (not owned by this session)
	void setGenerator(ofxStableDiffusion* sd);

	/// Poll for completed generations and dispatch result/latency callbacks.
	/// Call this every frame from the application update() method.
	void update();

	/// Submit a real-time generation request
	/// @param request Lightweight request for real-time generation
	/// @return True if request was accepted (not dropped)
	bool submit(const ofxStableDiffusionRealtimeRequest& request);

	/// Update prompt on the fly (for next generation)
	/// @param prompt New prompt text
	void updatePrompt(const std::string& prompt);

	/// Update negative prompt on the fly (for next generation)
	/// @param negativePrompt New negative prompt text
	void updateNegativePrompt(const std::string& negativePrompt);

	/// Update CFG scale on the fly (for next generation)
	/// @param cfgScale New CFG scale value
	void updateCfgScale(float cfgScale);

	/// Update strength on the fly (for next generation)
	/// @param strength New strength value
	void updateStrength(float strength);

	/// Update seed on the fly (for next generation; use -1 for random)
	/// @param seed New seed value
	void updateSeed(int seed);

	/// Update sample steps on the fly.
	/// Also resets the adaptive step counter when progressive refinement is enabled.
	/// @param steps New sample step count
	void updateSampleSteps(int steps);

	/// Update output dimensions on the fly (for next generation).
	/// Width and height must be multiples of 64 and no greater than 2048.
	/// @param width New output width in pixels
	/// @param height New output height in pixels
	void updateDimensions(int width, int height);

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

	/// Check whether a generation is currently in progress.
	/// @return True if a generation has been fired and its result not yet received
	bool isGenerating() const;

	/// Get the current adaptive sample step count.
	/// When progressive refinement is disabled this equals the last submitted request's steps.
	/// @return Current sample steps used by processQueue
	int getCurrentSampleSteps() const;

	/// Warmup model (pre-generate to eliminate first-run latency).
	/// Requires a generator to be attached via setGenerator() or start(settings, sd)
	/// and the session to be active; does nothing otherwise.
	/// @return True if a warmup generation was fired successfully
	bool warmup();

private:
	void processQueue();
	void updateStats(float latencyMs);
	/// Clamp steps to the [minSampleSteps, maxSampleSteps] range from current settings.
	int clampToStepRange(int steps) const;

	ofxStableDiffusionRealtimeSettings settings;
	ofxStableDiffusionRealtimeStats stats;
	ofxSdRealtimeResultCallback resultCallback;
	ofxSdRealtimeLatencyCallback latencyCallback;

	ofxStableDiffusionRealtimeRequest pendingRequest;
	ofxStableDiffusionResult lastResult;

	bool active = false;
	std::mutex requestMutex;
	std::deque<float> latencyHistory;

	// Generator state (main-thread-only)
	ofxStableDiffusion* generator = nullptr;
	bool generationInFlight = false;
	bool hasPendingRequest = false;
	bool warmingUp = false;
	uint64_t generationStartMicros = 0;
	/// Session-managed adaptive step count (used when enableProgressiveRefinement is on).
	int currentSampleSteps = 4;
};
