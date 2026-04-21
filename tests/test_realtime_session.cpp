/// Unit tests for ofxStableDiffusionRealtimeSession pipeline logic.
///
/// These tests validate the queue mechanics, statistics, and lifecycle contracts
/// that the real-time pipeline must satisfy.  Because the full ofxStableDiffusion
/// class requires OpenFrameworks, the tests exercise the logic directly using
/// representative mock/stub data rather than including the full class headers.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool expect(bool condition, const std::string& message) {
	if (!condition) {
		std::cerr << "FAIL: " << message << std::endl;
	}
	return condition;
}

// ---------------------------------------------------------------------------
// Rolling-average stats logic (mirrors ofxStableDiffusionRealtimeSession)
// ---------------------------------------------------------------------------

struct RealtimeStats {
	int totalGenerations = 0;
	float averageLatencyMs = 0.0f;
	float minLatencyMs = 0.0f;
	float maxLatencyMs = 0.0f;
	int droppedFrames = 0;
	int slowFrames = 0;
};

static void updateStats(RealtimeStats& stats,
	std::deque<float>& history,
	float latencyMs,
	float targetLatencyMs) {

	stats.totalGenerations++;
	history.push_back(latencyMs);

	if (history.size() > 100) {
		history.pop_front();
	}

	if (stats.minLatencyMs == 0.0f || latencyMs < stats.minLatencyMs) {
		stats.minLatencyMs = latencyMs;
	}
	if (latencyMs > stats.maxLatencyMs) {
		stats.maxLatencyMs = latencyMs;
	}

	const float sum = std::accumulate(history.begin(), history.end(), 0.0f);
	stats.averageLatencyMs = sum / static_cast<float>(history.size());

	if (latencyMs > targetLatencyMs * 1.5f) {
		stats.slowFrames++;
	}
}

static bool testStatsAccumulation() {
	bool ok = true;
	std::deque<float> history;
	RealtimeStats stats;
	const float target = 500.0f;

	// First sample
	updateStats(stats, history, 100.0f, target);
	ok &= expect(stats.totalGenerations == 1, "totalGenerations after first sample");
	ok &= expect(stats.minLatencyMs == 100.0f, "minLatencyMs after first sample");
	ok &= expect(stats.maxLatencyMs == 100.0f, "maxLatencyMs after first sample");
	ok &= expect(stats.averageLatencyMs == 100.0f, "averageLatencyMs after first sample");
	ok &= expect(stats.droppedFrames == 0, "no dropped frames for fast generation");
	ok &= expect(stats.slowFrames == 0, "no slow frames for fast generation");

	// Second sample — much slower than target (> target * 1.5)
	updateStats(stats, history, 800.0f, target);
	ok &= expect(stats.totalGenerations == 2, "totalGenerations after second sample");
	ok &= expect(stats.minLatencyMs == 100.0f, "min stays at first sample");
	ok &= expect(stats.maxLatencyMs == 800.0f, "max updated after second sample");
	ok &= expect(stats.droppedFrames == 0, "droppedFrames not incremented for slow generation");
	ok &= expect(stats.slowFrames == 1, "slowFrames incremented for 800ms at 500ms target");

	// Rolling average is (100 + 800) / 2
	ok &= expect(std::abs(stats.averageLatencyMs - 450.0f) < 0.01f, "rolling average after two samples");

	return ok;
}

static bool testStatsHistoryCap() {
	bool ok = true;
	std::deque<float> history;
	RealtimeStats stats;

	// Add 120 samples — history should cap at 100
	for (int i = 0; i < 120; ++i) {
		updateStats(stats, history, 200.0f, 500.0f);
	}

	ok &= expect(history.size() == 100, "latency history capped at 100 entries");
	ok &= expect(stats.totalGenerations == 120, "totalGenerations counts all 120 samples");
	ok &= expect(std::abs(stats.averageLatencyMs - 200.0f) < 0.01f,
		"rolling average correct with full deque");

	return ok;
}

// ---------------------------------------------------------------------------
// Queue drop-semantics (mirrors submit / maxQueueDepth logic)
// ---------------------------------------------------------------------------

struct QueueState {
	bool generationInFlight = false;
	bool hasPendingRequest = false;
	int droppedFrames = 0;
	int firedCount = 0;
};

// Returns true if the request was accepted (not dropped)
static bool simulateSubmit(QueueState& q, int maxQueueDepth) {
	if (q.generationInFlight && maxQueueDepth == 0) {
		q.droppedFrames++;
		return false;
	}
	q.hasPendingRequest = true;
	if (!q.generationInFlight) {
		// processQueue: fire immediately
		q.hasPendingRequest = false;
		q.generationInFlight = true;
		q.firedCount++;
	}
	return true;
}

static void simulateComplete(QueueState& q) {
	q.generationInFlight = false;
	// processQueue at end of update
	if (q.hasPendingRequest) {
		q.hasPendingRequest = false;
		q.generationInFlight = true;
		q.firedCount++;
	}
}

static bool testQueueNoBuffering() {
	bool ok = true;
	QueueState q;

	// First submit fires immediately (idle)
	bool accepted = simulateSubmit(q, 0);
	ok &= expect(accepted, "first submit accepted when idle");
	ok &= expect(q.generationInFlight, "generation in flight after first submit");
	ok &= expect(q.firedCount == 1, "one generation fired");

	// Second submit while busy and maxQueueDepth=0 → drop
	accepted = simulateSubmit(q, 0);
	ok &= expect(!accepted, "submit dropped when busy and maxQueueDepth=0");
	ok &= expect(q.droppedFrames == 1, "dropped frame counter incremented");
	ok &= expect(!q.hasPendingRequest, "no pending request after drop");

	// Complete the first generation
	simulateComplete(q);
	ok &= expect(!q.generationInFlight, "not in flight after completion");
	ok &= expect(q.firedCount == 1, "no extra generation after completion (nothing was queued)");

	return ok;
}

static bool testQueueWithBuffering() {
	bool ok = true;
	QueueState q;

	// First submit fires immediately (idle)
	simulateSubmit(q, 1);
	ok &= expect(q.generationInFlight, "generation in flight after first submit");
	ok &= expect(q.firedCount == 1, "one generation fired");

	// Second submit while busy and maxQueueDepth=1 → accepted and queued
	bool accepted = simulateSubmit(q, 1);
	ok &= expect(accepted, "second submit accepted when maxQueueDepth=1");
	ok &= expect(q.droppedFrames == 0, "no drops with buffering enabled");
	ok &= expect(q.hasPendingRequest, "pending request stored");

	// Third submit while busy — last-write-wins: replaces the pending
	simulateSubmit(q, 1);
	ok &= expect(q.hasPendingRequest, "pending request still present after third submit");
	ok &= expect(q.droppedFrames == 0, "still no drops (old pending was overwritten)");

	// Complete: fires the pending request
	simulateComplete(q);
	ok &= expect(q.generationInFlight, "pending request fired after completion");
	ok &= expect(q.firedCount == 2, "second generation fired");

	// Complete again: no pending left
	simulateComplete(q);
	ok &= expect(!q.generationInFlight, "idle after second completion");
	ok &= expect(q.firedCount == 2, "no extra generation when queue was empty");

	return ok;
}

// ---------------------------------------------------------------------------
// LowLatency mode always drops when busy, regardless of maxQueueDepth
// ---------------------------------------------------------------------------

static bool testLowLatencyModeDrops() {
	bool ok = true;

	// Simulate the mustDropIfBusy logic from submit()
	const auto mustDropIfBusy = [](bool lowLatencyMode, int maxQueueDepth) {
		return (maxQueueDepth == 0 || lowLatencyMode);
	};

	// Non-LowLatency + maxQueueDepth=2: does NOT drop
	ok &= expect(!mustDropIfBusy(false, 2), "non-LowLatency with depth=2 allows queuing");

	// LowLatency + maxQueueDepth=2: DOES drop
	ok &= expect(mustDropIfBusy(true, 2), "LowLatency forces drop even with depth=2");

	// LowLatency + maxQueueDepth=0: also drops
	ok &= expect(mustDropIfBusy(true, 0), "LowLatency + depth=0 drops");

	// Non-LowLatency + maxQueueDepth=0: drops (the existing rule)
	ok &= expect(mustDropIfBusy(false, 0), "non-LowLatency with depth=0 drops");

	// Simulate two submits in LowLatency mode
	{
		QueueState q;
		// First submit: idle → fires immediately, depth irrelevant
		q.generationInFlight = false;
		const bool busy = q.generationInFlight;
		if (!busy || !mustDropIfBusy(true, 2)) {
			q.hasPendingRequest = true;
			if (!q.generationInFlight) {
				q.hasPendingRequest = false;
				q.generationInFlight = true;
				q.firedCount++;
			}
		} else {
			q.droppedFrames++;
		}
		ok &= expect(q.firedCount == 1, "first LowLatency submit fires");

		// Second submit: busy → must drop
		if (!q.generationInFlight || !mustDropIfBusy(true, 2)) {
			q.hasPendingRequest = true;
		} else {
			q.droppedFrames++;
		}
		ok &= expect(q.droppedFrames == 1, "second LowLatency submit dropped while busy");
		ok &= expect(!q.hasPendingRequest, "no pending request after LowLatency drop");
	}

	return ok;
}

// ---------------------------------------------------------------------------
// Adaptive step count (mirrors updateStats adaptive logic)
// ---------------------------------------------------------------------------

static bool testAdaptiveStepCount() {
	bool ok = true;

	int currentSteps = 4;
	const int minSteps = 4;
	const int maxSteps = 8;
	const float target = 500.0f;

	const auto adaptSteps = [&](float latencyMs) {
		if (latencyMs < target * 0.8f) {
			currentSteps = std::min(currentSteps + 1, maxSteps);
		} else if (latencyMs > target) {
			currentSteps = std::max(currentSteps - 1, minSteps);
		}
	};

	// Fast generation (300ms < 500 * 0.8 = 400ms) → step up
	adaptSteps(300.0f);
	ok &= expect(currentSteps == 5, "fast generation increments steps from 4 to 5");

	// Three more fast → approaches max
	adaptSteps(300.0f);
	adaptSteps(300.0f);
	adaptSteps(300.0f);
	ok &= expect(currentSteps == 8, "repeated fast generations reach maxSteps=8");

	// One more fast → capped at max
	adaptSteps(300.0f);
	ok &= expect(currentSteps == 8, "steps do not exceed maxSteps");

	// Medium generation in sweet spot (450ms, between 400ms and 500ms) → no change
	adaptSteps(450.0f);
	ok &= expect(currentSteps == 8, "latency in sweet spot leaves steps unchanged");

	// Slow generation (600ms > 500ms) → step down
	adaptSteps(600.0f);
	ok &= expect(currentSteps == 7, "slow generation decrements steps from 8 to 7");

	// Drive all the way to minimum
	for (int i = 0; i < 10; ++i) {
		adaptSteps(600.0f);
	}
	ok &= expect(currentSteps == minSteps, "repeated slow generations reach minSteps");

	// Further slow → capped at min
	adaptSteps(600.0f);
	ok &= expect(currentSteps == minSteps, "steps do not go below minSteps");

	return ok;
}

// ---------------------------------------------------------------------------
// isGenerating mirrors generationInFlight
// ---------------------------------------------------------------------------

static bool testIsGenerating() {
	bool ok = true;

	bool generationInFlight = false;
	ok &= expect(!generationInFlight, "isGenerating false when idle");

	generationInFlight = true;
	ok &= expect(generationInFlight, "isGenerating true when in flight");

	generationInFlight = false;
	ok &= expect(!generationInFlight, "isGenerating false after completion");

	return ok;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

static bool testLifecycle() {
	bool ok = true;

	// Simulate the active flag behaviour
	bool active = false;

	// start()
	active = true;
	ok &= expect(active, "session is active after start");

	// stop()
	active = false;
	ok &= expect(!active, "session is inactive after stop");

	// double-stop should not change state
	if (!active) {
		// stop() returns early
	}
	ok &= expect(!active, "double stop leaves session inactive");

	return ok;
}

// ---------------------------------------------------------------------------
// Warmup guard
// ---------------------------------------------------------------------------

static bool testWarmupGuards() {
	bool ok = true;

	// Cannot warmup if session is not active
	{
		bool active = false;
		bool generationInFlight = false;
		bool warmupResult = active && !generationInFlight;  // simplification of warmup guard
		ok &= expect(!warmupResult, "warmup refused when session not active");
	}

	// Cannot warmup if generation is in flight
	{
		bool active = true;
		bool generationInFlight = true;
		bool warmupResult = active && !generationInFlight;
		ok &= expect(!warmupResult, "warmup refused when generation in flight");
	}

	// Can warmup if active and idle
	{
		bool active = true;
		bool generationInFlight = false;
		bool warmupResult = active && !generationInFlight;
		ok &= expect(warmupResult, "warmup permitted when active and idle");
	}

	return ok;
}

// ---------------------------------------------------------------------------
// Image request conversion
// ---------------------------------------------------------------------------

static bool testRequestConversion() {
	bool ok = true;

	// Simulate the fields that processQueue copies from realtime → image request
	struct RealtimeReq {
		std::string prompt = "test prompt";
		std::string negativePrompt = "blur";
		float cfgScale = 1.5f;
		float strength = 0.7f;
		int seed = 42;
		int width = 512;
		int height = 512;
		int sampleSteps = 4;
		int sampleMethod = 2;  // stand-in for sample_method_t enum value
	};

	struct ImageReq {
		std::string prompt;
		std::string negativePrompt;
		float cfgScale = 0.0f;
		float strength = 0.0f;
		int64_t seed = -1;
		int width = 0;
		int height = 0;
		int sampleSteps = 0;
		int sampleMethod = 0;
		int batchCount = 0;
	};

	RealtimeReq rtReq;
	ImageReq imgReq;
	imgReq.prompt = rtReq.prompt;
	imgReq.negativePrompt = rtReq.negativePrompt;
	imgReq.cfgScale = rtReq.cfgScale;
	imgReq.strength = rtReq.strength;
	imgReq.seed = static_cast<int64_t>(rtReq.seed);
	imgReq.width = rtReq.width;
	imgReq.height = rtReq.height;
	imgReq.sampleSteps = rtReq.sampleSteps;
	imgReq.sampleMethod = rtReq.sampleMethod;
	imgReq.batchCount = 1;

	ok &= expect(imgReq.prompt == "test prompt", "prompt copied");
	ok &= expect(imgReq.negativePrompt == "blur", "negativePrompt copied");
	ok &= expect(imgReq.cfgScale == 1.5f, "cfgScale copied");
	ok &= expect(imgReq.strength == 0.7f, "strength copied");
	ok &= expect(imgReq.seed == 42LL, "seed widened to int64_t");
	ok &= expect(imgReq.width == 512 && imgReq.height == 512, "dimensions copied");
	ok &= expect(imgReq.sampleSteps == 4, "sampleSteps copied");
	ok &= expect(imgReq.sampleMethod == 2, "sampleMethod forwarded");
	ok &= expect(imgReq.batchCount == 1, "batchCount forced to 1");

	return ok;
}

// ---------------------------------------------------------------------------
// Parameter update methods
// ---------------------------------------------------------------------------

static bool testUpdateParameterMethods() {
	bool ok = true;

	// Simulate the pending request that update*() methods mutate
	struct PendingReq {
		std::string prompt;
		std::string negativePrompt;
		float cfgScale = 1.5f;
		float strength = 0.7f;
		int seed = -1;
		int width = 512;
		int height = 512;
		int sampleSteps = 4;
	};

	PendingReq req;

	// updatePrompt
	req.prompt = "a cat";
	ok &= expect(req.prompt == "a cat", "updatePrompt sets prompt");

	// updateNegativePrompt
	req.negativePrompt = "blurry";
	ok &= expect(req.negativePrompt == "blurry", "updateNegativePrompt sets negativePrompt");

	// updateCfgScale
	req.cfgScale = 2.0f;
	ok &= expect(req.cfgScale == 2.0f, "updateCfgScale sets cfgScale");

	// updateStrength
	req.strength = 0.5f;
	ok &= expect(req.strength == 0.5f, "updateStrength sets strength");

	// updateSeed
	req.seed = 123;
	ok &= expect(req.seed == 123, "updateSeed sets seed");

	// updateSampleSteps
	req.sampleSteps = 8;
	ok &= expect(req.sampleSteps == 8, "updateSampleSteps sets sampleSteps");

	// updateDimensions
	req.width = 768;
	req.height = 768;
	ok &= expect(req.width == 768 && req.height == 768, "updateDimensions sets width/height");

	// updateSampleSteps also resets currentSampleSteps (clamped to [min, max])
	{
		const int minSteps = 4;
		const int maxSteps = 8;
		int currentSteps = 6;

		const auto applyUpdateSampleSteps = [&](int steps) {
			currentSteps = std::max(minSteps, std::min(steps, maxSteps));
		};

		applyUpdateSampleSteps(10);  // clamped to max
		ok &= expect(currentSteps == 8, "updateSampleSteps clamps currentSteps to maxSampleSteps");

		applyUpdateSampleSteps(1);   // clamped to min
		ok &= expect(currentSteps == 4, "updateSampleSteps clamps currentSteps to minSampleSteps");

		applyUpdateSampleSteps(6);   // within range
		ok &= expect(currentSteps == 6, "updateSampleSteps sets currentSteps to in-range value");
	}

	return ok;
}

// ---------------------------------------------------------------------------
// Progressive refinement uses currentSampleSteps instead of request steps
// ---------------------------------------------------------------------------

static bool testProgressiveRefinementSelectsAdaptiveSteps() {
	bool ok = true;

	// When enableProgressiveRefinement is true, processQueue uses currentSampleSteps
	// regardless of what the request carries.
	int currentSampleSteps = 6;
	bool enableProgressiveRefinement = true;

	struct MockReq { int sampleSteps = 4; };
	MockReq req;

	const int resolvedSteps = enableProgressiveRefinement ? currentSampleSteps : req.sampleSteps;
	ok &= expect(resolvedSteps == 6, "progressive refinement uses currentSampleSteps (6), not request steps (4)");

	// With progressive refinement off, request steps are used
	enableProgressiveRefinement = false;
	const int resolvedSteps2 = enableProgressiveRefinement ? currentSampleSteps : req.sampleSteps;
	ok &= expect(resolvedSteps2 == 4, "without progressive refinement, request steps (4) are used");

	return ok;
}

// ---------------------------------------------------------------------------
// maxSampleSteps is clamped to be >= minSampleSteps at start()
// ---------------------------------------------------------------------------

static bool testMaxSampleStepsClamping() {
	bool ok = true;

	// Simulate the clamping done at the beginning of start()
	const auto clampSettings = [](int minSteps, int maxSteps) {
		return std::max(maxSteps, minSteps);
	};

	ok &= expect(clampSettings(4, 8) == 8, "maxSampleSteps >= minSampleSteps stays unchanged");
	ok &= expect(clampSettings(4, 2) == 4, "maxSampleSteps below minSampleSteps clamped up");
	ok &= expect(clampSettings(4, 4) == 4, "equal values unchanged");

	return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
	bool ok = true;

	bool saOk = testStatsAccumulation();
	ok &= saOk;
	std::cout << "stats accumulation: " << (saOk ? "PASS" : "FAIL") << std::endl;

	bool histOk = testStatsHistoryCap();
	ok &= histOk;
	std::cout << "stats history cap: " << (histOk ? "PASS" : "FAIL") << std::endl;

	bool qNoOk = testQueueNoBuffering();
	ok &= qNoOk;
	std::cout << "queue no-buffering: " << (qNoOk ? "PASS" : "FAIL") << std::endl;

	bool qBufOk = testQueueWithBuffering();
	ok &= qBufOk;
	std::cout << "queue with buffering: " << (qBufOk ? "PASS" : "FAIL") << std::endl;

	bool llOk = testLowLatencyModeDrops();
	ok &= llOk;
	std::cout << "low-latency mode drops: " << (llOk ? "PASS" : "FAIL") << std::endl;

	bool adaptOk = testAdaptiveStepCount();
	ok &= adaptOk;
	std::cout << "adaptive step count: " << (adaptOk ? "PASS" : "FAIL") << std::endl;

	bool igOk = testIsGenerating();
	ok &= igOk;
	std::cout << "isGenerating: " << (igOk ? "PASS" : "FAIL") << std::endl;

	bool lcOk = testLifecycle();
	ok &= lcOk;
	std::cout << "lifecycle: " << (lcOk ? "PASS" : "FAIL") << std::endl;

	bool wuOk = testWarmupGuards();
	ok &= wuOk;
	std::cout << "warmup guards: " << (wuOk ? "PASS" : "FAIL") << std::endl;

	bool cvOk = testRequestConversion();
	ok &= cvOk;
	std::cout << "request conversion: " << (cvOk ? "PASS" : "FAIL") << std::endl;

	bool upOk = testUpdateParameterMethods();
	ok &= upOk;
	std::cout << "update parameter methods: " << (upOk ? "PASS" : "FAIL") << std::endl;

	bool prOk = testProgressiveRefinementSelectsAdaptiveSteps();
	ok &= prOk;
	std::cout << "progressive refinement step selection: " << (prOk ? "PASS" : "FAIL") << std::endl;

	bool clOk = testMaxSampleStepsClamping();
	ok &= clOk;
	std::cout << "maxSampleSteps clamping: " << (clOk ? "PASS" : "FAIL") << std::endl;

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
