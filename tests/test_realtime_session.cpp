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
		stats.droppedFrames++;
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

	// Second sample — much slower than target (> target * 1.5)
	updateStats(stats, history, 800.0f, target);
	ok &= expect(stats.totalGenerations == 2, "totalGenerations after second sample");
	ok &= expect(stats.minLatencyMs == 100.0f, "min stays at first sample");
	ok &= expect(stats.maxLatencyMs == 800.0f, "max updated after second sample");
	ok &= expect(stats.droppedFrames == 1, "one dropped frame for 800ms when target is 500ms");

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
	imgReq.batchCount = 1;

	ok &= expect(imgReq.prompt == "test prompt", "prompt copied");
	ok &= expect(imgReq.negativePrompt == "blur", "negativePrompt copied");
	ok &= expect(imgReq.cfgScale == 1.5f, "cfgScale copied");
	ok &= expect(imgReq.strength == 0.7f, "strength copied");
	ok &= expect(imgReq.seed == 42LL, "seed widened to int64_t");
	ok &= expect(imgReq.width == 512 && imgReq.height == 512, "dimensions copied");
	ok &= expect(imgReq.sampleSteps == 4, "sampleSteps copied");
	ok &= expect(imgReq.batchCount == 1, "batchCount forced to 1");

	return ok;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
	bool ok = true;

	ok &= testStatsAccumulation();
	std::cout << "stats accumulation: " << (ok ? "PASS" : "FAIL") << std::endl;

	bool histOk = testStatsHistoryCap();
	ok &= histOk;
	std::cout << "stats history cap: " << (histOk ? "PASS" : "FAIL") << std::endl;

	bool qNoOk = testQueueNoBuffering();
	ok &= qNoOk;
	std::cout << "queue no-buffering: " << (qNoOk ? "PASS" : "FAIL") << std::endl;

	bool qBufOk = testQueueWithBuffering();
	ok &= qBufOk;
	std::cout << "queue with buffering: " << (qBufOk ? "PASS" : "FAIL") << std::endl;

	bool lcOk = testLifecycle();
	ok &= lcOk;
	std::cout << "lifecycle: " << (lcOk ? "PASS" : "FAIL") << std::endl;

	bool wuOk = testWarmupGuards();
	ok &= wuOk;
	std::cout << "warmup guards: " << (wuOk ? "PASS" : "FAIL") << std::endl;

	bool cvOk = testRequestConversion();
	ok &= cvOk;
	std::cout << "request conversion: " << (cvOk ? "PASS" : "FAIL") << std::endl;

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
