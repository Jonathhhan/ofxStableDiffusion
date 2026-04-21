#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <vector>
#include <map>
#include <mutex>
#include <sstream>
#include <algorithm>

// Mock ofGetElapsedTimeMicros for testing
uint64_t testTimeMicros = 0;
uint64_t ofGetElapsedTimeMicros() {
	return testTimeMicros;
}

// Mock ofLogNotice, ofLogWarning
struct MockLogger {
	std::string message;
	MockLogger& operator<<(const std::string& msg) {
		message += msg;
		return *this;
	}
	MockLogger& operator<<(const char* msg) {
		message += msg;
		return *this;
	}
	MockLogger& operator<<(int val) {
		message += std::to_string(val);
		return *this;
	}
	MockLogger& operator<<(float val) {
		message += std::to_string(val);
		return *this;
	}
	MockLogger& operator<<(uint64_t val) {
		message += std::to_string(val);
		return *this;
	}
	~MockLogger() {}
};
MockLogger ofLogNotice(const std::string&) { return MockLogger(); }
MockLogger ofLogWarning(const std::string&) { return MockLogger(); }

// Include the header and implementation
#include "../src/core/ofxStableDiffusionPerformanceProfiler.h"
#include "../src/core/ofxStableDiffusionPerformanceProfiler.cpp"

void testBasicTiming() {
	std::cout << "Testing basic timing...";

	ofxStableDiffusionPerformanceProfiler profiler;

	testTimeMicros = 1000;
	profiler.begin("operation1");

	testTimeMicros = 3000;  // 2ms elapsed
	profiler.end("operation1");

	auto entry = profiler.getEntry("operation1");
	assert(entry.durationMicros == 2000);
	assert(entry.durationMs() == 2.0f);
	assert(entry.callCount == 1);

	std::cout << " ✓" << std::endl;
}

void testMultipleCalls() {
	std::cout << "Testing multiple calls...";

	ofxStableDiffusionPerformanceProfiler profiler;

	// First call
	testTimeMicros = 1000;
	profiler.begin("operation");
	testTimeMicros = 2000;
	profiler.end("operation");

	// Second call
	testTimeMicros = 3000;
	profiler.begin("operation");
	testTimeMicros = 5500;
	profiler.end("operation");

	auto entry = profiler.getEntry("operation");
	assert(entry.durationMicros == 3500);  // 1000 + 2500
	assert(entry.callCount == 2);

	std::cout << " ✓" << std::endl;
}

void testMemoryRecording() {
	std::cout << "Testing memory recording...";

	ofxStableDiffusionPerformanceProfiler profiler;

	profiler.recordMemory("model_load", 500 * 1024 * 1024);  // 500MB
	profiler.recordMemory("model_load", 300 * 1024 * 1024);  // 300MB (should keep max)

	auto entry = profiler.getEntry("model_load");
	assert(entry.memoryBytes == 500 * 1024 * 1024);  // Keeps max

	auto stats = profiler.getStats();
	assert(stats.peakMemoryBytes == 500 * 1024 * 1024);
	assert(stats.currentMemoryBytes == 300 * 1024 * 1024);

	std::cout << " ✓" << std::endl;
}

void testScopedTimer() {
	std::cout << "Testing scoped timer...";

	ofxStableDiffusionPerformanceProfiler profiler;

	{
		testTimeMicros = 1000;
		auto timer = profiler.scopedTimer("scoped_op");
		testTimeMicros = 4000;  // 3ms will be recorded when timer destructs
	}

	auto entry = profiler.getEntry("scoped_op");
	assert(entry.durationMicros == 3000);
	assert(entry.callCount == 1);

	std::cout << " ✓" << std::endl;
}

void testStats() {
	std::cout << "Testing stats aggregation...";

	ofxStableDiffusionPerformanceProfiler profiler;

	testTimeMicros = 0;
	profiler.begin("op1");
	testTimeMicros = 1000;
	profiler.end("op1");

	testTimeMicros = 2000;
	profiler.begin("op2");
	testTimeMicros = 5000;
	profiler.end("op2");

	auto stats = profiler.getStats();
	assert(stats.totalDurationMicros == 4000);  // 1000 + 3000
	assert(stats.totalDurationMs() == 4.0f);
	assert(stats.entries.size() == 2);
	assert(stats.entries.count("op1") == 1);
	assert(stats.entries.count("op2") == 1);

	std::cout << " ✓" << std::endl;
}

void testBottleneckDetection() {
	std::cout << "Testing bottleneck detection...";

	ofxStableDiffusionPerformanceProfiler profiler;

	testTimeMicros = 0;
	profiler.begin("fast_op");
	testTimeMicros = 100;  // 0.1ms (1% of total)
	profiler.end("fast_op");

	testTimeMicros = 200;
	profiler.begin("slow_op");
	testTimeMicros = 9900;  // 9.7ms (97% of total)
	profiler.end("slow_op");

	testTimeMicros = 10000;
	profiler.begin("medium_op");
	testTimeMicros = 10200;  // 0.2ms (2% of total)
	profiler.end("medium_op");

	// Get bottlenecks > 10% of total time
	auto bottlenecks = profiler.getBottlenecks(10.0f);
	assert(bottlenecks.size() == 1);
	assert(bottlenecks[0] == "slow_op");

	// Lower threshold should catch more
	bottlenecks = profiler.getBottlenecks(1.5f);
	assert(bottlenecks.size() == 2);  // slow_op and medium_op

	std::cout << " ✓" << std::endl;
}

void testEnableDisable() {
	std::cout << "Testing enable/disable...";

	ofxStableDiffusionPerformanceProfiler profiler;

	assert(profiler.isEnabled());

	profiler.setEnabled(false);
	assert(!profiler.isEnabled());

	// Operations when disabled should be no-ops
	testTimeMicros = 0;
	profiler.begin("disabled_op");
	testTimeMicros = 1000;
	profiler.end("disabled_op");

	auto entry = profiler.getEntry("disabled_op");
	assert(entry.durationMicros == 0);
	assert(entry.callCount == 0);

	// Re-enable
	profiler.setEnabled(true);
	testTimeMicros = 2000;
	profiler.begin("enabled_op");
	testTimeMicros = 3000;
	profiler.end("enabled_op");

	entry = profiler.getEntry("enabled_op");
	assert(entry.durationMicros == 1000);
	assert(entry.callCount == 1);

	std::cout << " ✓" << std::endl;
}

void testReset() {
	std::cout << "Testing reset...";

	ofxStableDiffusionPerformanceProfiler profiler;

	testTimeMicros = 0;
	profiler.begin("op");
	testTimeMicros = 1000;
	profiler.end("op");
	profiler.recordMemory("op", 1024);

	auto stats = profiler.getStats();
	assert(stats.totalDurationMicros > 0);
	assert(stats.currentMemoryBytes > 0);

	profiler.reset();

	stats = profiler.getStats();
	assert(stats.totalDurationMicros == 0);
	assert(stats.peakMemoryBytes == 0);
	assert(stats.currentMemoryBytes == 0);
	assert(stats.entries.empty());

	std::cout << " ✓" << std::endl;
}

void testJSONExport() {
	std::cout << "Testing JSON export...";

	ofxStableDiffusionPerformanceProfiler profiler;

	testTimeMicros = 0;
	profiler.begin("operation");
	testTimeMicros = 2000;
	profiler.end("operation");
	profiler.recordMemory("operation", 1024 * 1024);

	std::string json = profiler.toJSON();

	// Basic validation - check for expected structure
	assert(json.find("\"entries\"") != std::string::npos);
	assert(json.find("\"operation\"") != std::string::npos);
	assert(json.find("\"durationMs\"") != std::string::npos);
	assert(json.find("\"callCount\"") != std::string::npos);
	assert(json.find("\"memoryBytes\"") != std::string::npos);
	assert(json.find("\"peakMemoryBytes\"") != std::string::npos);

	std::cout << " ✓" << std::endl;
}

void testCSVExport() {
	std::cout << "Testing CSV export...";

	ofxStableDiffusionPerformanceProfiler profiler;

	testTimeMicros = 0;
	profiler.begin("op1");
	testTimeMicros = 1000;
	profiler.end("op1");

	testTimeMicros = 2000;
	profiler.begin("op2");
	testTimeMicros = 3500;
	profiler.end("op2");

	std::string csv = profiler.toCSV();

	// Check header
	assert(csv.find("name,durationMs,callCount,memoryBytes") != std::string::npos);
	assert(csv.find("op1") != std::string::npos);
	assert(csv.find("op2") != std::string::npos);

	std::cout << " ✓" << std::endl;
}

void testMismatchedEndCall() {
	std::cout << "Testing mismatched end() call...";

	ofxStableDiffusionPerformanceProfiler profiler;

	testTimeMicros = 0;
	profiler.begin("operation");
	testTimeMicros = 1000;

	// Call end with different name - should log warning and not crash
	profiler.end("different_operation");

	// The original operation should still be in active timers
	profiler.end("operation");

	auto entry = profiler.getEntry("operation");
	assert(entry.durationMicros == 1000);

	std::cout << " ✓" << std::endl;
}

int main() {
	try {
		testBasicTiming();
		testMultipleCalls();
		testMemoryRecording();
		testScopedTimer();
		testStats();
		testBottleneckDetection();
		testEnableDisable();
		testReset();
		testJSONExport();
		testCSVExport();
		testMismatchedEndCall();

		std::cout << "\n✅ All performance profiler tests passed!" << std::endl;
		return 0;
	} catch (const std::exception& e) {
		std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
		return 1;
	}
}
