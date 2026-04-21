#pragma once

#include "ofMain.h"
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <vector>

/// Performance profiling data for a specific operation
struct ofxStableDiffusionProfileEntry {
	std::string name;
	uint64_t startMicros = 0;
	uint64_t endMicros = 0;
	uint64_t durationMicros = 0;
	size_t memoryBytes = 0;
	int callCount = 0;

	float durationMs() const {
		return durationMicros / 1000.0f;
	}

	float durationSeconds() const {
		return durationMicros / 1000000.0f;
	}
};

/// Aggregated performance statistics
struct ofxStableDiffusionPerformanceStats {
	std::map<std::string, ofxStableDiffusionProfileEntry> entries;
	uint64_t totalDurationMicros = 0;
	size_t peakMemoryBytes = 0;
	size_t currentMemoryBytes = 0;

	float totalDurationMs() const {
		return totalDurationMicros / 1000.0f;
	}

	float totalDurationSeconds() const {
		return totalDurationMicros / 1000000.0f;
	}
};

/// RAII-style scoped timer for automatic profiling
class ofxStableDiffusionScopedTimer {
public:
	ofxStableDiffusionScopedTimer(const std::string& name, ofxStableDiffusionProfileEntry& entry)
		: name_(name), entry_(entry) {
		entry_.name = name;
		entry_.startMicros = ofGetElapsedTimeMicros();
		entry_.callCount++;
	}

	~ofxStableDiffusionScopedTimer() {
		entry_.endMicros = ofGetElapsedTimeMicros();
		entry_.durationMicros = entry_.endMicros - entry_.startMicros;
	}

private:
	std::string name_;
	ofxStableDiffusionProfileEntry& entry_;
};

/// Performance profiler for tracking generation pipeline bottlenecks
class ofxStableDiffusionPerformanceProfiler {
public:
	ofxStableDiffusionPerformanceProfiler();
	~ofxStableDiffusionPerformanceProfiler();

	/// Start tracking a named operation
	void begin(const std::string& name);

	/// End tracking a named operation
	void end(const std::string& name);

	/// Record memory usage
	void recordMemory(const std::string& name, size_t bytes);

	/// Get scoped timer for automatic profiling
	ofxStableDiffusionScopedTimer scopedTimer(const std::string& name);

	/// Get current statistics
	ofxStableDiffusionPerformanceStats getStats() const;

	/// Get specific entry
	ofxStableDiffusionProfileEntry getEntry(const std::string& name) const;

	/// Reset all profiling data
	void reset();

	/// Enable/disable profiling
	void setEnabled(bool enabled);
	bool isEnabled() const;

	/// Export profiling data to JSON
	std::string toJSON() const;

	/// Export profiling data to CSV
	std::string toCSV() const;

	/// Print summary to console
	void printSummary() const;

	/// Get bottleneck analysis (operations taking > threshold % of total time)
	std::vector<std::string> getBottlenecks(float thresholdPercent = 10.0f) const;

private:
	std::map<std::string, ofxStableDiffusionProfileEntry> entries_;
	std::map<std::string, uint64_t> activeTimers_;
	size_t peakMemory_ = 0;
	size_t currentMemory_ = 0;
	bool enabled_ = true;
	mutable std::mutex mutex_;

	void updatePeakMemory();
};
