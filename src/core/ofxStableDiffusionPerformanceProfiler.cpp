#include "ofxStableDiffusionPerformanceProfiler.h"
#include <algorithm>
#include <sstream>

ofxStableDiffusionPerformanceProfiler::ofxStableDiffusionPerformanceProfiler() {
}

ofxStableDiffusionPerformanceProfiler::~ofxStableDiffusionPerformanceProfiler() {
}

void ofxStableDiffusionPerformanceProfiler::begin(const std::string& name) {
	if (!enabled_) return;

	std::lock_guard<std::mutex> lock(mutex_);
	activeTimers_[name] = ofGetElapsedTimeMicros();

	if (entries_.find(name) == entries_.end()) {
		entries_[name].name = name;
		entries_[name].callCount = 0;
	}
	entries_[name].callCount++;
}

void ofxStableDiffusionPerformanceProfiler::end(const std::string& name) {
	if (!enabled_) return;

	const uint64_t endTime = ofGetElapsedTimeMicros();

	std::lock_guard<std::mutex> lock(mutex_);
	auto it = activeTimers_.find(name);
	if (it == activeTimers_.end()) {
		ofLogWarning("ofxStableDiffusionPerformanceProfiler") << "end() called without matching begin() for: " << name;
		return;
	}

	const uint64_t duration = endTime - it->second;
	activeTimers_.erase(it);

	auto& entry = entries_[name];
	entry.endMicros = endTime;
	entry.startMicros = it->second;
	entry.durationMicros += duration;
}

void ofxStableDiffusionPerformanceProfiler::recordMemory(const std::string& name, size_t bytes) {
	if (!enabled_) return;

	std::lock_guard<std::mutex> lock(mutex_);
	currentMemory_ = bytes;
	updatePeakMemory();

	auto& entry = entries_[name];
	entry.memoryBytes = std::max(entry.memoryBytes, bytes);
}

ofxStableDiffusionScopedTimer ofxStableDiffusionPerformanceProfiler::scopedTimer(const std::string& name) {
	std::lock_guard<std::mutex> lock(mutex_);
	if (entries_.find(name) == entries_.end()) {
		entries_[name].name = name;
		entries_[name].callCount = 0;
	}
	return ofxStableDiffusionScopedTimer(name, entries_[name]);
}

ofxStableDiffusionPerformanceStats ofxStableDiffusionPerformanceProfiler::getStats() const {
	std::lock_guard<std::mutex> lock(mutex_);

	ofxStableDiffusionPerformanceStats stats;
	stats.entries = entries_;
	stats.peakMemoryBytes = peakMemory_;
	stats.currentMemoryBytes = currentMemory_;

	for (const auto& pair : entries_) {
		stats.totalDurationMicros += pair.second.durationMicros;
	}

	return stats;
}

ofxStableDiffusionProfileEntry ofxStableDiffusionPerformanceProfiler::getEntry(const std::string& name) const {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = entries_.find(name);
	if (it != entries_.end()) {
		return it->second;
	}
	return ofxStableDiffusionProfileEntry();
}

void ofxStableDiffusionPerformanceProfiler::reset() {
	std::lock_guard<std::mutex> lock(mutex_);
	entries_.clear();
	activeTimers_.clear();
	peakMemory_ = 0;
	currentMemory_ = 0;
}

void ofxStableDiffusionPerformanceProfiler::setEnabled(bool enabled) {
	std::lock_guard<std::mutex> lock(mutex_);
	enabled_ = enabled;
}

bool ofxStableDiffusionPerformanceProfiler::isEnabled() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return enabled_;
}

std::string ofxStableDiffusionPerformanceProfiler::toJSON() const {
	std::lock_guard<std::mutex> lock(mutex_);

	std::ostringstream oss;
	oss << "{\n";
	oss << "  \"entries\": [\n";

	bool first = true;
	for (const auto& pair : entries_) {
		if (!first) oss << ",\n";
		first = false;

		const auto& entry = pair.second;
		oss << "    {\n";
		oss << "      \"name\": \"" << entry.name << "\",\n";
		oss << "      \"durationMs\": " << entry.durationMs() << ",\n";
		oss << "      \"callCount\": " << entry.callCount << ",\n";
		oss << "      \"memoryBytes\": " << entry.memoryBytes << "\n";
		oss << "    }";
	}

	oss << "\n  ],\n";
	oss << "  \"peakMemoryBytes\": " << peakMemory_ << ",\n";
	oss << "  \"currentMemoryBytes\": " << currentMemory_ << "\n";
	oss << "}";

	return oss.str();
}

std::string ofxStableDiffusionPerformanceProfiler::toCSV() const {
	std::lock_guard<std::mutex> lock(mutex_);

	std::ostringstream oss;
	oss << "name,durationMs,callCount,memoryBytes\n";

	for (const auto& pair : entries_) {
		const auto& entry = pair.second;
		oss << entry.name << ","
			<< entry.durationMs() << ","
			<< entry.callCount << ","
			<< entry.memoryBytes << "\n";
	}

	return oss.str();
}

void ofxStableDiffusionPerformanceProfiler::printSummary() const {
	std::lock_guard<std::mutex> lock(mutex_);

	ofLogNotice("ofxStableDiffusionPerformanceProfiler") << "=== Performance Summary ===";

	uint64_t totalDuration = 0;
	for (const auto& pair : entries_) {
		totalDuration += pair.second.durationMicros;
	}

	for (const auto& pair : entries_) {
		const auto& entry = pair.second;
		const float percent = totalDuration > 0 ? (entry.durationMicros * 100.0f / totalDuration) : 0.0f;

		ofLogNotice("ofxStableDiffusionPerformanceProfiler")
			<< entry.name << ": "
			<< entry.durationMs() << "ms (" << percent << "%) "
			<< "calls=" << entry.callCount << " "
			<< "mem=" << (entry.memoryBytes / 1024 / 1024) << "MB";
	}

	ofLogNotice("ofxStableDiffusionPerformanceProfiler")
		<< "Total: " << (totalDuration / 1000.0f) << "ms"
		<< " Peak Memory: " << (peakMemory_ / 1024 / 1024) << "MB";
}

std::vector<std::string> ofxStableDiffusionPerformanceProfiler::getBottlenecks(float thresholdPercent) const {
	std::lock_guard<std::mutex> lock(mutex_);

	std::vector<std::string> bottlenecks;

	uint64_t totalDuration = 0;
	for (const auto& pair : entries_) {
		totalDuration += pair.second.durationMicros;
	}

	if (totalDuration == 0) {
		return bottlenecks;
	}

	for (const auto& pair : entries_) {
		const float percent = (pair.second.durationMicros * 100.0f) / totalDuration;
		if (percent >= thresholdPercent) {
			bottlenecks.push_back(pair.first);
		}
	}

	return bottlenecks;
}

void ofxStableDiffusionPerformanceProfiler::updatePeakMemory() {
	if (currentMemory_ > peakMemory_) {
		peakMemory_ = currentMemory_;
	}
}
