#pragma once

#include <string>
#include <iostream>
#include <chrono>

namespace IntegrationTestUtils {

// Check if model file is available
inline bool hasModel() {
#ifdef SD_MODEL_PATH
    return true;
#else
    return false;
#endif
}

inline std::string getModelPath() {
#ifdef SD_MODEL_PATH
    return SD_MODEL_PATH;
#else
    return "";
#endif
}

// Skip test if no model available
inline bool skipIfNoModel() {
    if (!hasModel()) {
        std::cout << "SKIPPED: No model available (set SD_MODEL_PATH)" << std::endl;
        return true;
    }
    return false;
}

// Simple timer for benchmarking
class Timer {
public:
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    double elapsedSeconds() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end - startTime).count();
    }

    double elapsedMilliseconds() const {
        return elapsedSeconds() * 1000.0;
    }

private:
    std::chrono::high_resolution_clock::time_point startTime;
};

// Test result reporter
struct TestResult {
    std::string name;
    bool passed;
    double timeSeconds;
    std::string message;

    void print() const {
        std::cout << "[" << (passed ? "PASS" : "FAIL") << "] "
                  << name << " (" << timeSeconds << "s)";
        if (!message.empty()) {
            std::cout << " - " << message;
        }
        std::cout << std::endl;
    }
};

// Assertion helpers
inline void assertTrue(bool condition, const std::string& message = "") {
    if (!condition) {
        throw std::runtime_error("Assertion failed: " + message);
    }
}

inline void assertEqual(int actual, int expected, const std::string& message = "") {
    if (actual != expected) {
        throw std::runtime_error(
            "Assertion failed: expected " + std::to_string(expected) +
            " but got " + std::to_string(actual) +
            (message.empty() ? "" : " - " + message)
        );
    }
}

inline void assertNotNull(const void* ptr, const std::string& message = "") {
    if (ptr == nullptr) {
        throw std::runtime_error("Assertion failed: pointer is null - " + message);
    }
}

} // namespace IntegrationTestUtils
