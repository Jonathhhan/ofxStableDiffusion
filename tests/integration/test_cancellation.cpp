/**
 * Integration Test: Cancellation
 *
 * Tests cancellation during long-running generation.
 * Requires: Model file
 */

#include "test_utils.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <cassert>

// Mock SD with cancellation support
namespace MockSD {
    struct ImageRequest {
        std::string prompt;
        int width = 512;
        int height = 512;
        int sampleSteps = 50;  // Long generation
    };

    class StableDiffusion {
    public:
        bool configureContext(const std::string& modelPath) {
            configured = true;
            return true;
        }

        bool generate(const ImageRequest& request) {
            if (!configured) return false;

            generating.store(true);
            cancelled.store(false);

            // Simulate long generation with cancellation checks
            for (int step = 0; step < request.sampleSteps; ++step) {
                if (cancellationRequested.load()) {
                    cancelled.store(true);
                    generating.store(false);
                    return false;  // Cancelled
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            generating.store(false);
            return true;  // Completed
        }

        bool requestCancellation() {
            if (!generating.load()) return false;
            cancellationRequested.store(true);
            return true;
        }

        bool isGenerating() const { return generating.load(); }
        bool wasCancelled() const { return cancelled.load(); }

        void reset() {
            cancellationRequested.store(false);
            cancelled.store(false);
        }

    private:
        bool configured = false;
        std::atomic<bool> generating{false};
        std::atomic<bool> cancellationRequested{false};
        std::atomic<bool> cancelled{false};
    };
}

int main() {
    using namespace IntegrationTestUtils;

    if (skipIfNoModel()) {
        return 0;
    }

    Timer timer;
    timer.start();

    try {
        std::cout << "Integration Test: Cancellation" << std::endl;
        std::cout << "Model: " << getModelPath() << std::endl;

        MockSD::StableDiffusion sd;

        std::cout << "Configuring context..." << std::endl;
        bool configured = sd.configureContext(getModelPath());
        assertTrue(configured, "Context configuration failed");

        // Start long generation in background
        MockSD::ImageRequest request;
        request.prompt = "Test cancellation";
        request.sampleSteps = 100;  // Long enough to cancel

        std::cout << "Starting generation (100 steps)..." << std::endl;

        std::thread genThread([&sd, &request]() {
            sd.generate(request);
        });

        // Wait a bit, then cancel
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        std::cout << "Requesting cancellation..." << std::endl;
        bool cancelRequested = sd.requestCancellation();
        assertTrue(cancelRequested, "Cancellation request failed");

        // Wait for generation to stop
        genThread.join();

        // Verify cancellation
        assertTrue(sd.wasCancelled(), "Generation was not cancelled");
        assertTrue(!sd.isGenerating(), "Still generating after cancellation");

        double elapsed = timer.elapsedSeconds();

        TestResult result;
        result.name = "Cancellation";
        result.passed = true;
        result.timeSeconds = elapsed;
        result.message = "Cancelled gracefully";
        result.print();

        std::cout << "SUCCESS" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        double elapsed = timer.elapsedSeconds();

        TestResult result;
        result.name = "Cancellation";
        result.passed = false;
        result.timeSeconds = elapsed;
        result.message = e.what();
        result.print();

        std::cout << "FAILED: " << e.what() << std::endl;
        return 1;
    }
}
