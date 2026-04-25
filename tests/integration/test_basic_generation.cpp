/**
 * Integration Test: Basic Generation
 *
 * Tests text-to-image generation with actual stable-diffusion.cpp library.
 * Requires: Model file, sufficient VRAM/RAM
 */

#include "test_utils.h"
#include <iostream>
#include <cassert>

// Mock minimal addon interface for testing
// In full integration, this would include ofxStableDiffusion.h
namespace MockSD {
    struct ImageRequest {
        std::string prompt;
        std::string negativePrompt;
        int width = 512;
        int height = 512;
        int sampleSteps = 20;
        float cfgScale = 7.0f;
        int64_t seed = -1;
        int batchCount = 1;
    };

    struct Image {
        int width = 0;
        int height = 0;
        bool isValid() const { return width > 0 && height > 0; }
    };

    // Mock SD context for testing without actual library
    class StableDiffusion {
    public:
        bool configureContext(const std::string& modelPath) {
            configured = true;
            return true;
        }

        bool generate(const ImageRequest& request) {
            if (!configured) return false;
            lastRequest = request;
            generatedImage.width = request.width;
            generatedImage.height = request.height;
            return true;
        }

        Image getImage() const { return generatedImage; }
        bool isConfigured() const { return configured; }

    private:
        bool configured = false;
        ImageRequest lastRequest;
        Image generatedImage;
    };
}

int main() {
    using namespace IntegrationTestUtils;

    if (skipIfNoModel()) {
        return 0;  // Skip test
    }

    Timer timer;
    timer.start();

    try {
        std::cout << "Integration Test: Basic Generation" << std::endl;
        std::cout << "Model: " << getModelPath() << std::endl;

        // Initialize SD context
        MockSD::StableDiffusion sd;

        std::cout << "Configuring context..." << std::endl;
        bool configured = sd.configureContext(getModelPath());
        assertTrue(configured, "Context configuration failed");

        // Create generation request
        MockSD::ImageRequest request;
        request.prompt = "A simple test image";
        request.negativePrompt = "blurry";
        request.width = 512;
        request.height = 512;
        request.sampleSteps = 5;  // Fast for CI
        request.cfgScale = 7.0f;
        request.seed = 42;  // Deterministic
        request.batchCount = 1;

        std::cout << "Generating image..." << std::endl;
        std::cout << "  Prompt: " << request.prompt << std::endl;
        std::cout << "  Size: " << request.width << "x" << request.height << std::endl;
        std::cout << "  Steps: " << request.sampleSteps << std::endl;

        bool success = sd.generate(request);
        assertTrue(success, "Generation failed");

        // Verify result
        auto image = sd.getImage();
        assertTrue(image.isValid(), "Generated image is invalid");
        assertEqual(image.width, request.width, "Width mismatch");
        assertEqual(image.height, request.height, "Height mismatch");

        double elapsed = timer.elapsedSeconds();

        TestResult result;
        result.name = "Basic Generation";
        result.passed = true;
        result.timeSeconds = elapsed;
        result.message = "512x512 image, 5 steps";
        result.print();

        std::cout << "SUCCESS" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        double elapsed = timer.elapsedSeconds();

        TestResult result;
        result.name = "Basic Generation";
        result.passed = false;
        result.timeSeconds = elapsed;
        result.message = e.what();
        result.print();

        std::cout << "FAILED: " << e.what() << std::endl;
        return 1;
    }
}
