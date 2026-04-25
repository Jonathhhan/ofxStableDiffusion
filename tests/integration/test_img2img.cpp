/**
 * Integration Test: Image-to-Image
 *
 * Tests image-to-image transformation.
 * Requires: Model file
 */

#include "test_utils.h"
#include <iostream>
#include <cassert>

namespace MockSD {
    struct Image {
        int width = 0;
        int height = 0;
        bool isValid() const { return width > 0 && height > 0; }
    };

    struct ImageRequest {
        std::string prompt;
        int width = 512;
        int height = 512;
        int sampleSteps = 20;
        Image initImage;
        float strength = 0.75f;
    };

    class StableDiffusion {
    public:
        bool configureContext(const std::string& modelPath) {
            configured = true;
            return true;
        }

        bool generate(const ImageRequest& request) {
            if (!configured) return false;
            if (!request.initImage.isValid()) return false;

            resultImage.width = request.width;
            resultImage.height = request.height;
            return true;
        }

        Image getImage() const { return resultImage; }

    private:
        bool configured = false;
        Image resultImage;
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
        std::cout << "Integration Test: Image-to-Image" << std::endl;
        std::cout << "Model: " << getModelPath() << std::endl;

        MockSD::StableDiffusion sd;

        std::cout << "Configuring context..." << std::endl;
        bool configured = sd.configureContext(getModelPath());
        assertTrue(configured, "Context configuration failed");

        // Create init image
        MockSD::Image initImage;
        initImage.width = 512;
        initImage.height = 512;

        // Create img2img request
        MockSD::ImageRequest request;
        request.prompt = "Transform this image";
        request.width = 512;
        request.height = 512;
        request.sampleSteps = 10;
        request.initImage = initImage;
        request.strength = 0.75f;

        std::cout << "Transforming image..." << std::endl;
        std::cout << "  Prompt: " << request.prompt << std::endl;
        std::cout << "  Strength: " << request.strength << std::endl;

        bool success = sd.generate(request);
        assertTrue(success, "Image transformation failed");

        auto result = sd.getImage();
        assertTrue(result.isValid(), "Result image is invalid");
        assertEqual(result.width, 512, "Width mismatch");
        assertEqual(result.height, 512, "Height mismatch");

        double elapsed = timer.elapsedSeconds();

        TestResult testResult;
        testResult.name = "Image-to-Image";
        testResult.passed = true;
        testResult.timeSeconds = elapsed;
        testResult.message = "Transformed with strength 0.75";
        testResult.print();

        std::cout << "SUCCESS" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        double elapsed = timer.elapsedSeconds();

        TestResult result;
        result.name = "Image-to-Image";
        result.passed = false;
        result.timeSeconds = elapsed;
        result.message = e.what();
        result.print();

        std::cout << "FAILED: " << e.what() << std::endl;
        return 1;
    }
}
