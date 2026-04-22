#pragma once

#include "../ofxStableDiffusion.h"

#include <chrono>
#include <string>
#include <thread>

namespace ofxStableDiffusionGgmlBridge {

struct WaitOptions {
	std::chrono::milliseconds pollInterval{15};
	std::chrono::seconds timeout{300};
};

inline bool waitForIdle(
	const ofxStableDiffusion& engine,
	const WaitOptions& options = {},
	std::string* error = nullptr) {
	const auto started = std::chrono::steady_clock::now();
	while (engine.isBusy()) {
		if (std::chrono::steady_clock::now() - started > options.timeout) {
			if (error) {
				*error = "timed out while waiting for ofxStableDiffusion";
			}
			return false;
		}
		std::this_thread::sleep_for(options.pollInterval);
	}
	return true;
}

inline bool needsContextReload(
	const ofxStableDiffusion& engine,
	const ofxStableDiffusionContextSettings& settings) {
	return !engine.hasLoadedContext() || !engine.matchesContextSettings(settings);
}

inline bool supportsVideoWithoutInputImage(const ofxStableDiffusion& engine) {
	const auto capabilities = engine.getCapabilities();
	return capabilities.imageToVideo && !capabilities.videoRequiresInputImage;
}

}  // namespace ofxStableDiffusionGgmlBridge
