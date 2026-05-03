#pragma once

#include "core/ofxStableDiffusionTypes.h"

#include <deque>
#include <vector>

class ofxStableDiffusion {
public:
	void setQueuedResults(const std::vector<ofxStableDiffusionResult>& results) {
		queuedResults_.assign(results.begin(), results.end());
	}

	void generate(const ofxStableDiffusionImageRequest& request) {
		imageRequests_.push_back(request);
		if (!queuedResults_.empty()) {
			lastResult_ = queuedResults_.front();
			queuedResults_.pop_front();
		} else {
			lastResult_ = ofxStableDiffusionResult();
			lastResult_.success = true;
		}
	}

	bool isGenerating() const {
		return false;
	}

	ofxStableDiffusionResult getLastResult() const {
		return lastResult_;
	}

	bool requestCancellation() {
		cancelRequested_ = true;
		return true;
	}

	bool wasCancellationRequested() const {
		return cancelRequested_;
	}

	const std::vector<ofxStableDiffusionImageRequest>& getImageRequests() const {
		return imageRequests_;
	}

private:
	std::deque<ofxStableDiffusionResult> queuedResults_;
	std::vector<ofxStableDiffusionImageRequest> imageRequests_;
	ofxStableDiffusionResult lastResult_;
	bool cancelRequested_ = false;
};
