#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionTypes.h"
#include "ofxStableDiffusionEnums.h"

#include <queue>
#include <deque>
#include <functional>
#include <memory>

/// Priority levels for generation requests
enum class ofxStableDiffusionPriority {
	Low = 0,
	Normal = 1,
	High = 2,
	Critical = 3
};

/// Queue request state
enum class ofxStableDiffusionQueueState {
	Queued = 0,
	Processing,
	Completed,
	Failed,
	Cancelled
};

/// Queue request entry
struct ofxStableDiffusionQueueRequest {
	int requestId = -1;
	ofxStableDiffusionPriority priority = ofxStableDiffusionPriority::Normal;
	ofxStableDiffusionQueueState state = ofxStableDiffusionQueueState::Queued;
	ofxStableDiffusionTask taskType = ofxStableDiffusionTask::None;

	// Request data (one will be set based on taskType)
	ofxStableDiffusionImageRequest imageRequest;
	ofxStableDiffusionVideoRequest videoRequest;
	ofxStableDiffusionContextSettings contextSettings;

	// Result data
	ofxStableDiffusionResult result;

	// Timing
	uint64_t queuedTimeMicros = 0;
	uint64_t startedTimeMicros = 0;
	uint64_t completedTimeMicros = 0;

	// Callbacks
	std::function<void(const ofxStableDiffusionResult&)> onComplete;
	std::function<void(const std::string&)> onError;
	std::function<void(int step, int steps, float time)> onProgress;

	// User data
	std::string tag;
	std::map<std::string, std::string> metadata;

	bool isImageGeneration() const {
		return taskType == ofxStableDiffusionTask::TextToImage ||
			   taskType == ofxStableDiffusionTask::ImageToImage ||
			   taskType == ofxStableDiffusionTask::InstructImage ||
			   taskType == ofxStableDiffusionTask::ImageVariation ||
			   taskType == ofxStableDiffusionTask::ImageRestyle;
	}

	bool isVideoGeneration() const {
		return taskType == ofxStableDiffusionTask::ImageToVideo;
	}

	bool isModelLoad() const {
		return taskType == ofxStableDiffusionTask::LoadModel;
	}

	float getWaitTimeSeconds() const {
		if (startedTimeMicros == 0) {
			return (ofGetElapsedTimeMicros() - queuedTimeMicros) / 1000000.0f;
		}
		return (startedTimeMicros - queuedTimeMicros) / 1000000.0f;
	}

	float getProcessingTimeSeconds() const {
		if (startedTimeMicros == 0) {
			return 0.0f;
		}
		uint64_t endTime = completedTimeMicros > 0 ? completedTimeMicros : ofGetElapsedTimeMicros();
		return (endTime - startedTimeMicros) / 1000000.0f;
	}
};

/// Request comparator for priority queue
struct ofxStableDiffusionRequestComparator {
	bool operator()(const std::shared_ptr<ofxStableDiffusionQueueRequest>& a,
					const std::shared_ptr<ofxStableDiffusionQueueRequest>& b) const {
		// Higher priority first, then earlier queued time
		if (a->priority != b->priority) {
			return static_cast<int>(a->priority) < static_cast<int>(b->priority);
		}
		return a->queuedTimeMicros > b->queuedTimeMicros;
	}
};

/// Generation Queue Manager
class ofxStableDiffusionQueue {
public:
	ofxStableDiffusionQueue();
	~ofxStableDiffusionQueue();

	/// Add an image generation request to the queue
	int addImageRequest(const ofxStableDiffusionImageRequest& request,
						ofxStableDiffusionPriority priority = ofxStableDiffusionPriority::Normal,
						const std::string& tag = "");

	/// Add a video generation request to the queue
	int addVideoRequest(const ofxStableDiffusionVideoRequest& request,
						ofxStableDiffusionPriority priority = ofxStableDiffusionPriority::Normal,
						const std::string& tag = "");

	/// Add a model load request to the queue
	int addModelLoadRequest(const ofxStableDiffusionContextSettings& settings,
							ofxStableDiffusionPriority priority = ofxStableDiffusionPriority::High,
							const std::string& tag = "");

	/// Set callback for request completion
	void setCompletionCallback(int requestId, std::function<void(const ofxStableDiffusionResult&)> callback);

	/// Set callback for request error
	void setErrorCallback(int requestId, std::function<void(const std::string&)> callback);

	/// Set callback for request progress
	void setProgressCallback(int requestId, std::function<void(int, int, float)> callback);

	/// Cancel a specific request
	bool cancelRequest(int requestId);

	/// Cancel all requests with a specific tag
	int cancelRequestsByTag(const std::string& tag);

	/// Cancel all queued requests
	void cancelAll();

	/// Clear completed/failed requests from history
	void clearHistory();

	/// Get the next request to process (highest priority)
	std::shared_ptr<ofxStableDiffusionQueueRequest> getNextRequest();

	/// Mark a request as processing
	void markRequestProcessing(int requestId);

	/// Mark a request as completed
	void markRequestCompleted(int requestId, const ofxStableDiffusionResult& result);

	/// Mark a request as failed
	void markRequestFailed(int requestId, const std::string& errorMessage);

	/// Get request by ID
	std::shared_ptr<ofxStableDiffusionQueueRequest> getRequest(int requestId);

	/// Get all requests with a specific state
	std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>> getRequestsByState(ofxStableDiffusionQueueState state);

	/// Get all requests with a specific tag
	std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>> getRequestsByTag(const std::string& tag);

	/// Get queue statistics
	struct QueueStats {
		int totalRequests = 0;
		int queuedRequests = 0;
		int processingRequests = 0;
		int completedRequests = 0;
		int failedRequests = 0;
		int cancelledRequests = 0;
		float avgWaitTimeSeconds = 0.0f;
		float avgProcessingTimeSeconds = 0.0f;
	};
	QueueStats getStats() const;

	/// Get queue size (number of queued requests)
	int getQueueSize() const;

	/// Check if queue is empty
	bool isEmpty() const;

	/// Check if queue is processing
	bool isProcessing() const;

	/// Get current processing request
	std::shared_ptr<ofxStableDiffusionQueueRequest> getCurrentRequest() const;

	/// Enable/disable queue processing
	void setEnabled(bool enabled);

	/// Check if queue is enabled
	bool isEnabled() const;

	/// Set maximum queue size (0 = unlimited)
	void setMaxQueueSize(int size);

	/// Get maximum queue size
	int getMaxQueueSize() const;

	/// Save queue state to file
	bool saveToFile(const std::string& filepath);

	/// Load queue state from file
	bool loadFromFile(const std::string& filepath);

	/// Enable/disable auto-save on changes
	void setAutoSave(bool enabled, const std::string& filepath = "");

private:
	std::priority_queue<std::shared_ptr<ofxStableDiffusionQueueRequest>,
						std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>>,
						ofxStableDiffusionRequestComparator> requestQueue;

	std::map<int, std::shared_ptr<ofxStableDiffusionQueueRequest>> allRequests;
	std::shared_ptr<ofxStableDiffusionQueueRequest> currentRequest;

	int nextRequestId = 1;
	int queuedCount = 0;
	bool enabled = true;
	int maxQueueSize = 0;  // 0 = unlimited

	// Auto-save
	bool autoSaveEnabled = false;
	std::string autoSaveFilepath;

	// Internal methods
	int generateRequestId();
	void triggerAutoSave();
	std::shared_ptr<ofxStableDiffusionQueueRequest> createRequest(ofxStableDiffusionTask taskType,
		ofxStableDiffusionPriority priority,
		const std::string& tag);
};
