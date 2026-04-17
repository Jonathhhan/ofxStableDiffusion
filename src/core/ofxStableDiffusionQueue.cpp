#include "ofxStableDiffusionQueue.h"
#include <algorithm>

//--------------------------------------------------------------
ofxStableDiffusionQueue::ofxStableDiffusionQueue() {
}

//--------------------------------------------------------------
ofxStableDiffusionQueue::~ofxStableDiffusionQueue() {
	if (autoSaveEnabled && !autoSaveFilepath.empty()) {
		saveToFile(autoSaveFilepath);
	}
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::addImageRequest(const ofxStableDiffusionImageRequest& request,
											  ofxStableDiffusionPriority priority,
											  const std::string& tag) {
	if (!enabled) {
		ofLogWarning("ofxStableDiffusionQueue") << "Queue is disabled";
		return -1;
	}

	if (maxQueueSize > 0 && getQueueSize() >= maxQueueSize) {
		ofLogWarning("ofxStableDiffusionQueue") << "Queue is full (max: " << maxQueueSize << ")";
		return -1;
	}

	auto queueRequest = createRequest(ofxStableDiffusionTaskForImageMode(request.mode), priority, tag);
	queueRequest->imageRequest = request;

	requestQueue.push(queueRequest);
	allRequests[queueRequest->requestId] = queueRequest;

	ofLogNotice("ofxStableDiffusionQueue") << "Added image request #" << queueRequest->requestId
		<< " (priority: " << static_cast<int>(priority) << ", queue size: " << getQueueSize() << ")";

	triggerAutoSave();
	return queueRequest->requestId;
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::addVideoRequest(const ofxStableDiffusionVideoRequest& request,
											  ofxStableDiffusionPriority priority,
											  const std::string& tag) {
	if (!enabled) {
		ofLogWarning("ofxStableDiffusionQueue") << "Queue is disabled";
		return -1;
	}

	if (maxQueueSize > 0 && getQueueSize() >= maxQueueSize) {
		ofLogWarning("ofxStableDiffusionQueue") << "Queue is full (max: " << maxQueueSize << ")";
		return -1;
	}

	auto queueRequest = createRequest(ofxStableDiffusionTask::ImageToVideo, priority, tag);
	queueRequest->videoRequest = request;

	requestQueue.push(queueRequest);
	allRequests[queueRequest->requestId] = queueRequest;

	ofLogNotice("ofxStableDiffusionQueue") << "Added video request #" << queueRequest->requestId
		<< " (priority: " << static_cast<int>(priority) << ", queue size: " << getQueueSize() << ")";

	triggerAutoSave();
	return queueRequest->requestId;
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::addModelLoadRequest(const ofxStableDiffusionContextSettings& settings,
												  ofxStableDiffusionPriority priority,
												  const std::string& tag) {
	if (!enabled) {
		ofLogWarning("ofxStableDiffusionQueue") << "Queue is disabled";
		return -1;
	}

	if (maxQueueSize > 0 && getQueueSize() >= maxQueueSize) {
		ofLogWarning("ofxStableDiffusionQueue") << "Queue is full (max: " << maxQueueSize << ")";
		return -1;
	}

	auto queueRequest = createRequest(ofxStableDiffusionTask::LoadModel, priority, tag);
	queueRequest->contextSettings = settings;

	requestQueue.push(queueRequest);
	allRequests[queueRequest->requestId] = queueRequest;

	ofLogNotice("ofxStableDiffusionQueue") << "Added model load request #" << queueRequest->requestId
		<< " (priority: " << static_cast<int>(priority) << ", queue size: " << getQueueSize() << ")";

	triggerAutoSave();
	return queueRequest->requestId;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setCompletionCallback(int requestId, std::function<void(const ofxStableDiffusionResult&)> callback) {
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		it->second->onComplete = callback;
	}
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setErrorCallback(int requestId, std::function<void(const std::string&)> callback) {
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		it->second->onError = callback;
	}
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setProgressCallback(int requestId, std::function<void(int, int, float)> callback) {
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		it->second->onProgress = callback;
	}
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::cancelRequest(int requestId) {
	auto it = allRequests.find(requestId);
	if (it == allRequests.end()) {
		return false;
	}

	auto request = it->second;

	// Can only cancel queued requests
	if (request->state != ofxStableDiffusionQueueState::Queued) {
		ofLogWarning("ofxStableDiffusionQueue") << "Cannot cancel request #" << requestId
			<< " (state: " << static_cast<int>(request->state) << ")";
		return false;
	}

	request->state = ofxStableDiffusionQueueState::Cancelled;
	request->completedTimeMicros = ofGetElapsedTimeMicros();

	ofLogNotice("ofxStableDiffusionQueue") << "Cancelled request #" << requestId;

	triggerAutoSave();
	return true;
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::cancelRequestsByTag(const std::string& tag) {
	int cancelledCount = 0;

	for (auto& pair : allRequests) {
		if (pair.second->tag == tag && pair.second->state == ofxStableDiffusionQueueState::Queued) {
			pair.second->state = ofxStableDiffusionQueueState::Cancelled;
			pair.second->completedTimeMicros = ofGetElapsedTimeMicros();
			cancelledCount++;
		}
	}

	if (cancelledCount > 0) {
		ofLogNotice("ofxStableDiffusionQueue") << "Cancelled " << cancelledCount << " requests with tag: " << tag;
		triggerAutoSave();
	}

	return cancelledCount;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::cancelAll() {
	int cancelledCount = 0;

	for (auto& pair : allRequests) {
		if (pair.second->state == ofxStableDiffusionQueueState::Queued) {
			pair.second->state = ofxStableDiffusionQueueState::Cancelled;
			pair.second->completedTimeMicros = ofGetElapsedTimeMicros();
			cancelledCount++;
		}
	}

	if (cancelledCount > 0) {
		ofLogNotice("ofxStableDiffusionQueue") << "Cancelled all " << cancelledCount << " queued requests";
		triggerAutoSave();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::clearHistory() {
	auto it = allRequests.begin();
	while (it != allRequests.end()) {
		if (it->second->state == ofxStableDiffusionQueueState::Completed ||
			it->second->state == ofxStableDiffusionQueueState::Failed ||
			it->second->state == ofxStableDiffusionQueueState::Cancelled) {
			it = allRequests.erase(it);
		} else {
			++it;
		}
	}

	ofLogNotice("ofxStableDiffusionQueue") << "Cleared completed/failed/cancelled requests";
	triggerAutoSave();
}

//--------------------------------------------------------------
std::shared_ptr<ofxStableDiffusionQueueRequest> ofxStableDiffusionQueue::getNextRequest() {
	// Skip cancelled requests
	while (!requestQueue.empty()) {
		auto request = requestQueue.top();
		requestQueue.pop();

		if (request->state == ofxStableDiffusionQueueState::Queued) {
			return request;
		}
	}

	return nullptr;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::markRequestProcessing(int requestId) {
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		it->second->state = ofxStableDiffusionQueueState::Processing;
		it->second->startedTimeMicros = ofGetElapsedTimeMicros();
		currentRequest = it->second;

		ofLogNotice("ofxStableDiffusionQueue") << "Processing request #" << requestId;
		triggerAutoSave();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::markRequestCompleted(int requestId, const ofxStableDiffusionResult& result) {
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		it->second->state = ofxStableDiffusionQueueState::Completed;
		it->second->completedTimeMicros = ofGetElapsedTimeMicros();
		it->second->result = result;

		if (it->second->onComplete) {
			it->second->onComplete(result);
		}

		if (currentRequest && currentRequest->requestId == requestId) {
			currentRequest = nullptr;
		}

		ofLogNotice("ofxStableDiffusionQueue") << "Completed request #" << requestId
			<< " (processing time: " << it->second->getProcessingTimeSeconds() << "s)";

		triggerAutoSave();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::markRequestFailed(int requestId, const std::string& errorMessage) {
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		it->second->state = ofxStableDiffusionQueueState::Failed;
		it->second->completedTimeMicros = ofGetElapsedTimeMicros();
		it->second->result.success = false;
		it->second->result.error = errorMessage;

		if (it->second->onError) {
			it->second->onError(errorMessage);
		}

		if (currentRequest && currentRequest->requestId == requestId) {
			currentRequest = nullptr;
		}

		ofLogError("ofxStableDiffusionQueue") << "Failed request #" << requestId << ": " << errorMessage;

		triggerAutoSave();
	}
}

//--------------------------------------------------------------
std::shared_ptr<ofxStableDiffusionQueueRequest> ofxStableDiffusionQueue::getRequest(int requestId) {
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		return it->second;
	}
	return nullptr;
}

//--------------------------------------------------------------
std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>> ofxStableDiffusionQueue::getRequestsByState(ofxStableDiffusionQueueState state) {
	std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>> requests;

	for (const auto& pair : allRequests) {
		if (pair.second->state == state) {
			requests.push_back(pair.second);
		}
	}

	return requests;
}

//--------------------------------------------------------------
std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>> ofxStableDiffusionQueue::getRequestsByTag(const std::string& tag) {
	std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>> requests;

	for (const auto& pair : allRequests) {
		if (pair.second->tag == tag) {
			requests.push_back(pair.second);
		}
	}

	return requests;
}

//--------------------------------------------------------------
ofxStableDiffusionQueue::QueueStats ofxStableDiffusionQueue::getStats() const {
	QueueStats stats;
	stats.totalRequests = static_cast<int>(allRequests.size());

	float totalWaitTime = 0.0f;
	float totalProcessingTime = 0.0f;
	int completedCount = 0;

	for (const auto& pair : allRequests) {
		switch (pair.second->state) {
			case ofxStableDiffusionQueueState::Queued:
				stats.queuedRequests++;
				break;
			case ofxStableDiffusionQueueState::Processing:
				stats.processingRequests++;
				break;
			case ofxStableDiffusionQueueState::Completed:
				stats.completedRequests++;
				totalWaitTime += pair.second->getWaitTimeSeconds();
				totalProcessingTime += pair.second->getProcessingTimeSeconds();
				completedCount++;
				break;
			case ofxStableDiffusionQueueState::Failed:
				stats.failedRequests++;
				break;
			case ofxStableDiffusionQueueState::Cancelled:
				stats.cancelledRequests++;
				break;
		}
	}

	if (completedCount > 0) {
		stats.avgWaitTimeSeconds = totalWaitTime / completedCount;
		stats.avgProcessingTimeSeconds = totalProcessingTime / completedCount;
	}

	return stats;
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::getQueueSize() const {
	int count = 0;
	for (const auto& pair : allRequests) {
		if (pair.second->state == ofxStableDiffusionQueueState::Queued) {
			count++;
		}
	}
	return count;
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::isEmpty() const {
	return getQueueSize() == 0;
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::isProcessing() const {
	return currentRequest != nullptr;
}

//--------------------------------------------------------------
std::shared_ptr<ofxStableDiffusionQueueRequest> ofxStableDiffusionQueue::getCurrentRequest() const {
	return currentRequest;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setEnabled(bool enabled_) {
	enabled = enabled_;
	ofLogNotice("ofxStableDiffusionQueue") << "Queue " << (enabled ? "enabled" : "disabled");
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::isEnabled() const {
	return enabled;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setMaxQueueSize(int size) {
	maxQueueSize = size;
	ofLogNotice("ofxStableDiffusionQueue") << "Max queue size set to: " << (size == 0 ? "unlimited" : std::to_string(size));
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::getMaxQueueSize() const {
	return maxQueueSize;
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::saveToFile(const std::string& filepath) {
	ofJson json;
	json["version"] = "1.0";
	json["timestamp"] = ofGetElapsedTimeMicros();
	json["nextRequestId"] = nextRequestId;

	ofJson requestsJson = ofJson::array();
	for (const auto& pair : allRequests) {
		ofJson reqJson;
		reqJson["requestId"] = pair.second->requestId;
		reqJson["priority"] = static_cast<int>(pair.second->priority);
		reqJson["state"] = static_cast<int>(pair.second->state);
		reqJson["taskType"] = static_cast<int>(pair.second->taskType);
		reqJson["tag"] = pair.second->tag;
		reqJson["queuedTime"] = pair.second->queuedTimeMicros;
		reqJson["startedTime"] = pair.second->startedTimeMicros;
		reqJson["completedTime"] = pair.second->completedTimeMicros;

		// Note: We don't serialize the actual request data or callbacks
		// This is primarily for statistics and queue state recovery

		requestsJson.push_back(reqJson);
	}
	json["requests"] = requestsJson;

	return ofSaveJson(filepath, json);
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::loadFromFile(const std::string& filepath) {
	ofJson json = ofLoadJson(filepath);
	if (json.empty()) {
		ofLogError("ofxStableDiffusionQueue") << "Failed to load queue from: " << filepath;
		return false;
	}

	// Note: This is a basic implementation that loads queue state metadata
	// Full request restoration would require serializing/deserializing the request data
	nextRequestId = json.value("nextRequestId", 1);

	ofLogNotice("ofxStableDiffusionQueue") << "Loaded queue state from: " << filepath;
	return true;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setAutoSave(bool enabled_, const std::string& filepath) {
	autoSaveEnabled = enabled_;
	autoSaveFilepath = filepath;

	if (autoSaveEnabled && !autoSaveFilepath.empty()) {
		ofLogNotice("ofxStableDiffusionQueue") << "Auto-save enabled: " << autoSaveFilepath;
	}
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::generateRequestId() {
	return nextRequestId++;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::triggerAutoSave() {
	if (autoSaveEnabled && !autoSaveFilepath.empty()) {
		saveToFile(autoSaveFilepath);
	}
}

//--------------------------------------------------------------
std::shared_ptr<ofxStableDiffusionQueueRequest> ofxStableDiffusionQueue::createRequest(
	ofxStableDiffusionTask taskType,
	ofxStableDiffusionPriority priority,
	const std::string& tag) {

	auto request = std::make_shared<ofxStableDiffusionQueueRequest>();
	request->requestId = generateRequestId();
	request->priority = priority;
	request->state = ofxStableDiffusionQueueState::Queued;
	request->taskType = taskType;
	request->tag = tag;
	request->queuedTimeMicros = ofGetElapsedTimeMicros();

	return request;
}
