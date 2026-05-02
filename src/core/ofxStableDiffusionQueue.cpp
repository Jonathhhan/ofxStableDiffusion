#include "ofxStableDiffusionQueue.h"
#include <algorithm>

//--------------------------------------------------------------
ofxStableDiffusionQueue::ofxStableDiffusionQueue() {}

//--------------------------------------------------------------
ofxStableDiffusionQueue::~ofxStableDiffusionQueue() {
	std::string filepath;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (autoSaveEnabled) {
			filepath = autoSaveFilepath;
		}
	}
	if (!filepath.empty()) {
		saveToFile(filepath);
	}
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::addImageRequest(const ofxStableDiffusionImageRequest &request,
											 ofxStableDiffusionPriority priority, const std::string &tag) {
	int requestId = -1;
	int queueSize = 0;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (!enabled) {
			ofLogWarning("ofxStableDiffusionQueue") << "Queue is disabled";
			return -1;
		}

		if (maxQueueSize > 0 && queuedCount >= maxQueueSize) {
			ofLogWarning("ofxStableDiffusionQueue") << "Queue is full (max: " << maxQueueSize << ")";
			return -1;
		}

		auto queueRequest = createRequest(ofxStableDiffusionTaskForImageMode(request.mode), priority, tag);
		queueRequest->imageRequest = request;

		requestQueue.push(queueRequest);
		allRequests[queueRequest->requestId] = queueRequest;
		queuedCount++;
		requestId = queueRequest->requestId;
		queueSize = queuedCount;
	}

	ofLogNotice("ofxStableDiffusionQueue")
		<< "Added image request #" << requestId << " (priority: " << static_cast<int>(priority)
		<< ", queue size: " << queueSize << ")";

	triggerAutoSave();
	return requestId;
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::addVideoRequest(const ofxStableDiffusionVideoRequest &request,
											 ofxStableDiffusionPriority priority, const std::string &tag) {
	int requestId = -1;
	int queueSize = 0;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (!enabled) {
			ofLogWarning("ofxStableDiffusionQueue") << "Queue is disabled";
			return -1;
		}

		if (maxQueueSize > 0 && queuedCount >= maxQueueSize) {
			ofLogWarning("ofxStableDiffusionQueue") << "Queue is full (max: " << maxQueueSize << ")";
			return -1;
		}

		auto queueRequest = createRequest(ofxStableDiffusionTask::ImageToVideo, priority, tag);
		queueRequest->videoRequest = request;

		requestQueue.push(queueRequest);
		allRequests[queueRequest->requestId] = queueRequest;
		queuedCount++;
		requestId = queueRequest->requestId;
		queueSize = queuedCount;
	}

	ofLogNotice("ofxStableDiffusionQueue")
		<< "Added video request #" << requestId << " (priority: " << static_cast<int>(priority)
		<< ", queue size: " << queueSize << ")";

	triggerAutoSave();
	return requestId;
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::addModelLoadRequest(const ofxStableDiffusionContextSettings &settings,
												 ofxStableDiffusionPriority priority, const std::string &tag) {
	int requestId = -1;
	int queueSize = 0;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		if (!enabled) {
			ofLogWarning("ofxStableDiffusionQueue") << "Queue is disabled";
			return -1;
		}

		if (maxQueueSize > 0 && queuedCount >= maxQueueSize) {
			ofLogWarning("ofxStableDiffusionQueue") << "Queue is full (max: " << maxQueueSize << ")";
			return -1;
		}

		auto queueRequest = createRequest(ofxStableDiffusionTask::LoadModel, priority, tag);
		queueRequest->contextSettings = settings;

		requestQueue.push(queueRequest);
		allRequests[queueRequest->requestId] = queueRequest;
		queuedCount++;
		requestId = queueRequest->requestId;
		queueSize = queuedCount;
	}

	ofLogNotice("ofxStableDiffusionQueue")
		<< "Added model load request #" << requestId << " (priority: " << static_cast<int>(priority)
		<< ", queue size: " << queueSize << ")";

	triggerAutoSave();
	return requestId;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setCompletionCallback(int requestId,
													std::function<void(const ofxStableDiffusionResult &)> callback) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		it->second->onComplete = callback;
	}
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setErrorCallback(int requestId, std::function<void(const std::string &)> callback) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		it->second->onError = callback;
	}
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setProgressCallback(int requestId, std::function<void(int, int, float)> callback) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		it->second->onProgress = callback;
	}
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::cancelRequest(int requestId) {
	{
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = allRequests.find(requestId);
		if (it == allRequests.end()) {
			return false;
		}

		auto request = it->second;
		if (request->state != ofxStableDiffusionQueueState::Queued) {
			ofLogWarning("ofxStableDiffusionQueue")
				<< "Cannot cancel request #" << requestId << " (state: " << static_cast<int>(request->state) << ")";
			return false;
		}

		request->state = ofxStableDiffusionQueueState::Cancelled;
		request->completedTimeMicros = ofGetElapsedTimeMicros();
		queuedCount--;
	}

	ofLogNotice("ofxStableDiffusionQueue") << "Cancelled request #" << requestId;
	triggerAutoSave();
	return true;
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::cancelRequestsByTag(const std::string &tag) {
	int cancelledCount = 0;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		for (auto &pair : allRequests) {
			if (pair.second->tag == tag && pair.second->state == ofxStableDiffusionQueueState::Queued) {
				pair.second->state = ofxStableDiffusionQueueState::Cancelled;
				pair.second->completedTimeMicros = ofGetElapsedTimeMicros();
				cancelledCount++;
			}
		}

		if (cancelledCount > 0) {
			queuedCount -= cancelledCount;
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
	{
		std::lock_guard<std::mutex> lock(mutex_);
		for (auto &pair : allRequests) {
			if (pair.second->state == ofxStableDiffusionQueueState::Queued) {
				pair.second->state = ofxStableDiffusionQueueState::Cancelled;
				pair.second->completedTimeMicros = ofGetElapsedTimeMicros();
				cancelledCount++;
			}
		}

		if (cancelledCount > 0) {
			queuedCount -= cancelledCount;
		}
	}

	if (cancelledCount > 0) {
		ofLogNotice("ofxStableDiffusionQueue") << "Cancelled all " << cancelledCount << " queued requests";
		triggerAutoSave();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::clearHistory() {
	{
		std::lock_guard<std::mutex> lock(mutex_);
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
	}

	ofLogNotice("ofxStableDiffusionQueue") << "Cleared completed/failed/cancelled requests";
	triggerAutoSave();
}

//--------------------------------------------------------------
std::shared_ptr<ofxStableDiffusionQueueRequest> ofxStableDiffusionQueue::getNextRequest() {
	std::lock_guard<std::mutex> lock(mutex_);
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
	bool updated = false;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = allRequests.find(requestId);
		if (it != allRequests.end()) {
			if (it->second->state == ofxStableDiffusionQueueState::Queued) {
				queuedCount--;
			}
			it->second->state = ofxStableDiffusionQueueState::Processing;
			it->second->startedTimeMicros = ofGetElapsedTimeMicros();
			currentRequest = it->second;
			updated = true;
		}
	}

	if (updated) {
		ofLogNotice("ofxStableDiffusionQueue") << "Processing request #" << requestId;
		triggerAutoSave();
	}
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::markRequestCompleted(int requestId, const ofxStableDiffusionResult &result) {
	std::function<void(const ofxStableDiffusionResult &)> onComplete;
	float processingTimeSeconds = 0.0f;
	bool updated = false;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = allRequests.find(requestId);
		if (it != allRequests.end()) {
			it->second->state = ofxStableDiffusionQueueState::Completed;
			it->second->completedTimeMicros = ofGetElapsedTimeMicros();
			it->second->result = result;
			onComplete = it->second->onComplete;
			processingTimeSeconds = it->second->getProcessingTimeSeconds();

			if (currentRequest && currentRequest->requestId == requestId) {
				currentRequest = nullptr;
			}
			updated = true;
		}
	}

	if (!updated) {
		return;
	}
	if (onComplete) {
		onComplete(result);
	}

	ofLogNotice("ofxStableDiffusionQueue")
		<< "Completed request #" << requestId << " (processing time: " << processingTimeSeconds << "s)";
	triggerAutoSave();
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::markRequestFailed(int requestId, const std::string &errorMessage) {
	std::function<void(const std::string &)> onError;
	bool updated = false;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		auto it = allRequests.find(requestId);
		if (it != allRequests.end()) {
			it->second->state = ofxStableDiffusionQueueState::Failed;
			it->second->completedTimeMicros = ofGetElapsedTimeMicros();
			it->second->result.success = false;
			it->second->result.error = errorMessage;
			onError = it->second->onError;

			if (currentRequest && currentRequest->requestId == requestId) {
				currentRequest = nullptr;
			}
			updated = true;
		}
	}

	if (!updated) {
		return;
	}
	if (onError) {
		onError(errorMessage);
	}

	ofLogError("ofxStableDiffusionQueue") << "Failed request #" << requestId << ": " << errorMessage;
	triggerAutoSave();
}

//--------------------------------------------------------------
std::shared_ptr<ofxStableDiffusionQueueRequest> ofxStableDiffusionQueue::getRequest(int requestId) {
	std::lock_guard<std::mutex> lock(mutex_);
	auto it = allRequests.find(requestId);
	if (it != allRequests.end()) {
		return it->second;
	}
	return nullptr;
}

//--------------------------------------------------------------
std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>>
ofxStableDiffusionQueue::getRequestsByState(ofxStableDiffusionQueueState state) {
	std::lock_guard<std::mutex> lock(mutex_);
	std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>> requests;
	for (const auto &pair : allRequests) {
		if (pair.second->state == state) {
			requests.push_back(pair.second);
		}
	}
	return requests;
}

//--------------------------------------------------------------
std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>>
ofxStableDiffusionQueue::getRequestsByTag(const std::string &tag) {
	std::lock_guard<std::mutex> lock(mutex_);
	std::vector<std::shared_ptr<ofxStableDiffusionQueueRequest>> requests;
	for (const auto &pair : allRequests) {
		if (pair.second->tag == tag) {
			requests.push_back(pair.second);
		}
	}
	return requests;
}

//--------------------------------------------------------------
ofxStableDiffusionQueue::QueueStats ofxStableDiffusionQueue::getStats() const {
	std::lock_guard<std::mutex> lock(mutex_);
	QueueStats stats;
	stats.totalRequests = static_cast<int>(allRequests.size());

	float totalWaitTime = 0.0f;
	float totalProcessingTime = 0.0f;
	int completedCount = 0;

	for (const auto &pair : allRequests) {
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
	std::lock_guard<std::mutex> lock(mutex_);
	return queuedCount;
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::isEmpty() const { return getQueueSize() == 0; }

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::isProcessing() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return currentRequest != nullptr;
}

//--------------------------------------------------------------
std::shared_ptr<ofxStableDiffusionQueueRequest> ofxStableDiffusionQueue::getCurrentRequest() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return currentRequest;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setEnabled(bool enabled_) {
	{
		std::lock_guard<std::mutex> lock(mutex_);
		enabled = enabled_;
	}
	ofLogNotice("ofxStableDiffusionQueue") << "Queue " << (enabled_ ? "enabled" : "disabled");
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::isEnabled() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return enabled;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setMaxQueueSize(int size) {
	{
		std::lock_guard<std::mutex> lock(mutex_);
		maxQueueSize = size;
	}
	ofLogNotice("ofxStableDiffusionQueue")
		<< "Max queue size set to: " << (size == 0 ? "unlimited" : std::to_string(size));
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::getMaxQueueSize() const {
	std::lock_guard<std::mutex> lock(mutex_);
	return maxQueueSize;
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::saveToFile(const std::string &filepath) {
	ofJson json;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		json["version"] = "1.0";
		json["timestamp"] = ofGetElapsedTimeMicros();
		json["nextRequestId"] = nextRequestId.load(std::memory_order_relaxed);

		ofJson requestsJson = ofJson::array();
		for (const auto &pair : allRequests) {
			ofJson reqJson;
			reqJson["requestId"] = pair.second->requestId;
			reqJson["priority"] = static_cast<int>(pair.second->priority);
			reqJson["state"] = static_cast<int>(pair.second->state);
			reqJson["taskType"] = static_cast<int>(pair.second->taskType);
			reqJson["tag"] = pair.second->tag;
			reqJson["queuedTime"] = pair.second->queuedTimeMicros;
			reqJson["startedTime"] = pair.second->startedTimeMicros;
			reqJson["completedTime"] = pair.second->completedTimeMicros;
			requestsJson.push_back(reqJson);
		}
		json["requests"] = requestsJson;
	}

	return ofSaveJson(filepath, json);
}

//--------------------------------------------------------------
bool ofxStableDiffusionQueue::loadFromFile(const std::string &filepath) {
	ofJson json = ofLoadJson(filepath);
	if (json.empty()) {
		ofLogError("ofxStableDiffusionQueue") << "Failed to load queue from: " << filepath;
		return false;
	}

	{
		std::lock_guard<std::mutex> lock(mutex_);
		nextRequestId.store(json.value("nextRequestId", 1), std::memory_order_relaxed);
	}

	ofLogNotice("ofxStableDiffusionQueue") << "Loaded queue state from: " << filepath;
	return true;
}

//--------------------------------------------------------------
void ofxStableDiffusionQueue::setAutoSave(bool enabled_, const std::string &filepath) {
	{
		std::lock_guard<std::mutex> lock(mutex_);
		autoSaveEnabled = enabled_;
		autoSaveFilepath = filepath;
	}

	if (enabled_ && !filepath.empty()) {
		ofLogNotice("ofxStableDiffusionQueue") << "Auto-save enabled: " << filepath;
	}
}

//--------------------------------------------------------------
int ofxStableDiffusionQueue::generateRequestId() { return nextRequestId.fetch_add(1, std::memory_order_relaxed); }

//--------------------------------------------------------------
void ofxStableDiffusionQueue::triggerAutoSave() {
	bool enabledSnapshot = false;
	std::string filepath;
	{
		std::lock_guard<std::mutex> lock(mutex_);
		enabledSnapshot = autoSaveEnabled;
		filepath = autoSaveFilepath;
	}
	if (enabledSnapshot && !filepath.empty()) {
		saveToFile(filepath);
	}
}

//--------------------------------------------------------------
std::shared_ptr<ofxStableDiffusionQueueRequest>
ofxStableDiffusionQueue::createRequest(ofxStableDiffusionTask taskType, ofxStableDiffusionPriority priority,
									   const std::string &tag) {
	auto request = std::make_shared<ofxStableDiffusionQueueRequest>();
	request->requestId = generateRequestId();
	request->priority = priority;
	request->state = ofxStableDiffusionQueueState::Queued;
	request->taskType = taskType;
	request->tag = tag;
	request->queuedTimeMicros = ofGetElapsedTimeMicros();
	return request;
}
