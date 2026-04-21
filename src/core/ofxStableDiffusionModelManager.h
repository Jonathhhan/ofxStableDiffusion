#pragma once

#include "ofMain.h"
#include "ofxStableDiffusionEnums.h"
#include "stable-diffusion.h"

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <functional>

/// Model metadata information
struct ofxStableDiffusionModelInfo {
	std::string modelPath;
	std::string modelName;
	std::string modelType;  // e.g., "SD1.5", "SDXL", "SD-Turbo"
	uint64_t fileSizeBytes = 0;
	uint64_t estimatedMemoryBytes = 0;
	int nativeWidth = 512;
	int nativeHeight = 512;
	bool isLoaded = false;
	bool isValid = false;
	uint64_t lastAccessTimeMicros = 0;
	uint64_t loadTimeMicros = 0;
	std::string errorMessage;

	// Additional model info
	std::string vaePath;
	std::string taesdPath;
	std::string controlNetPath;
	std::string loraModelDir;
	sd_type_t weightType = SD_TYPE_F16;

	bool hasVAE() const {
		return !vaePath.empty();
	}

	bool hasControlNet() const {
		return !controlNetPath.empty();
	}
};

/// Model cache entry
struct ofxStableDiffusionModelCacheEntry {
	ofxStableDiffusionModelInfo info;
	sd_ctx_t* sdCtx = nullptr;
	uint64_t lastUsedTimeMicros = 0;
	int referenceCount = 0;
};

/// Model loading progress callback
using ofxModelLoadProgressCallback = std::function<void(const std::string& modelPath, float progress, const std::string& stage)>;

/// Model Manager for preloading and caching models
class ofxStableDiffusionModelManager {
public:
	ofxStableDiffusionModelManager();
	~ofxStableDiffusionModelManager();

	/// Set maximum cache size in bytes (0 = unlimited)
	void setMaxCacheSize(uint64_t bytes);

	/// Get current cache size in bytes
	uint64_t getCurrentCacheSize() const;

	/// Set maximum number of cached models (0 = unlimited)
	void setMaxCachedModels(int count);

	/// Get number of currently cached models
	int getCachedModelCount() const;

	/// Scan a directory for available models
	std::vector<ofxStableDiffusionModelInfo> scanModelsInDirectory(const std::string& directory);

	/// Extract metadata from a model file
	ofxStableDiffusionModelInfo extractModelInfo(const std::string& modelPath);

	/// Validate a model file
	bool validateModel(const std::string& modelPath, std::string& errorMessage);

	/// Preload a model into cache
	bool preloadModel(const ofxStableDiffusionModelInfo& modelInfo, std::string& errorMessage);

	/// Get a model context (loads if not cached)
	sd_ctx_t* getModelContext(const std::string& modelPath,
		const ofxStableDiffusionModelInfo& modelInfo,
		std::string& errorMessage);

	/// Release a model context (decrements reference count)
	void releaseModelContext(const std::string& modelPath);

	/// Unload a specific model from cache
	bool unloadModel(const std::string& modelPath);

	/// Clear all cached models
	void clearCache();

	/// Get info for a cached model
	const ofxStableDiffusionModelInfo* getCachedModelInfo(const std::string& modelPath) const;

	/// Get all cached model info
	std::vector<ofxStableDiffusionModelInfo> getCachedModels() const;

	/// Check if a model is currently loaded
	bool isModelLoaded(const std::string& modelPath) const;

	/// Set model loading progress callback
	void setProgressCallback(ofxModelLoadProgressCallback cb);

	/// Enable/disable automatic cache eviction (LRU)
	void setAutoEviction(bool enabled);

	/// Get cache statistics
	struct CacheStats {
		int totalModels = 0;
		int loadedModels = 0;
		uint64_t totalMemoryBytes = 0;
		uint64_t availableMemoryBytes = 0;
		int cacheHits = 0;
		int cacheMisses = 0;
	};
	CacheStats getCacheStats() const;

private:
	std::map<std::string, ofxStableDiffusionModelCacheEntry> modelCache;
	uint64_t maxCacheSizeBytes = 0;  // 0 = unlimited
	int maxCachedModels = 0;  // 0 = unlimited
	bool autoEvictionEnabled = true;
	ofxModelLoadProgressCallback progressCallback;

	// Statistics
	mutable int cacheHits = 0;
	mutable int cacheMisses = 0;

	// Internal methods
	bool evictLRUModel();
	uint64_t calculateCacheSize() const;
	void updateAccessTime(const std::string& modelPath);
	sd_ctx_t* loadModelContext(const ofxStableDiffusionModelInfo& modelInfo, std::string& errorMessage);
	void reportProgress(const std::string& modelPath, float progress, const std::string& stage);
	uint64_t estimateModelMemory(const std::string& modelPath) const;
	std::string extractModelType(const std::string& modelPath) const;
	bool isValidModelFile(const std::string& modelPath) const;
};

