#pragma once

#include "../core/ofxStableDiffusionTypes.h"
#include "../core/ofxStableDiffusionRankingHelpers.h"
#include "ofMain.h"

#include <functional>
#include <string>
#include <vector>

/**
 * @file ofxStableDiffusionGgmlBridge.h
 * @brief Integration helpers for connecting ofxStableDiffusion with ofxGgml
 *
 * This bridge provides utilities for:
 * - CLIP-based image ranking and scoring
 * - Embedding extraction and comparison
 * - Cross-addon data transfer
 * - Batch processing optimization
 *
 * Architecture:
 * - ofxStableDiffusion generates images
 * - ofxGgml provides CLIP inference for scoring
 * - This bridge facilitates clean communication without runtime coupling
 */

// Forward declarations for ofxGgml types (users will include ofxGgml separately)
// These are interface types that ofxGgml should implement

/**
 * @brief Interface for CLIP scoring provided by ofxGgml
 *
 * Users should implement this interface wrapping their ofxGgmlClipInference instance
 */
class ofxStableDiffusionClipScorer {
public:
	virtual ~ofxStableDiffusionClipScorer() = default;

	/**
	 * @brief Score an image against a text prompt
	 * @param pixels Image to score
	 * @param text Text prompt to compare against
	 * @return Similarity score (typically 0-1, higher is better)
	 */
	virtual float scoreImage(const ofPixels& pixels, const std::string& text) = 0;

	/**
	 * @brief Batch score multiple images against a single text prompt
	 * @param images Vector of images to score
	 * @param text Text prompt to compare against
	 * @return Vector of scores, one per image
	 */
	virtual std::vector<float> scoreImages(
		const std::vector<ofPixels>& images,
		const std::string& text) = 0;

	/**
	 * @brief Extract CLIP embedding for an image
	 * @param pixels Image to encode
	 * @return Embedding vector
	 */
	virtual std::vector<float> encodeImage(const ofPixels& pixels) = 0;

	/**
	 * @brief Extract CLIP embedding for text
	 * @param text Text to encode
	 * @return Embedding vector
	 */
	virtual std::vector<float> encodeText(const std::string& text) = 0;

	/**
	 * @brief Get the name/version of the CLIP model
	 * @return Model identifier string
	 */
	virtual std::string getModelName() const = 0;
};

/**
 * @brief Configuration for CLIP-based ranking
 */
struct ofxStableDiffusionClipRankingConfig {
	std::string targetPrompt;           // Prompt to rank against
	bool useTargetPrompt = true;        // If false, use original generation prompt
	float promptWeight = 1.0f;          // Weight for text-image similarity
	float aestheticWeight = 0.0f;       // Weight for aesthetic quality (if supported)
	bool normalizeScores = true;        // Normalize scores to 0-1 range
	bool includeMetadata = true;        // Include detailed scoring metadata
	int maxCandidates = -1;             // Limit scoring to top N candidates (-1 = all)
};

/**
 * @brief Result of CLIP-based ranking
 */
struct ofxStableDiffusionClipRankingResult {
	std::vector<ofxStableDiffusionImageScore> scores;
	std::vector<int> rankedIndices;     // Indices sorted by score (best first)
	int bestImageIndex = -1;            // Index of highest-scoring image
	float averageScore = 0.0f;
	float bestScore = 0.0f;
	float worstScore = 0.0f;
	std::string clipModel;
	bool success = false;
	std::string error;
};

/**
 * @brief Image data prepared for CLIP scoring
 */
struct ofxStableDiffusionClipImageBatch {
	std::vector<ofPixels> images;
	std::vector<int> sourceIndices;     // Map back to original frame indices
	std::string prompt;
	std::string negativePrompt;

	void addFrame(const ofxStableDiffusionImageFrame& frame) {
		images.push_back(frame.pixels);
		sourceIndices.push_back(frame.index);
	}

	void clear() {
		images.clear();
		sourceIndices.clear();
	}

	size_t size() const {
		return images.size();
	}
};

/**
 * @brief Embedding comparison metrics
 */
struct ofxStableDiffusionEmbeddingComparison {
	float cosineSimilarity = 0.0f;
	float euclideanDistance = 0.0f;
	float dotProduct = 0.0f;

	std::string summary() const {
		return "Cosine: " + std::to_string(cosineSimilarity) +
			", Euclidean: " + std::to_string(euclideanDistance) +
			", Dot: " + std::to_string(dotProduct);
	}
};

namespace ofxStableDiffusionGgmlBridge {

/**
 * @brief Create a ranking callback using a CLIP scorer
 *
 * This creates a callback that can be passed to setImageRankCallback()
 *
 * @param scorer CLIP scoring implementation (typically from ofxGgml)
 * @param config Ranking configuration
 * @return Callback function compatible with ofxStableDiffusion API
 */
inline std::function<std::vector<ofxStableDiffusionImageScore>(
	const ofxStableDiffusionImageRequest&,
	const std::vector<ofxStableDiffusionImageFrame>&)>
createClipRankingCallback(
	std::shared_ptr<ofxStableDiffusionClipScorer> scorer,
	const ofxStableDiffusionClipRankingConfig& config = {}) {

	return [scorer, config](
		const ofxStableDiffusionImageRequest& request,
		const std::vector<ofxStableDiffusionImageFrame>& frames)
		-> std::vector<ofxStableDiffusionImageScore> {

		std::vector<ofxStableDiffusionImageScore> scores;

		if (!scorer || frames.empty()) {
			// Return empty scores on error
			return std::vector<ofxStableDiffusionImageScore>(frames.size());
		}

		try {
			// Determine which prompt to use
			std::string targetPrompt = config.useTargetPrompt && !config.targetPrompt.empty()
				? config.targetPrompt
				: request.prompt;

			// Prepare images for batch scoring
			std::vector<ofPixels> images;
			for (const auto& frame : frames) {
				images.push_back(frame.pixels);
			}

			// Score all images
			std::vector<float> rawScores = scorer->scoreImages(images, targetPrompt);

			// Normalize if requested
			if (config.normalizeScores && !rawScores.empty()) {
				float minScore = *std::min_element(rawScores.begin(), rawScores.end());
				float maxScore = *std::max_element(rawScores.begin(), rawScores.end());
				float range = maxScore - minScore;

				if (range > 0.001f) {
					for (auto& score : rawScores) {
						score = (score - minScore) / range;
					}
				}
			}

			// Build result scores
			for (size_t i = 0; i < frames.size(); ++i) {
				ofxStableDiffusionImageScore score;
				score.valid = i < rawScores.size();
				score.score = i < rawScores.size() ? rawScores[i] : 0.0f;
				score.scorer = scorer->getModelName();

				if (config.includeMetadata) {
					score.metadata.push_back({"prompt", targetPrompt});
					score.metadata.push_back({"raw_score", std::to_string(rawScores[i])});
					score.metadata.push_back({"frame_index", std::to_string(frames[i].index)});
				}

				score.summary = "CLIP score: " + std::to_string(score.score);
				scores.push_back(score);
			}

		} catch (const std::exception& e) {
			ofLogError("ofxStableDiffusionGgmlBridge") << "CLIP ranking failed: " << e.what();
			// Return empty scores on exception
			return std::vector<ofxStableDiffusionImageScore>(frames.size());
		}

		return scores;
	};
}

/**
 * @brief Rank images using CLIP scoring
 *
 * Direct ranking without needing the callback mechanism
 *
 * @param scorer CLIP scoring implementation
 * @param frames Images to rank
 * @param prompt Prompt to rank against
 * @param config Ranking configuration
 * @return Ranking result with scores and sorted indices
 */
inline ofxStableDiffusionClipRankingResult rankImagesWithClip(
	std::shared_ptr<ofxStableDiffusionClipScorer> scorer,
	const std::vector<ofxStableDiffusionImageFrame>& frames,
	const std::string& prompt,
	const ofxStableDiffusionClipRankingConfig& config = {}) {

	ofxStableDiffusionClipRankingResult result;

	if (!scorer || frames.empty()) {
		result.error = "Invalid scorer or empty frames";
		return result;
	}

	try {
		result.clipModel = scorer->getModelName();

		// Prepare images
		std::vector<ofPixels> images;
		for (const auto& frame : frames) {
			images.push_back(frame.pixels);
		}

		// Score images
		std::vector<float> rawScores = scorer->scoreImages(images, prompt);

		if (rawScores.size() != images.size()) {
			result.error = "Score count mismatch";
			return result;
		}

		// Calculate statistics
		result.averageScore = 0.0f;
		result.bestScore = rawScores[0];
		result.worstScore = rawScores[0];

		for (float score : rawScores) {
			result.averageScore += score;
			result.bestScore = std::max(result.bestScore, score);
			result.worstScore = std::min(result.worstScore, score);
		}
		result.averageScore /= rawScores.size();

		// Normalize if requested
		std::vector<float> finalScores = rawScores;
		if (config.normalizeScores) {
			float range = result.bestScore - result.worstScore;
			if (range > 0.001f) {
				for (auto& score : finalScores) {
					score = (score - result.worstScore) / range;
				}
			}
		}

		// Build image scores
		for (size_t i = 0; i < frames.size(); ++i) {
			ofxStableDiffusionImageScore imageScore;
			imageScore.valid = true;
			imageScore.score = finalScores[i];
			imageScore.scorer = result.clipModel;
			imageScore.summary = "CLIP: " + std::to_string(finalScores[i]);

			if (config.includeMetadata) {
				imageScore.metadata.push_back({"prompt", prompt});
				imageScore.metadata.push_back({"raw_score", std::to_string(rawScores[i])});
				imageScore.metadata.push_back({"normalized_score", std::to_string(finalScores[i])});
			}

			result.scores.push_back(imageScore);
		}

		// Build ranked indices
		result.rankedIndices = ofxStableDiffusionBuildRankedImageOrder(result.scores);
		result.bestImageIndex = result.rankedIndices.empty() ? -1 : result.rankedIndices[0];
		result.success = true;

	} catch (const std::exception& e) {
		result.error = std::string("CLIP ranking exception: ") + e.what();
		result.success = false;
	}

	return result;
}

/**
 * @brief Prepare image batch for efficient CLIP processing
 *
 * @param frames Source frames
 * @param maxBatchSize Maximum images per batch
 * @return Batches ready for CLIP scoring
 */
inline std::vector<ofxStableDiffusionClipImageBatch> prepareBatches(
	const std::vector<ofxStableDiffusionImageFrame>& frames,
	int maxBatchSize = 16) {

	std::vector<ofxStableDiffusionClipImageBatch> batches;

	if (frames.empty()) {
		return batches;
	}

	ofxStableDiffusionClipImageBatch currentBatch;
	currentBatch.prompt = frames[0].generation.prompt;
	currentBatch.negativePrompt = frames[0].generation.negativePrompt;

	for (const auto& frame : frames) {
		if (static_cast<int>(currentBatch.size()) >= maxBatchSize) {
			batches.push_back(currentBatch);
			currentBatch.clear();
			currentBatch.prompt = frame.generation.prompt;
			currentBatch.negativePrompt = frame.generation.negativePrompt;
		}

		currentBatch.addFrame(frame);
	}

	if (currentBatch.size() > 0) {
		batches.push_back(currentBatch);
	}

	return batches;
}

/**
 * @brief Compare two embeddings
 *
 * @param embedding1 First embedding vector
 * @param embedding2 Second embedding vector
 * @return Comparison metrics
 */
inline ofxStableDiffusionEmbeddingComparison compareEmbeddings(
	const std::vector<float>& embedding1,
	const std::vector<float>& embedding2) {

	ofxStableDiffusionEmbeddingComparison comparison;

	if (embedding1.size() != embedding2.size() || embedding1.empty()) {
		return comparison;
	}

	// Cosine similarity
	float dot = 0.0f;
	float norm1 = 0.0f;
	float norm2 = 0.0f;

	for (size_t i = 0; i < embedding1.size(); ++i) {
		dot += embedding1[i] * embedding2[i];
		norm1 += embedding1[i] * embedding1[i];
		norm2 += embedding2[i] * embedding2[i];
	}

	norm1 = std::sqrt(norm1);
	norm2 = std::sqrt(norm2);

	if (norm1 > 0.0f && norm2 > 0.0f) {
		comparison.cosineSimilarity = dot / (norm1 * norm2);
	}

	comparison.dotProduct = dot;

	// Euclidean distance
	float sumSquaredDiff = 0.0f;
	for (size_t i = 0; i < embedding1.size(); ++i) {
		float diff = embedding1[i] - embedding2[i];
		sumSquaredDiff += diff * diff;
	}
	comparison.euclideanDistance = std::sqrt(sumSquaredDiff);

	return comparison;
}

/**
 * @brief Find most similar image to a target embedding
 *
 * @param scorer CLIP scorer for encoding images
 * @param frames Images to search
 * @param targetEmbedding Target embedding to match
 * @return Index of most similar image, or -1 if error
 */
inline int findMostSimilarImage(
	std::shared_ptr<ofxStableDiffusionClipScorer> scorer,
	const std::vector<ofxStableDiffusionImageFrame>& frames,
	const std::vector<float>& targetEmbedding) {

	if (!scorer || frames.empty() || targetEmbedding.empty()) {
		return -1;
	}

	int bestIndex = -1;
	float bestSimilarity = -1.0f;

	for (size_t i = 0; i < frames.size(); ++i) {
		try {
			std::vector<float> imageEmbedding = scorer->encodeImage(frames[i].pixels);
			ofxStableDiffusionEmbeddingComparison comp = compareEmbeddings(
				imageEmbedding, targetEmbedding);

			if (comp.cosineSimilarity > bestSimilarity) {
				bestSimilarity = comp.cosineSimilarity;
				bestIndex = static_cast<int>(i);
			}
		} catch (const std::exception& e) {
			ofLogWarning("ofxStableDiffusionGgmlBridge")
				<< "Failed to encode image " << i << ": " << e.what();
		}
	}

	return bestIndex;
}

/**
 * @brief Create aesthetic scorer callback (if ofxGgml provides aesthetic model)
 *
 * @param scorer CLIP scorer (should support aesthetic scoring if available)
 * @return Callback for aesthetic-based ranking
 */
inline std::function<std::vector<ofxStableDiffusionImageScore>(
	const ofxStableDiffusionImageRequest&,
	const std::vector<ofxStableDiffusionImageFrame>&)>
createAestheticRankingCallback(
	std::shared_ptr<ofxStableDiffusionClipScorer> scorer) {

	return [scorer](
		const ofxStableDiffusionImageRequest& request,
		const std::vector<ofxStableDiffusionImageFrame>& frames)
		-> std::vector<ofxStableDiffusionImageScore> {

		std::vector<ofxStableDiffusionImageScore> scores;

		// This is a placeholder for aesthetic scoring
		// Users should implement aesthetic model scoring in their ofxGgml integration
		// For now, we can use generic CLIP scoring as a proxy

		if (!scorer || frames.empty()) {
			return std::vector<ofxStableDiffusionImageScore>(frames.size());
		}

		// Use "beautiful, high quality, aesthetic" as proxy aesthetic prompt
		std::string aestheticPrompt = "beautiful, high quality, aesthetic, artistic";

		std::vector<ofPixels> images;
		for (const auto& frame : frames) {
			images.push_back(frame.pixels);
		}

		std::vector<float> rawScores = scorer->scoreImages(images, aestheticPrompt);

		for (size_t i = 0; i < frames.size(); ++i) {
			ofxStableDiffusionImageScore score;
			score.valid = i < rawScores.size();
			score.score = i < rawScores.size() ? rawScores[i] : 0.0f;
			score.scorer = scorer->getModelName() + " (aesthetic proxy)";
			score.summary = "Aesthetic score: " + std::to_string(score.score);
			scores.push_back(score);
		}

		return scores;
	};
}

} // namespace ofxStableDiffusionGgmlBridge
