#include "core/ofxStableDiffusionRankingHelpers.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const std::string & message) {
	if (condition) {
		return true;
	}

	std::cerr << "FAIL: " << message << std::endl;
	return false;
}

} // namespace

int main() {
	bool ok = true;

	// Test ofxStableDiffusionImageSelectionModeName - all modes
	ok &= expect(std::string(ofxStableDiffusionImageSelectionModeName(
		ofxStableDiffusionImageSelectionMode::KeepOrder)) == "KeepOrder",
		"keep-order label");
	ok &= expect(std::string(ofxStableDiffusionImageSelectionModeName(
		ofxStableDiffusionImageSelectionMode::Rerank)) == "Rerank",
		"rerank label");
	ok &= expect(std::string(ofxStableDiffusionImageSelectionModeName(
		ofxStableDiffusionImageSelectionMode::BestOnly)) == "BestOnly",
		"best-only label");

	// Test basic ranking
	{
		std::vector<ofxStableDiffusionImageScore> scores(4);
		scores[0].valid = true;
		scores[0].score = 0.25f;
		scores[1].valid = true;
		scores[1].score = 0.90f;
		scores[2].valid = false;
		scores[3].valid = true;
		scores[3].score = 0.50f;

		const std::vector<std::size_t> order =
			ofxStableDiffusionBuildRankedImageOrder(scores);
		ok &= expect(order.size() == 4, "ranking order size");
		ok &= expect(order[0] == 1, "highest score first");
		ok &= expect(order[1] == 3, "second highest score second");
		ok &= expect(order[2] == 0, "third highest score third");
		ok &= expect(order[3] == 2, "invalid score last");
	}

	// Test empty scores
	{
		std::vector<ofxStableDiffusionImageScore> emptyScores;
		const std::vector<std::size_t> order =
			ofxStableDiffusionBuildRankedImageOrder(emptyScores);
		ok &= expect(order.empty(), "empty scores produces empty order");
	}

	// Test single score
	{
		std::vector<ofxStableDiffusionImageScore> singleScore(1);
		singleScore[0].valid = true;
		singleScore[0].score = 0.75f;

		const std::vector<std::size_t> order =
			ofxStableDiffusionBuildRankedImageOrder(singleScore);
		ok &= expect(order.size() == 1, "single score order size");
		ok &= expect(order[0] == 0, "single score index");
	}

	// Test all invalid scores
	{
		std::vector<ofxStableDiffusionImageScore> invalidScores(3);
		invalidScores[0].valid = false;
		invalidScores[1].valid = false;
		invalidScores[2].valid = false;

		const std::vector<std::size_t> order =
			ofxStableDiffusionBuildRankedImageOrder(invalidScores);
		ok &= expect(order.size() == 3, "all invalid scores order size");
		// All invalid scores should maintain original order (stable sort)
		ok &= expect(order[0] == 0, "first invalid maintains position");
		ok &= expect(order[1] == 1, "second invalid maintains position");
		ok &= expect(order[2] == 2, "third invalid maintains position");
	}

	// Test identical scores (stable sort should preserve original order)
	{
		std::vector<ofxStableDiffusionImageScore> identicalScores(4);
		identicalScores[0].valid = true;
		identicalScores[0].score = 0.5f;
		identicalScores[1].valid = true;
		identicalScores[1].score = 0.5f;
		identicalScores[2].valid = true;
		identicalScores[2].score = 0.5f;
		identicalScores[3].valid = true;
		identicalScores[3].score = 0.5f;

		const std::vector<std::size_t> order =
			ofxStableDiffusionBuildRankedImageOrder(identicalScores);
		ok &= expect(order.size() == 4, "identical scores order size");
		// Stable sort should preserve original order when scores are equal
		ok &= expect(order[0] == 0, "identical score 0 position");
		ok &= expect(order[1] == 1, "identical score 1 position");
		ok &= expect(order[2] == 2, "identical score 2 position");
		ok &= expect(order[3] == 3, "identical score 3 position");
	}

	// Test mix of valid and invalid with same valid scores
	{
		std::vector<ofxStableDiffusionImageScore> mixedScores(5);
		mixedScores[0].valid = true;
		mixedScores[0].score = 0.5f;
		mixedScores[1].valid = false;
		mixedScores[2].valid = true;
		mixedScores[2].score = 0.5f;
		mixedScores[3].valid = false;
		mixedScores[4].valid = true;
		mixedScores[4].score = 0.5f;

		const std::vector<std::size_t> order =
			ofxStableDiffusionBuildRankedImageOrder(mixedScores);
		ok &= expect(order.size() == 5, "mixed scores order size");
		// Valid scores with same value should come first in original order
		ok &= expect(order[0] == 0, "first valid 0.5 score");
		ok &= expect(order[1] == 2, "second valid 0.5 score");
		ok &= expect(order[2] == 4, "third valid 0.5 score");
		// Invalid scores should come last in original order
		ok &= expect(order[3] == 1, "first invalid score");
		ok &= expect(order[4] == 3, "second invalid score");
	}

	// Test extreme scores
	{
		std::vector<ofxStableDiffusionImageScore> extremeScores(3);
		extremeScores[0].valid = true;
		extremeScores[0].score = 0.0f;
		extremeScores[1].valid = true;
		extremeScores[1].score = 1.0f;
		extremeScores[2].valid = true;
		extremeScores[2].score = 0.5f;

		const std::vector<std::size_t> order =
			ofxStableDiffusionBuildRankedImageOrder(extremeScores);
		ok &= expect(order[0] == 1, "highest score (1.0) first");
		ok &= expect(order[1] == 2, "middle score (0.5) second");
		ok &= expect(order[2] == 0, "lowest score (0.0) last");
	}

	// Test negative scores (if supported)
	{
		std::vector<ofxStableDiffusionImageScore> negativeScores(3);
		negativeScores[0].valid = true;
		negativeScores[0].score = -0.5f;
		negativeScores[1].valid = true;
		negativeScores[1].score = 0.5f;
		negativeScores[2].valid = true;
		negativeScores[2].score = -1.0f;

		const std::vector<std::size_t> order =
			ofxStableDiffusionBuildRankedImageOrder(negativeScores);
		ok &= expect(order[0] == 1, "positive score first");
		ok &= expect(order[1] == 0, "less negative score second");
		ok &= expect(order[2] == 2, "most negative score last");
	}

	// Test ofxStableDiffusionImageScore metadata
	{
		ofxStableDiffusionImageScore score;
		ok &= expect(!score.valid, "default score invalid");
		ok &= expect(score.score == 0.0f, "default score value is 0");
		ok &= expect(score.scorer.empty(), "default scorer empty");
		ok &= expect(score.summary.empty(), "default summary empty");
		ok &= expect(score.metadata.empty(), "default metadata empty");

		score.valid = true;
		score.score = 0.75f;
		score.scorer = "test_scorer";
		score.summary = "test summary";
		score.metadata.push_back(std::make_pair("key1", "value1"));
		score.metadata.push_back(std::make_pair("key2", "value2"));

		ok &= expect(score.valid, "score set to valid");
		ok &= expect(score.score == 0.75f, "score value set");
		ok &= expect(score.scorer == "test_scorer", "scorer set");
		ok &= expect(score.summary == "test summary", "summary set");
		ok &= expect(score.metadata.size() == 2, "metadata size");
		ok &= expect(score.metadata[0].first == "key1", "metadata key 1");
		ok &= expect(score.metadata[0].second == "value1", "metadata value 1");
		ok &= expect(score.metadata[1].first == "key2", "metadata key 2");
		ok &= expect(score.metadata[1].second == "value2", "metadata value 2");
	}

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
