#include "core/ofxStableDiffusionRankingHelpers.h"

#include <cstdlib>
#include <iostream>
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

	ok &= expect(std::string(ofxStableDiffusionImageSelectionModeName(
		ofxStableDiffusionImageSelectionMode::KeepOrder)) == "KeepOrder",
		"keep-order label");
	ok &= expect(std::string(ofxStableDiffusionImageSelectionModeName(
		ofxStableDiffusionImageSelectionMode::Rerank)) == "Rerank",
		"rerank label");
	ok &= expect(std::string(ofxStableDiffusionImageSelectionModeName(
		ofxStableDiffusionImageSelectionMode::BestOnly)) == "BestOnly",
		"best-only label");

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

	return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
